#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GNN + PPO + Embedded WGF Pilot Assignment (Martingale-SNC DVP) — self-contained, run directly in VSCode.

What this file does (end-to-end):
  1) Fix system parameters below (CF-mMIMO, FZF proxy stats, Martingale-SNC DVP).
  2) Policy (GNN) outputs serving AP set A(k) for each user k (size S).
  3) Environment embeds WGF pilot assignment using A(k) and large-scale fading Beta.
  4) Environment computes per-user Martingale-SNC DVP bound and weighted-sum-rate (WSR).
  5) PPO trains the GNN policy to maximize WSR subject to average DVP constraint.
  6) Saves reward curve (reward_curve.png) and prints trained WSR.

Notes:
  - This implementation mirrors the structure of your dvp_evaluation code: WGF construction,
    C_est/Eta computation, Gamma-quantile service samples, brentq solve for theta, then DVP.
  - DVP evaluation is computationally heavy (root finding per UE). For debugging, reduce K/M,
    quantiles, or brentq_maxiter. For paper-grade evaluation, increase them.

Requires: numpy, torch, scipy, matplotlib
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.special import gammaincinv, erfcinv
from scipy.optimize import brentq
import matplotlib.pyplot as plt


# =============================================================================
# 0) System parameters (EDIT HERE)
# =============================================================================
CFG = dict(
    # --- Topology / size ---
    M_ap=60,          # number of APs
    K=30,             # number of users
    S=10,             # serving APs per user (|A(k)|)
    lp=4,             # tau: number of pilots

    # --- PHY / URLLC ---
    N_ant=16,         # antennas per AP (N)
    n=200,            # blocklength (channel uses)
    B=2e6,            # bandwidth (Hz)
    epsilon_dec=1e-5, # decoding error probability for FBL rate (Q^{-1}(epsilon_dec))

    # --- Delay constraints (all users identical as you specified) ---
    d_th_time=1e-3,   # delay threshold in seconds (set to your value, e.g., 3e-4 / 8e-4 / 1e-3 / 2e-3)
    eps_target_sys=1e-5,  # target average DVP constraint threshold

    # --- Arrival model used in your dvp evaluation: mu_arrival = rho_target * mean_service ---
    rho_target=0.9905,

    # --- Large-scale fading model (same style as your evaluation script) ---
    area=500.0,
    d0=36.0,
    alpha_path=3.6,

    # --- Power / noise (same style as your evaluation script) ---
    noise_figure_db=5.0,
    P_AP_W=1.0,
    rho_p=10 ** (10 / 10),  # pilot power / noise (linear)

    # --- Martingale-SNC numerical knobs ---
    num_quantiles=20,    # set 50 to match your evaluation script; use 20-30 for training speed
    theta_hi=50.0,       # brentq upper bound
    brentq_maxiter=80,  # brentq iterations

    # --- Reward scaling / penalties ---
    R_ref_bps=1e6,          # normalize WSR
    infeasible_penalty=10.0,

    # --- Random seed ---
    seed=41,
)

# =============================================================================
# 1) PPO hyperparameters (EDIT HERE)
# =============================================================================
PPO_HYPER = dict(
    device="cuda",         # GPU enabled
    total_updates=100,
    batch_episodes=128,    # 大幅增加以充分利用 GPU
    update_epochs=4,
    minibatch_size=32,     # 增加 minibatch 以加速 GPU 计算
    lr=3e-4,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    # Lagrange multiplier for avg DVP constraint
    lambda_init=1.0,
    lambda_lr=0.05,
    lambda_max=100.0,
)

# =============================================================================
# 2) Utilities
# =============================================================================
def set_seed(seed: int, torch_deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def qinv(eps: float) -> float:
    return math.sqrt(2.0) * float(erfcinv(2.0 * eps))


def safe_logmeanexp(x: np.ndarray) -> float:
    m = float(np.max(x))
    return m + math.log(float(np.mean(np.exp(x - m))))


# =============================================================================
# 3) WGF pilot assignment (embedded in environment)
# =============================================================================
def wgf_pilot_assignment(Beta_mat: np.ndarray, ServingAP: List[np.ndarray], lp_val: int) -> np.ndarray:
    M, K = Beta_mat.shape
    pilots = np.zeros(K, dtype=int)

    # S(k,j) = sum_{m in A(k)} Beta(m,j)
    S = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        Mk = ServingAP[k]
        S[k, :] = np.sum(Beta_mat[Mk, :], axis=0)

    # symmetric weights omega_{k,j}
    omega = np.zeros((K, K), dtype=np.float64)
    eps = np.finfo(float).eps
    for k in range(K):
        denom_k = S[k, k] if S[k, k] > 0 else eps
        for j in range(k + 1, K):
            denom_j = S[j, j] if S[j, j] > 0 else eps
            w = (S[k, j] / denom_k) ** 2 + (S[j, k] / denom_j) ** 2
            omega[k, j] = omega[j, k] = w

    # sort by weighted degree
    deg = np.sum(omega, axis=1)
    sorted_users = np.argsort(-deg)

    # seeds
    groups = [[] for _ in range(lp_val)]
    for p in range(lp_val):
        u = int(sorted_users[p])
        pilots[u] = p
        groups[p] = [u]

    # assign remaining
    for idx in range(lp_val, K):
        u = int(sorted_users[idx])
        costs = np.zeros(lp_val, dtype=np.float64)
        for p in range(lp_val):
            costs[p] = 0.0 if len(groups[p]) == 0 else float(np.sum(omega[u, groups[p]]))
        best_p = int(np.argmin(costs))
        pilots[u] = best_p
        groups[best_p].append(u)

    return pilots


# =============================================================================
# 4) Martingale-SNC components (your evaluation logic)
# =============================================================================
def get_quantiles(alpha2: float, beta2: float, C_val: float, n: int, delta_fbl: float, num_q: int):
    p = (np.arange(1, num_q + 1) - 0.5) / num_q
    I = gammaincinv(alpha2, p) * beta2
    R = np.maximum(0.0, n * np.log2(1.0 + C_val / I) - n * delta_fbl)
    return I, R


def get_log_Ms(theta: float, R_samp: np.ndarray) -> float:
    return safe_logmeanexp(-theta * R_samp)


def solve_theta_martingale(mu_arrival: float, R_samp: np.ndarray, hi: float, maxiter: int) -> Optional[float]:
    def mart_eq(t: float) -> float:
        lms = get_log_Ms(t, R_samp)
        Ka = (mu_arrival / t) * (math.exp(t) - 1.0)
        Ks = -lms / t
        return Ka - Ks

    try:
        return float(brentq(mart_eq, 1e-9, float(hi), maxiter=int(maxiter)))
    except Exception:
        return None


def dvp_martingale(theta: float, R_samp: np.ndarray, d_th_frames: int) -> float:
    lms = get_log_Ms(theta, R_samp)
    Ks = -lms / theta
    dvp = math.exp(-theta * Ks * float(d_th_frames))
    return float(min(1.0, max(0.0, dvp)))


# =============================================================================
# 5) PHY proxy blocks (same structure as dvp evaluation)
# =============================================================================
def compute_C_est(Beta: np.ndarray, pilots: np.ndarray, lp: int, rho_p: float) -> np.ndarray:
    M, K = Beta.shape
    C_est = np.zeros((M, K), dtype=np.float64)
    for k in range(K):
        p_k = pilots[k]
        same = np.where(pilots == p_k)[0]
        for m in range(M):
            num = lp * rho_p * (Beta[m, k] ** 2)
            den = lp * rho_p * float(np.sum(Beta[m, same])) + 1.0
            C_est[m, k] = num / den
    return C_est


def compute_Eta(C_est: np.ndarray, ServingAP: List[np.ndarray]) -> np.ndarray:
    M, K = C_est.shape
    Eta = np.zeros((M, K), dtype=np.float64)

    served_by_m = [[] for _ in range(M)]
    for k in range(K):
        for m in ServingAP[k]:
            served_by_m[int(m)].append(int(k))

    for m in range(M):
        served = served_by_m[m]
        if len(served) == 0:
            continue
        sum_c = float(np.sum(C_est[m, served]))
        if sum_c <= 0:
            continue
        Eta[m, served] = C_est[m, served] / sum_c

    return Eta


def compute_user_statistics(
    Beta: np.ndarray,
    C_est: np.ndarray,
    Eta: np.ndarray,
    ServingAP: List[np.ndarray],
    pilots: np.ndarray,
    k_curr: int,
    N_ant: int,
    lp: int,
    rho_d: float,
) -> Tuple[float, float, float]:
    M, K = Beta.shape
    Mk = ServingAP[k_curr]
    Pk = np.where(pilots == pilots[k_curr])[0]

    # mu_2
    term_est = 0.0
    for k1 in range(K):
        sum_val = 0.0
        for m in Mk:
            sum_val += Eta[m, k1] * (Beta[m, k_curr] - C_est[m, k_curr])
        term_est += sum_val
    mu_2 = rho_d * term_est

    I_pc = 0.0
    for k1 in Pk:
        if k1 == k_curr:
            continue
        coh = 0.0
        for m in Mk:
            coh += math.sqrt(Eta[m, k1]) * math.sqrt((N_ant - lp) * C_est[m, k_curr])
        I_pc += rho_d * (coh ** 2)
    mu_2 = mu_2 + I_pc + 1.0

    # sigma_2
    sigma_2 = 0.0
    for k1 in range(K):
        sum_E = 0.0
        sum_E2 = 0.0
        for m in Mk:
            val = Eta[m, k1] * (Beta[m, k_curr] - C_est[m, k_curr])
            sum_E += val
            sum_E2 += val ** 2
        sigma_2 += (sum_E ** 2) + (2.0 / float(N_ant)) * sum_E2
    sigma_2 = sigma_2 * (rho_d ** 2)

    # C_val (signal strength)
    sum_ds = 0.0
    for m in Mk:
        sum_ds += math.sqrt(Eta[m, k_curr] * (N_ant - lp) * C_est[m, k_curr])
    C_val = rho_d * (sum_ds ** 2)

    return float(mu_2), float(sigma_2), float(C_val)


def compute_mean_service_and_dvp(
    Beta: np.ndarray,
    ServingAP: List[np.ndarray],
    pilots: np.ndarray,
    N_ant: int,
    lp: int,
    n: int,
    epsilon_dec: float,
    rho_target: float,
    d_th_frames: int,
    rho_p: float,
    rho_d: float,
    num_quantiles: int,
    theta_hi: float,
    brentq_maxiter: int,
) -> Tuple[np.ndarray, np.ndarray]:
    M, K = Beta.shape
    C_est = compute_C_est(Beta, pilots, lp, rho_p=rho_p)
    Eta = compute_Eta(C_est, ServingAP)

    delta_fbl = math.sqrt(1.0 / float(n)) * qinv(epsilon_dec)

    mean_bits_frame = np.zeros((K,), dtype=np.float64)
    dvp = np.ones((K,), dtype=np.float64)

    for k in range(K):
        mu_2, sigma_2, C_val = compute_user_statistics(
            Beta, C_est, Eta, ServingAP, pilots, k, N_ant=N_ant, lp=lp, rho_d=rho_d
        )

        if sigma_2 > 0.0 and mu_2 > 0.0:
            alpha2 = (mu_2 ** 2) / sigma_2
            beta2 = sigma_2 / mu_2
        else:
            alpha2 = 1e-10
            beta2 = 1e-10

        _, R_samp = get_quantiles(alpha2, beta2, C_val, n=n, delta_fbl=delta_fbl, num_q=num_quantiles)
        S_bar = float(np.mean(R_samp))
        mean_bits_frame[k] = S_bar

        # arrival model used in your evaluation code
        mu_arrival = rho_target * S_bar
        theta = solve_theta_martingale(mu_arrival, R_samp, hi=theta_hi, maxiter=brentq_maxiter)
        dvp[k] = 1.0 if theta is None else dvp_martingale(theta, R_samp, d_th_frames=d_th_frames)

    return mean_bits_frame.astype(np.float32), dvp.astype(np.float32)


# =============================================================================
# 6) Environment (one-step bandit)
# =============================================================================
class Env:
    def __init__(self, cfg: dict):
        self.M_ap = int(cfg["M_ap"])
        self.K = int(cfg["K"])
        self.S = int(cfg["S"])
        self.lp = int(cfg["lp"])
        self.N_ant = int(cfg["N_ant"])
        self.n = int(cfg["n"])
        self.B = float(cfg["B"])
        self.epsilon_dec = float(cfg["epsilon_dec"])
        self.d_th_time = float(cfg["d_th_time"])
        self.eps_target_sys = float(cfg["eps_target_sys"])
        self.rho_target = float(cfg["rho_target"])
        self.num_quantiles = int(cfg["num_quantiles"])
        self.theta_hi = float(cfg["theta_hi"])
        self.brentq_maxiter = int(cfg["brentq_maxiter"])
        self.R_ref_bps = float(cfg["R_ref_bps"])
        self.infeasible_penalty = float(cfg["infeasible_penalty"])

        # Noise / power
        P_noise_W = 10 ** ((-174.0 + 10.0 * math.log10(self.B) + float(cfg["noise_figure_db"]) - 30.0) / 10.0)
        self.rho_d = float(cfg["P_AP_W"]) / P_noise_W
        self.rho_p = float(cfg["rho_p"])

        # Delay threshold in frames
        Tf = self.n / self.B
        self.d_th_frames = int(np.round(self.d_th_time / Tf))

        # Fixed topology (same style as your evaluation code: fixed AP/UE positions)
        rng = np.random.default_rng(int(cfg["seed"]))
        area = float(cfg["area"])
        self.user_loc = rng.random((self.K, 2)) * area
        self.ap_loc = rng.random((self.M_ap, 2)) * area
        self.Beta = self._compute_Beta(d0=float(cfg["d0"]), alpha_path=float(cfg["alpha_path"]))

        # Scheme C weights; with identical eps and d_th => uniform weights
        eps_k = np.full((self.K,), self.eps_target_sys, dtype=np.float32)
        dth_k = np.full((self.K,), self.d_th_time, dtype=np.float32)
        w = (np.log(1.0 / eps_k) / dth_k).astype(np.float64)
        self.w_k = (w / np.sum(w)).astype(np.float32)

        self.lambda_lag = 1.0

    def _compute_Beta(self, d0: float, alpha_path: float) -> np.ndarray:
        Beta = np.zeros((self.M_ap, self.K), dtype=np.float64)
        for m in range(self.M_ap):
            for k in range(self.K):
                dist = np.linalg.norm(self.user_loc[k, :] - self.ap_loc[m, :])
                Beta[m, k] = 1.0 / (1.0 + (dist / d0) ** alpha_path)
        Beta = Beta / (np.mean(Beta) + 1e-12)
        return Beta.astype(np.float32)

    def set_lambda(self, lam: float):
        self.lambda_lag = float(max(0.0, lam))

    def reset(self) -> np.ndarray:
        return np.concatenate([self.Beta.reshape(-1), np.array([self.N_ant], dtype=np.float32)], axis=0).astype(np.float32)

    def _decode_action(self, action: np.ndarray) -> Tuple[List[np.ndarray], float]:
        infeasible = 0.0
        ServingAP: List[np.ndarray] = []
        for k in range(self.K):
            chosen = action[k].astype(np.int64)
            if np.any(chosen < 0) or np.any(chosen >= self.M_ap):
                infeasible += 1.0
                chosen = np.clip(chosen, 0, self.M_ap - 1)
            if len(np.unique(chosen)) < len(chosen):
                infeasible += 1.0
            chosen = np.unique(chosen)
            if chosen.size == 0:
                infeasible += 1.0
                chosen = np.array([int(np.argmax(self.Beta[:, k]))], dtype=np.int64)
            if chosen.size > self.S:
                chosen = chosen[: self.S]
            ServingAP.append(chosen)
        return ServingAP, infeasible

    def _wsr_bps(self, mean_bits_frame: np.ndarray) -> float:
        Tf = self.n / self.B
        rate_bps = mean_bits_frame.astype(np.float64) / Tf
        return float(np.sum(self.w_k.astype(np.float64) * rate_bps))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, float]]:
        ServingAP, infeasible = self._decode_action(action)

        pilots = wgf_pilot_assignment(self.Beta.astype(np.float64), ServingAP, self.lp)

        mean_bits_frame, dvp = compute_mean_service_and_dvp(
            Beta=self.Beta.astype(np.float64),
            ServingAP=ServingAP,
            pilots=pilots,
            N_ant=self.N_ant,
            lp=self.lp,
            n=self.n,
            epsilon_dec=self.epsilon_dec,
            rho_target=self.rho_target,
            d_th_frames=self.d_th_frames,
            rho_p=self.rho_p,
            rho_d=self.rho_d,
            num_quantiles=self.num_quantiles,
            theta_hi=self.theta_hi,
            brentq_maxiter=self.brentq_maxiter,
        )

        avg_dvp = float(np.mean(dvp))
        max_dvp = float(np.max(dvp))
        wsr_bps = self._wsr_bps(mean_bits_frame)

        # Avg-DVP constraint penalty (log-scale hinge)
        vio = max(0.0, math.log10(max(avg_dvp, 1e-300) / self.eps_target_sys))
        phi = vio ** 2

        reward_obj = wsr_bps / self.R_ref_bps
        reward = reward_obj - self.lambda_lag * phi - self.infeasible_penalty * float(infeasible)

        obs = self.reset()
        info = {
            "wsr_bps": wsr_bps,
            "avg_dvp": avg_dvp,
            "max_dvp": max_dvp,
            "phi": phi,
            "infeasible": float(infeasible),
        }
        return obs, float(reward), info


# =============================================================================
# 7) GNN policy + value head
# =============================================================================
class BipartiteGNN(nn.Module):
    def __init__(self, d: int = 64, mp_layers: int = 2):
        super().__init__()
        self.ap_in = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.ue_in = nn.Sequential(nn.Linear(2, d), nn.Tanh())
        self.ap_upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.Tanh()) for _ in range(mp_layers)])
        self.ue_upd = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.Tanh()) for _ in range(mp_layers)])

    def forward(self, beta: torch.Tensor):
        # beta: (B,M,K)
        ap_feat = torch.stack([beta.mean(dim=2), beta.max(dim=2).values], dim=-1)
        ue_feat = torch.stack([beta.mean(dim=1), beta.max(dim=1).values], dim=-1)
        h_ap = self.ap_in(ap_feat)
        h_ue = self.ue_in(ue_feat)

        a2u = beta / torch.clamp(beta.sum(dim=1, keepdim=True), min=1e-12)
        u2a = beta / torch.clamp(beta.sum(dim=2, keepdim=True), min=1e-12)

        for l in range(len(self.ap_upd)):
            msg_ue = torch.einsum("bmk,bmd->bkd", a2u, h_ap)
            h_ue = self.ue_upd[l](torch.cat([h_ue, msg_ue], dim=-1))
            msg_ap = torch.einsum("bmk,bkd->bmd", u2a, h_ue)
            h_ap = self.ap_upd[l](torch.cat([h_ap, msg_ap], dim=-1))
        return h_ap, h_ue


class Agent(nn.Module):
    def __init__(self, M: int, K: int, S: int, d: int = 64, mp_layers: int = 2):
        super().__init__()
        self.M, self.K, self.S = M, K, S
        self.gnn = BipartiteGNN(d=d, mp_layers=mp_layers)
        self.ap_proj = nn.Linear(d, d, bias=False)
        self.ue_proj = nn.Linear(d, d, bias=False)
        self.value_head = nn.Sequential(nn.Linear(2 * d, 128), nn.Tanh(), nn.Linear(128, 1))

    @staticmethod
    def masked_softmax(logits: torch.Tensor, mask: torch.Tensor):
        masked = logits.masked_fill(~mask, float("-inf"))
        p = torch.softmax(masked, dim=-1)
        p = torch.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
        s = p.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return p / s

    def forward(self, beta: torch.Tensor):
        h_ap, h_ue = self.gnn(beta)
        a = self.ap_proj(h_ap)
        u = self.ue_proj(h_ue)
        logits = torch.einsum("bkd,bmd->bkm", u, a) / math.sqrt(a.shape[-1])  # (B,K,M)
        v = self.value_head(torch.cat([h_ap.mean(dim=1), h_ue.mean(dim=1)], dim=-1)).squeeze(-1)
        return logits, v

    def sample_action(self, logits: torch.Tensor):
        """
        For each user, sample S APs WITHOUT replacement using sequential masked categorical
        (Plackett–Luce style).
        """
        B, K, M = logits.shape
        device = logits.device
        actions = torch.zeros((B, K, self.S), dtype=torch.long, device=device)
        logp = torch.zeros((B,), dtype=torch.float32, device=device)
        ent = torch.zeros((B,), dtype=torch.float32, device=device)

        for b in range(B):
            lp_b, ent_b = 0.0, 0.0
            for k in range(K):
                mask = torch.ones((M,), dtype=torch.bool, device=device)
                for s in range(self.S):
                    probs = self.masked_softmax(logits[b, k], mask)
                    dist = torch.distributions.Categorical(probs=probs)
                    a = dist.sample()
                    actions[b, k, s] = a
                    lp_b = lp_b + dist.log_prob(a)
                    ent_b = ent_b + dist.entropy()
                    mask[a] = False
            logp[b] = lp_b
            ent[b] = ent_b
        return actions, logp, ent

    def logprob_entropy(self, logits: torch.Tensor, actions: torch.Tensor):
        B, K, M = logits.shape
        device = logits.device
        logp = torch.zeros((B,), dtype=torch.float32, device=device)
        ent = torch.zeros((B,), dtype=torch.float32, device=device)

        for b in range(B):
            lp_b, ent_b = 0.0, 0.0
            for k in range(K):
                mask = torch.ones((M,), dtype=torch.bool, device=device)
                for s in range(actions.shape[2]):
                    probs = self.masked_softmax(logits[b, k], mask)
                    dist = torch.distributions.Categorical(probs=probs)
                    a = actions[b, k, s]
                    lp_b = lp_b + dist.log_prob(a)
                    ent_b = ent_b + dist.entropy()
                    mask[a] = False
            logp[b] = lp_b
            ent[b] = ent_b
        return logp, ent


# =============================================================================
# 8) PPO training
# =============================================================================
@dataclass
class PPOCfg:
    device: str
    total_updates: int
    batch_episodes: int
    update_epochs: int
    minibatch_size: int
    lr: float
    clip_coef: float
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    lambda_init: float
    lambda_lr: float
    lambda_max: float


def train(env: Env, agent: Agent, cfg: PPOCfg):
    device = torch.device(cfg.device)
    agent.to(device)
    opt = optim.Adam(agent.parameters(), lr=cfg.lr, eps=1e-5)

    # ===== GPU 预热 =====
    if device.type == 'cuda':
        print("[GPU] Warming up CUDA kernels...")
        dummy_x = torch.randn(2, env.M_ap, env.K, device=device)
        for _ in range(5):
            _ = agent(dummy_x)
        torch.cuda.synchronize()
        print("[GPU] Warmup complete, starting training...")

    lam = cfg.lambda_init
    hist = {"update": [], "avg_return": [], "avg_wsr_bps": [], "avg_dvp": [], "lambda": []}

    for upd in range(1, cfg.total_updates + 1):
        env.set_lambda(lam)

        beta_batch, act_batch, oldlogp_batch = [], [], []
        ret_batch, val_batch = [], []
        wsr_batch, dvp_batch, phi_batch = [], [], []

        # ===== 数据收集阶段（批量转移到 GPU）=====
        for _ in range(cfg.batch_episodes):
            obs = env.reset()
            beta = obs[: env.M_ap * env.K].reshape(env.M_ap, env.K)
            # 直接在 GPU 上创建张量，避免 CPU 中间转换
            beta_t = torch.from_numpy(beta).float().to(device).unsqueeze(0)  # (1,M,K)

            with torch.no_grad():
                logits, v = agent(beta_t)
                action, logp, _ = agent.sample_action(logits)

            _, reward, info = env.step(action.squeeze(0).cpu().numpy())

            # 批量追加到列表（GPU 张量）
            beta_batch.append(beta_t.squeeze(0))
            act_batch.append(action.squeeze(0))
            oldlogp_batch.append(logp.squeeze(0))
            ret_batch.append(torch.tensor(reward, dtype=torch.float32, device=device))
            val_batch.append(v.squeeze(0))

            # 预分配张量而不是每次创建新张量
            wsr_batch.append(torch.tensor(info["wsr_bps"], dtype=torch.float32, device=device))
            dvp_batch.append(torch.tensor(info["avg_dvp"], dtype=torch.float32, device=device))
            phi_batch.append(torch.tensor(info["phi"], dtype=torch.float32, device=device))

        # ===== 批量 stack 操作（在 GPU 上进行）=====
        beta_b = torch.stack(beta_batch, dim=0)  # 已在 GPU
        actions_b = torch.stack(act_batch, dim=0)  # 已在 GPU
        oldlogp_b = torch.stack(oldlogp_batch, dim=0)  # 已在 GPU
        returns_b = torch.stack(ret_batch, dim=0)  # 已在 GPU
        values_b = torch.stack(val_batch, dim=0)  # 已在 GPU

        if device.type == 'cuda':
            torch.cuda.synchronize()  # 等待所有 GPU 操作完成

        # ===== 优势函数计算 =====
        adv_b = returns_b - values_b.detach()
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        B = beta_b.shape[0]
        inds = torch.randperm(B, device=device)

        # ===== PPO 更新循环（带 GPU 同步）=====
        for _ in range(cfg.update_epochs):
            for start in range(0, B, cfg.minibatch_size):
                mb = inds[start : start + cfg.minibatch_size]
                
                # 前向传播
                logits, v = agent(beta_b[mb])
                newlogp, ent = agent.logprob_entropy(logits, actions_b[mb])

                # PPO 损失计算
                ratio = torch.exp(newlogp - oldlogp_b[mb])
                pg1 = -adv_b[mb] * ratio
                pg2 = -adv_b[mb] * torch.clamp(ratio, 1.0 - cfg.clip_coef, 1.0 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()

                v_loss = 0.5 * (returns_b[mb] - v).pow(2).mean()
                ent_loss = ent.mean()

                loss = pg_loss - cfg.ent_coef * ent_loss + cfg.vf_coef * v_loss

                # 反向传播
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                opt.step()

        # ===== 统计指标计算（高效 GPU 操作）=====
        phi_mean = float(torch.stack(phi_batch).mean().item())
        lam = min(cfg.lambda_max, max(0.0, lam + cfg.lambda_lr * phi_mean))

        # 使用 .item() 一次性提取所有 metrics，减少 GPU 同步
        avg_ret = float(returns_b.mean().item())
        avg_wsr = float(torch.stack(wsr_batch).mean().item())
        avg_dvp = float(torch.stack(dvp_batch).mean().item())

        hist["update"].append(upd)
        hist["avg_return"].append(avg_ret)
        hist["avg_wsr_bps"].append(avg_wsr)
        hist["avg_dvp"].append(avg_dvp)
        hist["lambda"].append(lam)

        if upd % 1 == 0:
            print(f"[{upd:4d}/{cfg.total_updates}] return={avg_ret:.4f}  WSR={avg_wsr/1e6:.3f} Mbps  avgDVP={avg_dvp:.2e}  lambda={lam:.2f}")

    return hist, agent


@torch.no_grad()
def evaluate(env: Env, agent: Agent, episodes: int = 20, device: str = "cpu") -> Dict[str, float]:
    device = torch.device(device)
    agent.eval()
    wsr_list, dvp_list = [], []
    for _ in range(episodes):
        obs = env.reset()
        beta = obs[: env.M_ap * env.K].reshape(env.M_ap, env.K)
        beta_t = torch.tensor(beta, dtype=torch.float32, device=device).unsqueeze(0)
        logits, _ = agent(beta_t)

        # Deterministic evaluation: choose top-S APs per UE by logits
        actions = np.zeros((env.K, env.S), dtype=np.int64)
        for k in range(env.K):
            top = torch.topk(logits[0, k], k=env.S, dim=-1).indices.cpu().numpy()
            actions[k, :] = top

        _, _, info = env.step(actions)
        wsr_list.append(info["wsr_bps"])
        dvp_list.append(info["avg_dvp"])

    return {"wsr_bps_mean": float(np.mean(wsr_list)), "avg_dvp_mean": float(np.mean(dvp_list))}


def plot_curve(hist: Dict[str, list], out_png: str = "reward_curve.png"):
    plt.figure(figsize=(8, 5))
    plt.plot(hist["update"], hist["avg_return"])
    plt.grid(True, alpha=0.3)
    plt.xlabel("Update")
    plt.ylabel("Average Return")
    plt.title("Reward Curve (GNN+PPO, embedded WGF, Martingale-SNC)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


# =============================================================================
# 9) Main
# =============================================================================
def main():
    set_seed(int(CFG["seed"]))

    env = Env(CFG)
    agent = Agent(M=env.M_ap, K=env.K, S=env.S, d=64, mp_layers=2)

    cfg = PPOCfg(**PPO_HYPER)

    hist, agent = train(env, agent, cfg)
    plot_curve(hist, "reward_curve.png")

    stats = evaluate(env, agent, episodes=20, device=cfg.device)
    print("\n=== Evaluation (deterministic top-S) ===")
    print(f"Trained weighted-sum-rate: {stats['wsr_bps_mean']/1e6:.3f} Mbps")
    print(f"Average DVP:               {stats['avg_dvp_mean']:.2e}")
    print("Saved reward curve to reward_curve.png")

    torch.save(agent.state_dict(), "gnn_ppo_wgf_martingale.pt")
    print("Saved model to gnn_ppo_wgf_martingale.pt")


if __name__ == "__main__":
    main()
