#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
[Observability Version] GNN+PPO
Feature: Explicitly logs and plots WSR (Mbps) to verify convergence and trade-offs.
"""

import os
import math
import random
import json
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple, Dict
from scipy.special import gammaincinv, erfcinv

# =============================================================================
# 1) Utilities & Physics
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def qinv(eps: float) -> float:
    return math.sqrt(2.0) * erfcinv(2.0 * eps)

def wgf_pilot_assignment(Beta: np.ndarray, ServingAP: List[np.ndarray], lp_val: int) -> np.ndarray:
    M, K = Beta.shape
    pilots = np.zeros(K, dtype=np.int64)
    S = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        Ak = ServingAP[k]
        if Ak is None or len(Ak) == 0: continue
        S[k, :] = np.sum(Beta[Ak, :], axis=0)

    eps = 1e-12
    Skk = S[np.arange(K), np.arange(K)] + eps
    Ratio = S / Skk[:, None] 
    omega = Ratio**2 + (Ratio.T)**2
    deg = np.sum(omega, axis=1)
    order = np.argsort(-deg)

    pilot_groups = [[] for _ in range(lp_val)]
    for k in order:
        best_p = 0
        best_cost = float("inf")
        for p in range(lp_val):
            cost = 0.0
            for u in pilot_groups[p]:
                cost += omega[k, u]
            if cost < best_cost:
                best_cost = cost
                best_p = p
        pilots[k] = best_p
        pilot_groups[best_p].append(int(k))
    return pilots

def compute_C_est(Beta: np.ndarray, pilots: np.ndarray, lp: int, rho_p: float) -> np.ndarray:
    pilot_matrix = (pilots[:, None] == pilots[None, :]).astype(np.float64)
    Sum_Beta_SamePilot = Beta @ pilot_matrix
    Num = lp * rho_p * (Beta ** 2)
    Den = lp * rho_p * Sum_Beta_SamePilot + 1.0
    return Num / Den

def compute_Eta(C_est: np.ndarray, ServingAP: List[np.ndarray]) -> np.ndarray:
    M, K = C_est.shape
    Eta = np.zeros((M, K), dtype=np.float64)
    ap_users = [[] for _ in range(M)]
    for k, aps in enumerate(ServingAP):
        for m in aps: ap_users[m].append(k)
    for m in range(M):
        users = ap_users[m]
        if not users: continue
        c_vals = C_est[m, users]
        sum_c = np.sum(c_vals)
        if sum_c > 1e-12: Eta[m, users] = c_vals / sum_c
    return Eta

def compute_user_statistics_vectorized(Beta, C_est, Eta, ServingAP, pilots, N_ant, lp, rho_d):
    M, K = Beta.shape
    mu_2_arr = np.zeros(K, dtype=np.float64)
    sigma_2_arr = np.zeros(K, dtype=np.float64)
    C_val_arr = np.zeros(K, dtype=np.float64)
    pilot_match = (pilots[:, None] == pilots[None, :])
    np.fill_diagonal(pilot_match, False)
    Factor_C = (N_ant - lp) * C_est
    total_eta_per_ap = np.sum(Eta, axis=1)

    for k in range(K):
        Mk = ServingAP[k]
        if len(Mk) == 0: continue
        diff_m_k = Beta[Mk, k] - C_est[Mk, k]
        eta_sum_m = total_eta_per_ap[Mk]
        mu_2 = rho_d * np.sum(diff_m_k * eta_sum_m)
        pk_users = np.where(pilot_match[k])[0]
        if pk_users.size > 0:
            sqrt_C = np.sqrt(np.maximum(Factor_C[Mk, k], 0.0))[:, None]
            sqrt_Eta = np.sqrt(np.maximum(Eta[Mk][:, pk_users], 0.0))
            coh = np.sum(sqrt_Eta * sqrt_C, axis=0)
            mu_2 += rho_d * np.sum(coh ** 2)
        mu_2 += 1.0
        w_diff = Eta[Mk, :] * diff_m_k[:, None]
        sum_E = np.sum(w_diff, axis=0)
        sum_E2 = np.sum(w_diff ** 2, axis=0)
        sig_val = np.sum(sum_E ** 2) + (2.0 / max(float(N_ant), 1.0)) * np.sum(sum_E2)
        sigma_2 = (rho_d ** 2) * sig_val
        term_in = Eta[Mk, k] * Factor_C[Mk, k]
        sum_ds = np.sum(np.sqrt(np.maximum(term_in, 0.0)))
        C_val = rho_d * (sum_ds ** 2)
        mu_2_arr[k] = mu_2
        sigma_2_arr[k] = sigma_2
        C_val_arr[k] = C_val
    return mu_2_arr, sigma_2_arr, C_val_arr

@torch.jit.script
def solve_theta_jit(mu_arr: torch.Tensor, R_tensor: torch.Tensor, max_iter: int = 15) -> torch.Tensor:
    K, Q = R_tensor.shape
    theta = torch.full((K,), 0.1, device=R_tensor.device)
    log_Q = math.log(float(Q))
    for _ in range(max_iter):
        neg_theta_R = -theta.unsqueeze(-1) * R_tensor
        max_val, _ = torch.max(neg_theta_R, dim=-1, keepdim=True)
        log_sum_exp = max_val + torch.log(torch.sum(torch.exp(neg_theta_R - max_val), dim=-1, keepdim=True))
        log_Ms = log_sum_exp - log_Q
        exp_theta = torch.exp(theta)
        f_val = mu_arr * (exp_theta - 1.0) + log_Ms.squeeze(-1)
        exp_shifted = torch.exp(neg_theta_R - max_val)
        numerator = torch.sum(R_tensor * exp_shifted, dim=-1)
        denominator = torch.sum(exp_shifted, dim=-1) + 1e-12
        moment_1 = numerator / denominator
        f_prime = mu_arr * exp_theta - moment_1
        step = f_val / (f_prime + 1e-8)
        theta = torch.clamp(theta - 0.8 * step, min=1e-6, max=100.0)
    return theta

# =============================================================================
# 2) Environment
# =============================================================================

class Env:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.M_ap = cfg["M_ap"]
        self.K = cfg["K"]
        self.S = cfg["S"]
        self.lp = cfg["lp"]
        self.N_ant = cfg["N_ant"]
        self.n = cfg["n"]
        self.B = cfg["B"]
        self.epsilon_dec = cfg["epsilon_dec"]
        self.d_th_time = cfg["d_th_time"]
        self.eps_target_sys = cfg["eps_target_sys"]
        self.rho_target = cfg["rho_target"]
        self.fixed_positions = cfg.get("fixed_positions", True)
        self.num_quantiles = cfg["num_quantiles"]
        self.R_ref_bps = cfg["R_ref_bps"]
        self.infeasible_penalty = cfg["infeasible_penalty"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        P_noise_W = 10 ** ((-174.0 + 10.0 * math.log10(self.B) + float(cfg["noise_figure_db"]) - 30.0) / 10.0)
        self.rho_d = float(cfg["P_AP_W"]) / P_noise_W
        self.rho_p = float(cfg["rho_p"])
        Tf = self.n / self.B
        self.d_th_frames = int(np.round(self.d_th_time / Tf))
        
        self._rng = np.random.default_rng(int(cfg["seed"]))
        self._area = float(cfg["area"])
        self._d0 = float(cfg["d0"])
        self._alpha_path = float(cfg["alpha_path"])
        
        self._sample_positions()
        
        eps_k = np.full((self.K,), self.eps_target_sys, dtype=np.float32)
        dth_k = np.full((self.K,), self.d_th_time, dtype=np.float32)
        w = (np.log(1.0 / eps_k) / dth_k).astype(np.float64)
        self.w_k = (w / np.sum(w)).astype(np.float32)

    def _compute_Beta(self, d0, alpha_path):
        Beta = np.zeros((self.M_ap, self.K), dtype=np.float64)
        for m in range(self.M_ap):
            for k in range(self.K):
                dist = np.linalg.norm(self.user_loc[k] - self.ap_loc[m])
                Beta[m, k] = 1.0 / (1.0 + (dist / d0) ** alpha_path)
        return Beta.astype(np.float32)

    def _sample_positions(self):
        self.user_loc = self._rng.random((self.K, 2)) * self._area
        self.ap_loc = self._rng.random((self.M_ap, 2)) * self._area
        self.Beta = self._compute_Beta(d0=self._d0, alpha_path=self._alpha_path)

    def get_candidate_mask(self, top_k_candidates=20):
        mask = torch.full((self.K, self.M_ap), -1e9, device=self.device)
        Beta_t = torch.from_numpy(self.Beta.T).to(self.device) 
        _, indices = torch.topk(Beta_t, k=top_k_candidates, dim=1)
        mask.scatter_(1, indices, 0.0)
        return mask

    def reset(self):
        if not self.fixed_positions:
            self._sample_positions()
        return np.concatenate([self.Beta.reshape(-1), np.array([self.N_ant], dtype=np.float32)], axis=0).astype(np.float32)

    def _decode_action(self, action):
        infeasible = 0.0
        ServingAP = []
        for k in range(self.K):
            chosen = action[k].astype(np.int64)
            if np.any(chosen < 0) or np.any(chosen >= self.M_ap):
                infeasible += 1.0
                chosen = np.clip(chosen, 0, self.M_ap - 1)
            u = np.unique(chosen)
            if len(u) < len(chosen):
                infeasible += 0.5 
                chosen = u
            if chosen.size == 0:
                infeasible += 1.0
                chosen = np.array([int(np.argmax(self.Beta[:, k]))], dtype=np.int64)
            if chosen.size > self.S:
                chosen = chosen[: self.S]
            ServingAP.append(chosen)
        return ServingAP, infeasible

    def _evaluate_action(self, ServingAP: List[np.ndarray], infeasible: float) -> Dict[str, float]:
        pilots = wgf_pilot_assignment(self.Beta.astype(np.float64), ServingAP, self.lp)
        C_est = compute_C_est(self.Beta.astype(np.float64), pilots, self.lp, self.rho_p)
        Eta = compute_Eta(C_est, ServingAP)
        mu2, sigma2, C_val = compute_user_statistics_vectorized(
            self.Beta.astype(np.float64), C_est, Eta, ServingAP, pilots, self.N_ant, self.lp, self.rho_d
        )

        safe_mask = (sigma2 > 1e-9) & (mu2 > 1e-9)
        alpha2 = np.zeros_like(mu2); beta2 = np.zeros_like(mu2)
        alpha2[safe_mask] = (mu2[safe_mask] ** 2) / sigma2[safe_mask]
        beta2[safe_mask] = sigma2[safe_mask] / mu2[safe_mask]
        alpha2[~safe_mask] = 1e-2; beta2[~safe_mask] = 1e-2

        delta_fbl = math.sqrt(1.0 / self.n) * qinv(self.epsilon_dec)
        p = (np.arange(1, self.num_quantiles + 1) - 0.5) / self.num_quantiles
        I_mat = gammaincinv(alpha2[:, None], p[None, :]) * beta2[:, None]
        R_mat = np.maximum(0.0, self.n * np.log2(1.0 + C_val[:, None] / I_mat) - self.n * delta_fbl)
        S_bar = np.mean(R_mat, axis=1)

        mu_arrival_np = self.rho_target * S_bar
        R_tensor = torch.from_numpy(R_mat).float().to(self.device)
        mu_arr = torch.from_numpy(mu_arrival_np).float().to(self.device)
        theta_tensor = solve_theta_jit(mu_arr, R_tensor)

        neg_theta_R = -theta_tensor.unsqueeze(-1) * R_tensor
        max_val, _ = torch.max(neg_theta_R, dim=-1, keepdim=True)
        log_sum = max_val + torch.log(torch.sum(torch.exp(neg_theta_R - max_val), dim=-1, keepdim=True))
        log_Ms = log_sum.squeeze(-1) - math.log(float(self.num_quantiles))
        
        Ks = -log_Ms / (theta_tensor + 1e-10)
        dvp = torch.exp(-theta_tensor * Ks * float(self.d_th_frames)).clamp(0, 1).cpu().numpy()

        wsr = float(np.sum(self.w_k * (S_bar * (self.B / self.n))))
        
        eps = float(self.eps_target_sys)
        worst_dvp = float(np.max(dvp))
        safe_dvp = max(worst_dvp, 1e-12)
        
        r_wsr = wsr / self.R_ref_bps
        log_term = math.log10(safe_dvp / eps)
        
        if log_term > 0:
            penalty = 2.0 * log_term 
            reward = r_wsr - penalty
        else:
            reward = r_wsr - 0.1 * log_term
            
        reward -= (self.infeasible_penalty * infeasible)

        return {
            'wsr_bps': wsr,
            'worst_dvp': worst_dvp,
            'reward': float(reward),
            'constraint_satisfied': float(worst_dvp <= eps)
        }

    def step(self, action):
        ServingAP, infeasible = self._decode_action(action)
        metrics = self._evaluate_action(ServingAP, infeasible)

        next_state = self.reset()
        info = {
            'wsr_bps': metrics['wsr_bps'],
            'worst_dvp': metrics['worst_dvp'],
            'reward': metrics['reward']
        }
        return next_state, float(metrics['reward']), True, info

# =============================================================================
# 3) GNN & Agent
# =============================================================================

class GNNPolicy(nn.Module):
    def __init__(self, M_ap: int, K: int, S: int, hidden: int = 128):
        super().__init__()
        self.ap_embed = nn.Sequential(nn.Linear(K, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.ue_embed = nn.Sequential(nn.Linear(M_ap, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.msg_ap_to_ue = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.msg_ue_to_ap = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.out = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.ReLU(), nn.Linear(hidden, M_ap))

    def forward(self, Beta: torch.Tensor, mask: torch.Tensor = None):
        feat = torch.log10(Beta + 1e-15) 
        B, M, K = feat.shape
        ap_h = self.ap_embed(feat)
        ue_h = self.ue_embed(feat.transpose(1, 2))
        ap_msg = self.msg_ap_to_ue(ap_h)
        ue_agg = torch.einsum("bmh,bkm->bkh", ap_msg, Beta.transpose(1, 2)) / (M + 1e-6)
        ue_cat = torch.cat([ue_h, ue_agg], dim=-1)
        logits = self.out(ue_cat) 
        if mask is not None:
            mask_b = mask.unsqueeze(0).expand(B, -1, -1)
            logits = logits + mask_b
        return logits

class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))
    def forward(self, obs): return self.net(obs).squeeze(-1)

def plackett_luce_sample(logits: torch.Tensor, S: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, K, M = logits.shape
    u = torch.rand_like(logits)
    g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
    scores = logits + g
    topk = torch.topk(scores, k=S, dim=-1)
    act_idx = topk.indices 
    log_probs = F.log_softmax(logits, dim=-1)
    selected_logp = torch.gather(log_probs, -1, act_idx).sum(dim=(1, 2))
    return act_idx, selected_logp


def evaluate(env: Env, policy: nn.Module, candidate_mask: torch.Tensor, mode: str = "ppo", runs: int = 1,
             temperature: float = None, greedy_top_s: bool = True) -> Dict[str, float]:
    assert mode in {"ppo", "greedy"}
    results = []
    if mode == "ppo" and policy is not None:
        policy.eval()
    for _ in range(runs):
        if not env.fixed_positions:
            env._sample_positions()
        Beta = env.Beta.reshape(env.M_ap, env.K)

        if mode == "greedy":
            action = np.argsort(-Beta, axis=0)[:env.S, :].T
        else:
            Beta_t = torch.from_numpy(Beta).float().to(env.device).unsqueeze(0)
            with torch.no_grad():
                logits = policy(Beta_t, candidate_mask)
                if temperature is not None and temperature > 0 and not greedy_top_s:
                    scaled_logits = logits / float(temperature)
                    act_idx, _ = plackett_luce_sample(scaled_logits, env.S)
                else:
                    act_idx = torch.topk(logits, k=env.S, dim=-1).indices
                action = act_idx.squeeze(0).cpu().numpy()

        ServingAP, infeasible = env._decode_action(action)
        metrics = env._evaluate_action(ServingAP, infeasible)
        results.append(metrics)

    avg_wsr = float(np.mean([r['wsr_bps'] for r in results]) / 1e6)
    avg_worst_dvp = float(np.mean([r['worst_dvp'] for r in results]))
    constraint_rate = float(np.mean([r['constraint_satisfied'] for r in results]))
    return {
        "mode": mode,
        "avg_wsr_mbps": avg_wsr,
        "avg_worst_dvp": avg_worst_dvp,
        "constraint_rate": constraint_rate
    }


def save_eval_results(results: List[Dict[str, float]], json_path: str = "eval_results.json", csv_path: str = "eval_results.csv"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    fieldnames = ["mode", "avg_wsr_mbps", "avg_worst_dvp", "constraint_rate"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

# =============================================================================
# 4) Training Loop
# =============================================================================

@dataclass
class Transition:
    obs: np.ndarray; action: np.ndarray; logp: float; reward: float; value: float

class PPOBuffer:
    def __init__(self): self.data = []
    def add(self, obs, action, logp, reward, value): self.data.append(Transition(obs, action, logp, reward, value))
    def clear(self): self.data = []

def train(cfg: Dict):
    set_seed(int(cfg["seed"]))
    env = Env(cfg)
    obs_dim = env.reset().shape[0]
    M_val, K_val = cfg["M_ap"], cfg["K"]
    candidate_mask = env.get_candidate_mask(top_k_candidates=20) 

    policy = GNNPolicy(M_ap=M_val, K=K_val, S=cfg["S"], hidden=cfg["hidden"]).to(env.device)
    value_fn = ValueNet(obs_dim=obs_dim, hidden=cfg["hidden_v"]).to(env.device)
    
    pi_opt = optim.Adam(policy.parameters(), lr=3e-5)
    vf_opt = optim.Adam(value_fn.parameters(), lr=1e-3)
    buf = PPOBuffer()

    # Pretrain
    print("=== [Phase 1] Pre-training ===")
    target_indices = np.argsort(-env.Beta, axis=0)[:cfg["S"], :].T
    target_multi_hot = torch.zeros((1, K_val, M_val), device=env.device)
    for k in range(K_val): target_multi_hot[0, k, target_indices[k]] = 1.0
    Beta_static_t = torch.from_numpy(env.Beta.reshape(M_val, K_val)).float().to(env.device).unsqueeze(0)
    pre_opt = optim.Adam(policy.parameters(), lr=1e-3)
    for i in range(1001):
        logits = policy(Beta_static_t, candidate_mask)
        loss = F.binary_cross_entropy_with_logits(logits, target_multi_hot)
        pre_opt.zero_grad(); loss.backward(); pre_opt.step()
        if i % 1 == 0:
            with torch.no_grad():
                act_idx, _ = plackett_luce_sample(logits, cfg["S"])
                match_count = 0
                sampled = act_idx.squeeze(0).cpu().numpy()
                for k in range(K_val): match_count += np.intersect1d(sampled[k], target_indices[k]).size
                acc = match_count / (K_val * cfg["S"])
                print(f"[Pretrain {i:04d}] Acc={acc:.1%}")
                if acc > 0.6: break

    # PPO
    print(f"\n=== [Phase 2] PPO Training (Observability Mode) ===")
    target_kl = 0.015; kl_coef = 1.5
    metrics = {"rewards": [], "worst_dvp": [], "wsr": []}

    for it in range(cfg["train_iters"]):
        obs = env.reset()
        Beta = env.Beta.reshape(env.M_ap, env.K)
        Beta_t = torch.from_numpy(Beta).float().to(env.device).unsqueeze(0)

        with torch.no_grad():
            logits = policy(Beta_t, candidate_mask)
            act_idx, logp_tensor = plackett_luce_sample(logits, cfg["S"])
            action = act_idx.squeeze(0).cpu().numpy()
            value = value_fn(torch.from_numpy(obs).float().to(env.device).unsqueeze(0)).item()
            logp_total = logp_tensor.item()

        next_obs, reward, done, info = env.step(action)
        buf.add(obs, action, logp_total, reward, value)
        
        metrics["rewards"].append(reward)
        metrics["worst_dvp"].append(info["worst_dvp"])
        metrics["wsr"].append(info["wsr_bps"] / 1e6) # Store in Mbps

        if (it + 1) % cfg["batch_size"] == 0:
            obs_batch = torch.from_numpy(np.stack([t.obs for t in buf.data])).float().to(env.device)
            act_batch = torch.from_numpy(np.stack([t.action for t in buf.data])).long().to(env.device)
            old_logp = torch.tensor([t.logp for t in buf.data], dtype=torch.float32, device=env.device)
            rew_batch = torch.tensor([t.reward for t in buf.data], dtype=torch.float32, device=env.device)
            val_old = torch.tensor([t.value for t in buf.data], dtype=torch.float32, device=env.device)
            
            adv = (rew_batch - val_old).detach()
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            batch_betas = obs_batch[:, :-1].view(-1, M_val, K_val)
            
            continue_training = True
            for epoch in range(cfg["update_epochs"]):
                if not continue_training: break

                logits = policy(batch_betas, candidate_mask)
                log_probs = F.log_softmax(logits, dim=-1)
                new_logp = torch.gather(log_probs, -1, act_batch).sum(dim=(1, 2))
                probs = F.softmax(logits, dim=-1)
                entropy = -(probs * log_probs).sum(dim=-1).mean()
                
                with torch.no_grad():
                    kl = (old_logp - new_logp).mean().item()
                    if kl > kl_coef * target_kl: continue_training = False; break

                ratio = torch.exp(new_logp - old_logp)
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1.0 - cfg["clip_ratio"], 1.0 + cfg["clip_ratio"]) * adv
                pi_loss = -torch.mean(torch.min(surr1, surr2))
                v_loss = torch.mean((value_fn(obs_batch) - rew_batch) ** 2)
                loss_pi = pi_loss - cfg["ent_coef"] * entropy
                
                pi_opt.zero_grad(); loss_pi.backward(); torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg["grad_clip"]); pi_opt.step()
                vf_opt.zero_grad(); v_loss.backward(); torch.nn.utils.clip_grad_norm_(value_fn.parameters(), cfg["grad_clip"]); vf_opt.step()
            buf.clear()
            
            if (it + 1) % (cfg["batch_size"] * 2) == 0:
                avg_r = np.mean(metrics["rewards"][-cfg["batch_size"]:])
                avg_wdvp = np.mean(metrics["worst_dvp"][-cfg["batch_size"]:])
                avg_wsr = np.mean(metrics["wsr"][-cfg["batch_size"]:])
                print(f"[Iter {it+1:04d}] Reward: {avg_r:.3f} | WSR: {avg_wsr:.2f} Mbps | WorstDVP: {avg_wdvp:.2e}")

    print("\n=== [Phase 3] Evaluation ===")
    eval_runs = int(cfg.get("eval_runs", 1))
    eval_temp = cfg.get("eval_temperature", None)
    greedy_top_s = bool(cfg.get("eval_greedy_top", True))

    greedy_res = evaluate(env, policy, candidate_mask, mode="greedy", runs=eval_runs)
    ppo_res = evaluate(env, policy, candidate_mask, mode="ppo", runs=eval_runs, temperature=eval_temp, greedy_top_s=greedy_top_s)
    eval_results = [greedy_res, ppo_res]
    save_eval_results(eval_results)

    print("Evaluation summary (average over runs):")
    for res in eval_results:
        print(
            f" - {res['mode'].upper():6s} | WSR: {res['avg_wsr_mbps']:.3f} Mbps | "
            f"Worst DVP: {res['avg_worst_dvp']:.2e} | Constraint rate: {res['constraint_rate']:.2%}"
        )

    # Plot
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. Reward
    axes[0].plot(metrics["rewards"], color='tab:blue', alpha=0.3)
    axes[0].set_ylabel('Reward')
    axes[0].set_title('Training Metrics')
    
    # 2. WSR
    axes[1].plot(metrics["wsr"], color='tab:orange', alpha=0.3)
    axes[1].set_ylabel('WSR (Mbps)')
    w = 50
    if len(metrics["wsr"]) > w:
        ma = np.convolve(metrics["wsr"], np.ones(w)/w, mode='valid')
        axes[1].plot(range(w-1, len(metrics["wsr"])), ma, color='tab:orange', linewidth=2)
    axes[1].axhline(y=greedy_res["avg_wsr_mbps"], color='gray', linestyle='--', label=f"Greedy: {greedy_res['avg_wsr_mbps']:.2f} Mbps")
    axes[1].legend()
        
    # 3. DVP
    dvp_np = np.maximum(np.array(metrics["worst_dvp"]), 1e-16)
    axes[2].plot(dvp_np, color='tab:red', alpha=0.3)
    axes[2].set_ylabel('Worst DVP (Log)')
    axes[2].set_yscale('log')
    axes[2].axhline(y=cfg['eps_target_sys'], color='green', linestyle='--')
    axes[2].set_xlabel('Steps')

    fig.tight_layout()
    plt.savefig("wsr_observability.png")
    print("\nSaved plot to wsr_observability.png")

if __name__ == "__main__":
    cfg = dict(
        seed=2025, 
        M_ap=60, K=30, S=10, lp=4, N_ant=16,
        n=200, B=2e6, 
        
        # 保持你的物理参数
        rho_target=0.95,  # 0.95 是个很好的挑战值
        eps_target_sys=1e-2, 
        d_th_time=5e-3, epsilon_dec=1e-2,
        R_ref_bps=9e6,
        
        # === 关键修改点 ===
        batch_size=512,      # 增大 Batch Size 以稳定方差
        train_iters=4000,    # 稍微多跑一点
        update_epochs=4,
        lr_pi=3e-5, 
        lr_v=1e-3, 
        clip_ratio=0.2,      # 稍微放宽一点 clip (0.1 -> 0.2)
        grad_clip=0.5, 
        ent_coef=0.001,      # 降低熵，减少无意义的随机探索
        # ================
        
        infeasible_penalty=0.1,
        P_AP_W=1.0, noise_figure_db=5.0, rho_p=10.0,
        area=500.0, d0=36.0, alpha_path=3.6,
        num_quantiles=50, fixed_positions=True, hidden=128, hidden_v=256,
        # 评估相关
        eval_runs=3,
        eval_temperature=None,
        eval_greedy_top=True
    )
    train(cfg)

