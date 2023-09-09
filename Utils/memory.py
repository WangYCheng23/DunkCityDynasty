import numpy as np
import torch

def compute_advantages(rs, dones, values):
    gamma = 0.95
    lambd = 0.95
    advantages = []
    returns = []
    adv = 0
    for tt in range(len(rs)-1, -1, -1):
        m = 1 - int(dones[tt])
        delta = rs[tt] + gamma * values[tt+1] * m - values[tt]
        adv = delta + gamma * lambd * m * adv
        R = adv + values[tt]
        advantages.append(adv)
        returns.append(R)
    advantages.reverse()
    returns.reverse()
    returns = torch.tensor(np.array(returns))
    advantages = torch.tensor(np.array(advantages))
    return returns, advantages

class ReplayMemory:
    def __init__(self, dtype):
        self.dtype = dtype
        self.clear()
    def add(self, obs, act, act_logprob, r, done, value):
        self.obs.append(obs)
        self.act.append(act)
        self.act_logprob.append(act_logprob)
        self.r.append(r)
        self.done.append(done)
        self.value.append(value)
    def clear(self):
        self.obs = []
        self.act = []
        self.act_logprob = []
        self.r = []
        self.done = []
        self.value = []
    def sample(self, ctx_size, batch_size, last_value):
        M = len(self.r)
        i = np.random.choice(range(ctx_size - 1, M), batch_size)

        returns, advantages = compute_advantages(self.r, self.done, self.value + [last_value])
        returns = np.array(returns)[i]
        advantages = np.array(advantages)[i]

        obs_np = np.array(self.obs)
        act_np = np.array(self.act)
        act_logprob_np = np.array(self.act_logprob)
        obs_ctx = []
        for ctx_idx in range(ctx_size - 1, -1, -1):
            obs_ctx.append(obs_np[i - ctx_idx])
        obs_ctx = np.stack(obs_ctx, axis=1)

        obs_ctx = torch.tensor(obs_ctx, dtype=self.dtype)
        act = torch.tensor(act_np[i], dtype=self.dtype)
        act_logprob = torch.tensor(act_logprob_np[i], dtype=self.dtype)
        returns = torch.tensor(returns, dtype=self.dtype)
        advantages = torch.tensor(advantages, dtype=self.dtype)
        return obs_ctx, act, act_logprob, returns, advantages