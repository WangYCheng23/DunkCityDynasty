import time
import numpy as np
import ray
import gymnasium
import torch

from Utils.config import readParser
from DunkCityDynasty.env.gym_env import GymEnv
from Agents.random_agent import RandomAgent

ray.init(num_cpus=2, num_gpus=1, local_mode=False)
    
@ray.remote(num_cpus=1,num_gpus=0.5)
class Worker:
    def __init__(self, id, params):
        params['id'] = id
        params['xvfb_display'] = id+5
        params['rl_server_port'] = params['rl_server_ports'][params['id']]
        self.batch_size = params['batch_size']
        self.ctx_size = params['ctx_size']
        self.dtype = params['dtype']
        self.env = GymEnv(params)
        self.d_obs = self.env.observation_space.shape[0]
        self.d_act = self.env.action_space.shape[0]
        # self.low = self.env.action_space.low
        # self.high = self.env.action_space.high
        # 用于决策的神经网络
        self.agent = RandomAgent()
        # self.agent = AgentModel(self.ctx_size, self.d_obs, self.d_act, self.low, self.high)
        # self.opt = torch.optim.Adam(self.agent.parameters(), params['lr'])
        # self.memory = ReplayMemory(self.dtype)
        # self.context = Context(self.ctx_size, self.d_obs, self.dtype)
        self.T = 0
        self.rewards = [0.0]
        self.done = False
        
    def get_info_dims(self):
        return self.d_obs, self.d_act
    
    def get_weights(self):
        # 异步收集每个worker的权重用于平均
        return self.agent.state_dict()
    
    def get_avg_reward(self):
        # 异步收集当前任务成功率等信息
        avg_reward_finished = np.mean(self.rewards[-4:])
        return avg_reward_finished
    
    def train_get_weights_infos(self):
        # 合并多个异步收集任务，防止时间不同步
        if self.T > self.ctx_size:
            # 若episode时间够长，则训练
            self.train_policy()
            self.memory.clear()
        return self.get_weights(), self.get_avg_reward(), self.context.normalizer
    
    def set_weights(self, w):
        # 为每个worker分发平均后的权重
        self.agent.load_state_dict(w)
        
    def set_normalizer(self, normalizer):
        self.context.set_normalizer(normalizer)
        
    def reset_initialize(self):
        # 初始化仿真环境，上下文和log信息
        self.context.reset()
        obs, _ = self.env.reset()
        self.context.add(obs)
        self.T = 0
        
    def train_policy(self):
        # episode结束，训练策略网络
        n_batches = int(self.T / self.batch_size) + 1

        obs_ctx = self.context.get() # 获取状态上下文
        _, _, _, last_value = self.agent(obs_ctx) # 获取不完全轨迹最后一个value用于bootstrap
        last_value = float(last_value.detach().numpy()[0])

        # 计算Generalized Advantage Estimation
        # self.opt.zero_grad()
        # obs_ctxs, acts, act_logprobs, returns, advantages = self.memory.sample(self.ctx_size, self.batch_size, last_value)   # 从重放记忆中采样经验
        # for _ in range(n_batches):
        #     _, _, pred_act_dist, pred_value = self.agent(obs_ctxs)
        #     loss = ppo_loss(pred_act_dist, pred_value, acts, act_logprobs, returns, advantages)
        #     (loss/n_batches).backward()
        # self.opt.step()
        
    def rollout(self, T_rollout):
        # 仿真循环，一直展开仿真到done为True
        for _ in range(T_rollout):
            if self.done:
                self.done = False
                self.reset_initialize()
            obs_ctx = self.context.get() # 获取状态上下文

            # 根据状态上下文决策，得到动作，概率，和价值
            act, act_logprob, _, value = self.agent(obs_ctx)
            act = act.detach().numpy()[0]
            act_logprob = float(act_logprob.detach().numpy()[0])
            value = float(value.detach().numpy()[0])

            # 仿真一步
            obs_, r, terminated, truncated, _ = self.env.step(act)
            self.done = terminated or truncated

            # 将历史经验加入重放记忆中
            self.memory.add(self.context.obs_ctx[-1], act, act_logprob, r, self.done, value)
            # 将需要累积的状态向量加入上下文
            self.context.add(obs_)
            self.rewards.append(r)
            self.T += 1
        return  
@ray.remote
class WorkerCaller:
    def __init__(self, workers, rollout_steps):
        # 设置一个对应的worker
        self.workers = workers
        self.n_workers = len(workers)
        self.rollout_steps = rollout_steps
    def start(self):
        # 对workers持续不断地触发rollout函数
        finish_indicators = [worker.rollout.remote(self.rollout_steps) for worker in self.workers]
        while True:
            for i in range(self.n_workers):
                if is_ready(finish_indicators[i]):
                    finish_indicators[i] = self.workers[i].rollout.remote(self.rollout_steps)

def is_ready(obj):
    ready_oids, _ = ray.wait([obj])
    if ready_oids:
        return True
    else:
        return False

def run_parallel():
    params = readParser()
    n_episodes = params['n_episodes']
    n_workers = params['n_workers']

    # 初始化worker
    workers = [Worker.options(name=f'worker-{id}',get_if_exists=True).remote(id, params) for id in range(n_workers)]
    # avg_weight = ray.get(workers[0].get_weights.remote())
    d_obs, d_act = ray.get(workers[0].get_info_dims.remote())
    ray.get([worker.reset_initialize.remote() for worker in workers])

    # 初始化标准化器
    # normalizer = Normalizer((d_obs,), np.float32)

    # 初始化持续调用worker的caller
    worker_caller = WorkerCaller.remote(workers, params['rollout_steps'])

    # 启动worker的caller，开始持续异步触发worker的rollout函数
    worker_caller.start.remote()
    time.sleep(1)

    # 主循环
    for i_episodes in range(n_episodes):
        # 收集worker的权重，只要有一个未收集完就会阻塞在这里
        weights_infos = ray.get([worker.train_get_weights_infos.remote() for worker in workers])
        workers_weights, workers_reward, workers_normalizer = zip(*weights_infos)
        # 计算平均权重
        avg_weight = {k:sum([workers_weights[wid][k] for wid in range(n_workers)])/n_workers for k in avg_weight.keys()}
        # 收集标准化器信息
        # normalizer.aggregate_collection(workers_normalizer)

        # 非阻塞异步地分发权重给每个worker
        finish_setting_indicator = []
        # for worker in workers:
            # finish_setting_indicator.append(worker.set_weights.remote(avg_weight))
            # finish_setting_indicator.append(worker.set_normalizer.remote(normalizer))
        ray.get(finish_setting_indicator)

        # 处理所有worker的log信息
        avg_reward = sum(workers_reward)/n_workers
        print(avg_reward)
        time.sleep(0.5)
        
        
if __name__ == '__main__':
    run_parallel()  