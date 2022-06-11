#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: run_ppo.py
    @Author: Milo
    @Date: 2022/06/09 16:17:45
    @Version: 1.0
    @Description: ppo网络训练main函数
'''
import Constraints as cons
import torch
import Arguments as Arg
from networks import PPO
import RLEnv
import numpy as np


def main():
    hyper_params = vars(Arg.get_args())
    
    a_lr = hyper_params["actor_lr"]
    c_lr = hyper_params["critic_lr"]
    eps = hyper_params["eps"]
    epsilon = hyper_params["epsilon"]
    actor_update_step = hyper_params["actor_update_step"]
    critic_update_step = hyper_params["critic_update_step"]
    ep_num = hyper_params["episode_num"]
    ep_len = hyper_params["episode_len"]
    batch_size = hyper_params["batch_size"]
    reward_decay = hyper_params["reward_decay"]
    
    position = [100, 120, -10] # 位置
    sigma = 1e-8 # 噪声
    env = RLEnv.RLEnv(position, sigma)
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    ppo = PPO.PPO(state_dim, action_dim, a_lr, c_lr, eps, epsilon, actor_update_step, critic_update_step)
    p_list = [0,0]
    for ep in range(ep_num):
        all_ep_r = []
        
        s = env.reset()
        buf_s, buf_a, buf_r = [], [], []
        ep_r = 0
        
        for step in range(ep_len):
            tensor_s = torch.tensor(s, dtype=torch.float32)
            tensor_a = ppo.choose_action(tensor_s)  
            a = tensor_a.detach().numpy()          
            s_, rate, reward, crb = env.step(a)
            buf_s.append(s)
            buf_a.append(a)
            buf_r.append(reward)
            s = s_
            ep_r += reward
            
            if (step+1)%batch_size == 0 or step == ep_len-1:
                tensor_s_ = torch.tensor(s_, dtype=torch.float32)
                v_s_ = ppo.get_v(tensor_s_)
                
                buf_R = []
                for r in buf_r[::-1]:
                    v_s_ = r + reward_decay * v_s_
                    buf_R.insert(0, v_s_)
                
                bs = torch.tensor(buf_s, dtype=torch.float32)
                ba = torch.tensor(buf_a, dtype=torch.float32)
                bR = torch.tensor(buf_R, dtype=torch.float32)
                buf_s.clear()
                buf_a.clear()
                buf_r.clear()
                print("开始{}训练, {}".format(step, ep_r))
                ppo.update(bs, ba, bR)
                
        # if ep == 0:
        #     all_ep_r.append(ep_r)
        # else:
        #     all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1)
        # print(
        #     'Episode: {}/{}  | Episode Reward: {:.4f}  | Running Time: {:.4f}'.format(
        #         ep, ep_num, ep_r
        #     )
        # )
            
    if hyper_params["train"]:
        print("aaa")

if __name__ == "__main__":
    main()
