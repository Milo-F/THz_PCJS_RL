#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: run_ppo.py
    @Author: Milo
    @Date: 2022/06/09 16:17:45
    @Version: 1.0
    @Description: ppo网络训练main函数
'''
from math import ceil
import Constraints as cons
import torch
import Arguments as Arg
from networks import PPO
import RLEnv
from test_env import Test_Env
import numpy as np
from matplotlib import pyplot as plt

def print_log(str:str):
    s_tmp = (40-ceil(len(str)/2))
    e_tmp = s_tmp
    if (len(str)%2):
        e_tmp = e_tmp +1
    print("="*s_tmp + str + "="*e_tmp)

def plot_jpg(y:list, x_str:str, y_str:str, fig_name:str):
    x = [x for x in range(len(y))]
    plt.plot(x, y)
    plt.xlabel(x_str)
    plt.ylabel(y_str) 
    plt.savefig("{}.jpg".format(fig_name))

def main():
    hyper_params = vars(Arg.get_args())
    
    # 超参数
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
    # 
    print_log("INIT ENVIROMENT")
    env = Test_Env()
    
    state_dim = env.state_dim
    action_dim = env.action_dim
    
    print_log("INIT PPO2 NETWORK")
    ppo = PPO.PPO(state_dim, action_dim, a_lr, c_lr, eps, epsilon, actor_update_step, critic_update_step)
    
    ep_x_list = []
    
    print_log("START TRAINING")
    for ep in range(ep_num):
        
        s = env.reset()
        buf_s, buf_a, buf_r = [], [], []
        ep_r = 0
        ep_x = 0
        
        for step in range(ep_len):
            tensor_s = torch.tensor(s, dtype=torch.float32)
            tensor_a = ppo.choose_action(tensor_s)  
            a = tensor_a.detach().numpy()          
            s_, reward, x = env.step(a)
            buf_s.append(s)
            buf_a.append(a)
            buf_r.append(reward)
            s = s_
            ep_r += reward
            ep_x += x
            
            
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
                ppo.update(bs, ba, bR)
                
            # 一个EPOCH训练结束，打印相关的信息
            if step == ep_len-1:
                print(
                    "Episode: {}".format(ep)
                    +"\tEpisode Reward: {:.4f}".format(ep_r/ep_len)
                    +"\tRate: {:.4f}".format(ep_x/ep_len) 
                    + "\tPosition Power: {:.4f}".format(a[0]*6)
                    + "\tCom Power: {:.4f}".format(a[1]*6)
                    )
        ep_x_list.append(ep_x/ep_len)       

    plot_jpg(ep_x_list, "epoch", "average reward", "ppo_avg_reward")
    
if __name__ == "__main__":
    main()
