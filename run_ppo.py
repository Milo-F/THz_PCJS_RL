#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
    @File: run_ppo.py
    @Author: Milo
    @Date: 2022/06/09 16:17:45
    @Version: 1.0
    @Description: ppo网络训练
'''
import random
import torch
import Arguments as Arg
from networks.PPO import PPO
from enviroment import Constraints as cons
from enviroment.Test_Env import Test_Env
from enviroment.Env3D import Env3D
from enviroment.env2d.Env2D import Env2D
from enviroment.env2d.Env2DMinCrb import Env2DMinCrb
import Tools
import numpy as np
import math


def main():
    # 创建文件夹
    Tools.create_folder()
    # 随机种子
    sd = 0
    np.random.seed(sd)
    torch.manual_seed(sd)
    random.seed(sd)
    # 获取超参数
    hyper_params = vars(Arg.get_args())

    # 超参数
    a_lr = hyper_params["actor_lr"]  # actor网络学习率
    c_lr = hyper_params["critic_lr"]  # critic网络学习率
    eps = hyper_params["eps"]
    epsilon = hyper_params["epsilon"]  # PPO2网络重要性截断系数
    actor_update_step = hyper_params["actor_update_step"]  # actor网络训练更新步数
    critic_update_step = hyper_params["critic_update_step"]  # critic网络训练更新步数
    ep_num = hyper_params["episode_num"]  # 训练epoch数
    ep_len = hyper_params["episode_len"]  # 每个epoch训练step数
    batch_size = hyper_params["batch_size"]
    reward_decay = hyper_params["reward_decay"]  # 奖励折扣

    #
    Tools.print_log("INIT ENVIROMENT")
    ############ 测试环境 #####################################################
    # env = Test_Env()  # 创建测试环境验证网络收敛性
    ############ 3D通信环境 #####################################################
    # position = [50, 50, -10]
    # sigma = 5e-8
    # env = Env3D(position, sigma) # 创建通信定位一体化环境
    ##########################################################################
    ############ 2D通信环境 #####################################################
    # position = [100, math.pi/4]
    # sigma = 1e-8
    # env = Env2D(position, sigma) # 创建通信定位一体化环境
    ##########################################################################
    ############ 2D最小化CRB环境 ##########################################
    position = [100, math.pi/4]
    sigma = 1e-8
    env = Env2DMinCrb(position, sigma) # 创建通信定位一体化环境
    ###################################################################

    state_dim = env.state_dim  # 从环境获取状态维度
    action_dim = env.action_dim  # 从环境获取动作维度

    Tools.print_log("INIT PPO2 NETWORK")
    # 创建PPO网络
    ppo = PPO.PPO(state_dim, action_dim, a_lr, c_lr, eps,
                  epsilon, actor_update_step, critic_update_step)

    # 用于画图的数据
    ep_error_list = []
    ep_rate_list = []
    ep_reward_list = []

    Tools.print_log("START TRAINING")
    # 训练
    for ep in range(ep_num):

        ##########################################################################
        # 初始化环境获得初始状态
        s = env.reset()
        ##########################################################################

        buf_s, buf_a, buf_r = [], [], []

        # 用于计算epoch平均信息的变量
        ep_r = 0
        ep_rate = 0
        ep_error = 0

        # 开始一个epoch的训练
        for step in range(ep_len):
            # 转化状态张量
            tensor_s = torch.tensor(s, dtype=torch.float32)
            # 获取动作张量
            tensor_a = ppo.choose_action(tensor_s)
            # 获取动作========================================
            a = tensor_a.detach().numpy()
            # 等功率分配======================================
            # a = [1,1]
            
            ##########################################################################
            # 将动作与环境互动获得下一个状态与动作奖励
            s_, reward = env.step(a)
            rate = env.rate_eff
            p_error = env.p_error
            ##########################################################################
            
            # 保存状态、动作和奖励
            buf_s.append(s)
            buf_a.append(a)
            buf_r.append(reward)
            # 更新状态
            s = s_
            
            # 累加每一步的奖励
            ep_r += reward
            ep_rate += rate
            ep_error += p_error

            # 进度
            Tools.progress((step+1)/ep_len)
            
            # buffer中存满一个batch的数据或者结束了就训练
            if (step+1) % batch_size == 0 or step == ep_len-1:
                # 获得下一个状态的价值
                tensor_s_ = torch.tensor(s_, dtype=torch.float32)
                v_s_ = ppo.get_v(tensor_s_)

                # 缓存累计奖励
                buf_R = []
                for r in buf_r[::-1]:
                    v_s_ = r + reward_decay * v_s_
                    buf_R.insert(0, v_s_.numpy()[0])

                # 将一个batch的训练采样数据进行训练
                bs = torch.tensor(np.array(buf_s), dtype=torch.float32)
                ba = torch.tensor(np.array(buf_a), dtype=torch.float32)
                bR = torch.tensor(np.array(buf_R), dtype=torch.float32)

                # 清空batch buffer
                buf_s.clear()
                buf_a.clear()
                buf_r.clear()

                # 训练网络
                ppo.update(bs, ba, bR)
                
            # 一个EPOCH训练结束，打印相关的信息
            if step == ep_len-1:
                print(
                    "Epoch: {}".format(ep)
                    + "\tReward: {:.4f}".format(ep_r/ep_len)
                    + "\tRate: {:.4f}".format(ep_rate/ep_len)
                    + "\tError: {:.4f}".format(ep_error/ep_len)
                    + "\tPos Power: {:.4f}".format(a[0]*cons.BETA_P)
                    + "\tCom Power: {:.4f}".format(a[1]*cons.BETA_C)
                )

        # 每个epoch需要保存的信息
        ep_rate_list.append(ep_rate/ep_len)
        ep_reward_list.append(ep_r/ep_len)
        ep_error_list.append(ep_error/ep_len)
        
    # 保存配置日志
    dg = "独立功率限制0.9，速率门限9，同时更新估计位置"
    Tools.save_log(hyper_params, dg)
        
    # 画图
    Tools.plot_fig(ep_reward_list, "epoch", "average reward", "PPO_avg_reward")
    Tools.plot_fig(ep_rate_list, "epoch", "average rate", "PPO_avg_rate")
    Tools.plot_fig(ep_error_list, "epoch", "position error", "PPO_avg_position_error")


if __name__ == "__main__":
    main()
