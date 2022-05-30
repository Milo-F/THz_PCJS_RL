# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:16:00 2022
    训练定位
@author: milo
"""

import torch
from networks import network
import numpy as np
import config as cfg
import matplotlib.pyplot as plt
import random

# 加载数据函数
def load_data(data_root_path, idx, epoch):
    data_path = data_root_path + "data_{}.npy".format(idx)
    lable_path = data_root_path + "lable_{}.npy".format(idx)
    data_loader = np.load(data_path)  
    lable_loader = np.load(lable_path)
    random.seed(idx+epoch)
    random.shuffle(data_loader)
    random.seed(idx+epoch)
    random.shuffle(lable_loader)
    return data_loader, lable_loader

# 训练函数 batch：100
def train(model):
    # 配置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Loss和Optimizer
    criterion = torch.nn.MSELoss() # 均方误差损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-6) # 优化函数
    loss_plot = []
    loss_mean = []
    # 训练
    epoch_num = 20
    total_step = 6*8000
    for epoch in range(epoch_num):
        for i_idx in range(1):
            data_root_path = "position_set/snr({})/train/".format(i_idx*5)
            for j_idx in range(8):
                data_loader, lable_loader = load_data(data_root_path, j_idx, epoch)
                for k_idx in range(len(data_loader)):
                    data_complex = torch.from_numpy(data_loader[k_idx,:]).to(device)
                    data = torch.cat((data_complex.real, data_complex.imag), 0).to(torch.float32)
                    lable = torch.from_numpy(lable_loader[k_idx,0:3]).to(device).to(torch.float32)
                    # forward pass
                    output = model(data)
                    loss = criterion(output, lable)
                    
                    # backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss_mean.append(loss.item())
                    # print loss
                    if (k_idx+1) % 500 == 0 :
                        loss_plot.append(np.mean(loss_mean))
                        loss_mean.clear()
                        print("Epoch: [{}/{}], Step: [{}/{}], Loss: {}"
                              .format(epoch+1, epoch_num, i_idx*8000+j_idx*1000+k_idx, total_step, loss.item()))
    # 保存模型参数
    
    torch.save(model.state_dict(), "./position_model_params.pkl")
    # 使用模型
    # new_model = network.Network() # 调用模型Model
    # new_model.load_state_dict(torch.load("./position_model_params.pkl")) # 加载模型参数     
    # new_model.forward(input) # 进行使用 
    return loss_plot                 
    
# 主函数   
if __name__ == "__main__":
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    print(torch.cuda.is_available())
    model = network.Network(in_dimension=2*(cfg.N+4), out_dimention=3)
    loss_plot = train(model)
    loss_x = [x for x in range(len(loss_plot))]
    plt.plot(loss_x, loss_plot)
    # 画图