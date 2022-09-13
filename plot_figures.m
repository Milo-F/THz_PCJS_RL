x = 1:200;
eq_rate = eq_rate * 1e8;
op_rate = op_rate * 1e8;
%%
rate_th = 9*ones(1,200)*1e8;
%% 画对比图
f = figure(Name="milo");
left_color = [.15 .15 .15];
right_color = [.15 .15 .15];
set(f,'defaultAxesColorOrder',[left_color; right_color]);
xlabel("Epoch");
yyaxis left;
plot(x,op_rate,"bo-",x,eq_rate,"k^-",x,rate_th,"rd--",'MarkerIndices',1:8:length(op_rate),MarkerSize=5);
ylabel("Average Rate (bps)")
yyaxis right;
plot(x,op_pos_error,"m>-",x,eq_pos_error,'gs-','MarkerIndices',1:8:length(op_rate),MarkerSize=5)
ylabel("Position MSE (m)");
grid on;
legend("PPO Power Avg Rate","Equal Power Avg Rate","Rate Threshold","PPO Power Position MSE", "Equal Power Postion MSE")
MagInset(f, -1, [100,195 0.2 0.5],[60,190 1.5 4],{'NW','NW';'SE','SE'} );
%% 画横轴功率图
x_power = 0.2:0.2:1.4;
p_to_power = [0.5367 0.3794 0.3097 0.2682 0.2399 0.2190 0.2027];
r_to_power = [7.4007 8.4041 8.9902 9.4058 9.7281 9.9914 10.2139] * 1e8;
f_power = figure(Name="power");
left_color = [.15 .15 .15];
right_color = [.15 .15 .15];
set(f_power,'defaultAxesColorOrder',[left_color; right_color]);
xlabel("Power (W)");
yyaxis right;
plot(x_power, p_to_power,"bo-");
ylabel("Position MSE (m)")

yyaxis left;
plot(x_power, r_to_power,"m^-");
ylabel("Rate (bps)")
grid on;
legend("Rate", "Position MSE");