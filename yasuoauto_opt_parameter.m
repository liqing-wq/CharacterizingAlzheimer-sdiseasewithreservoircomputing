%% config
clc;
clear all;
iter_max = 50;
repeat_num = 5; % ensemble average size
load S03_ADallData.mat; % 加载数据
% num_p=2;
len=100;
%len = size(allData, 1); % 获取第三维度的大小
num_p = size(allData, 3); % 获取第三维度的大小
% 参数范围
lb = [0 0 0 10^-8  0];
ub = [3 3 1 10^-3  1];
options = optimoptions('surrogateopt', 'MaxFunctionEvaluations', iter_max, 'PlotFcn', 'surrogateoptplot');
% 保存结果的数组或结构
all_results = struct();
%%
resSize=500;%num数据数目
compressrate=0.1;
R_N = floor(compressrate*resSize);%压缩后储备池结点个数
Phi= PartHadamardMtx(R_N,resSize);
fprintf("%d %d\n",R_N,resSize);
Psi = dwtmtx(resSize, 'haar', 1);
%% main loop over third dimension
for p_num = 1:num_p
    train_data = allData(1:len, :, p_num); % 获取训练数据
    testdata1 = train_data; % 设置测试数据为训练数据，按需修改
    min_rng_set=[];
    save min_rng_set.mat min_rng_set
    % 动态生成文件名
    filename = sprintf('opt_S03_new_autoyasuo_0.1_mse2_stru_brain_all_rc_500_train3_AD_%d_len_%d.mat', p_num, len);
    % 目标函数
    func = @(x) func_yasuoauto_AD_brain_train_repeat_mse(x, repeat_num,train_data,testdata1,resSize,Phi,Psi,R_N);
    % 运行优化
    tic;
    [opt_result, opt_fval, opt_exitflag, opt_output, opt_trials] = surrogateopt(func, lb, ub, options);
    toc;
     % 保存结果到文件
     save(filename) 
%     save(filename, 'opt_result', 'opt_fval', 'opt_exitflag', 'opt_output', 'opt_trials', 'min_rmse', 'min_rmse_maxly', 'min_rng');
    % 使用优化结果计算最小 RMSE 和相关值
     [min_rmse_maxly,min_rng,similarity_metric,PLV_ad,avg_cluster_rc,avg_cluster_rc_ys] = func_yasuoauto_AD_brain_train_test_mse(p_num, len,train_data,testdata1,resSize,Phi,Psi,R_N);

    % 保存当前 p_num 的结果
    all_results(p_num).avg_cluster_rc = avg_cluster_rc;
    all_results(p_num).avg_cluster_rc_ys = avg_cluster_rc_ys;
    all_results(p_num).PLV = PLV_ad;
%     all_results(p_num).opt_result = opt_result;
%     all_results(p_num).opt_fval = opt_fval;
%     all_results(p_num).opt_exitflag = opt_exitflag;
%     all_results(p_num).opt_output = opt_output;
%     all_results(p_num).opt_trials = opt_trials;
    all_results(p_num).similarity_metric = similarity_metric;
    all_results(p_num).min_rmse_maxly = min_rmse_maxly;
    all_results(p_num).min_rng = min_rng;

   
end

% 保存所有结果
save('all_opt_results_S03_AD_yasuo_0.3.mat', 'all_results');

% 如果在非个人电脑环境运行，结束运行
if ~ispc
    exit;
end
