function min_rmse = func_yasuoauto_AD_brain_train_repeat_mse(x,repeat_num,train_data,testdata1,resSize,Phi,Psi,R_N)
% repeat func_train
%rmse_set = zeros(repeat_num,1);
min_rmse=11;
for repeat_i = 1:repeat_num
    %repeat_i=1;
   % rng_num=repeat_i*20000 + (now*1000-floor(now*1000))*100000;
    rng_num=randi(100);
    rng(rng_num,'twister')
%     rng(rng_num)
    rmse = func_yasuoauto_AD_brain_train_mse2(x,rng_num,train_data,testdata1,resSize,Phi,Psi,R_N);
    if rmse<=min_rmse
        min_rmse=rmse                                                                                                                         ;
        min_rng=rng_num;
%         min_rmse_maxly=max_ly;
    end   
end
% min_rng_set=[min_rng_set,min_rng];
% filename =['min_rng_set_S01_new_stru_brain_all_rc_500_train1_NC_' num2str(p_num) 'len_' num2str(len)  '.mat'];
% save(filename,'min_rng_set') 
% if exist('min_rng_set.mat', 'file')
%     load('min_rng_set.mat', 'min_rng_set');
% else
%     min_rng_set = []; % 如果文件不存在，初始化为空数组
% end
load min_rng_set.mat min_rng_set
min_rng_set=[min_rng_set,min_rng];
save min_rng_set.mat min_rng_set
% filename = sprintf('min_rng_set_S03_new_auto_mse1_stru_brain_all_rc_500_train1_NC_%d_len_%d.mat', p_num, len);
% % filename =['min_rng_set_S03_new_stru_brain_all_rc_500_train1_NC_' num2str(p_num) 'len_' num2str(len)  '.mat'];
% save(filename,'min_rng_set') 
end
% load min_rmse_dynamic_set.mat min_rmse_dynamic_set
% min_rmse_dynamic_set=[min_rmse_dynamic_set,min_rmse_dynamic];
% save min_rmse_dynamic_set.mat min_rmse_dynamic_set
% 
% fprintf('\nrmse_dynamic is %f\n',rmse_dynamics)
% fprintf('\nrmse is %f\n',rmse);