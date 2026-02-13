function [max_ly,rng_num,similarity_metric,PLV_ad,avg_cluster_rc,avg_cluster_rc_ys] = func_yasuoauto_AD_brain_train_test_mse(p_num, len,train_data,testdata1,resSize,Phi,Psi,R_N)
filename = sprintf('opt_S03_new_autoyasuo_0.3_mse2_stru_brain_all_rc_500_train3_AD_%d_len_%d.mat', p_num, len);
load(filename)
load min_rng_set.mat min_rng_set
filename1 = sprintf('min_rng_set_S03_new_autoyasuo_0.3_mse2_stru_brain_all_rc_500_train3_AD_%d_len_%d.mat', p_num, len);
save(filename1,'min_rng_set') 
load(filename1)
%%
result = getfield ( opt_trials,'Fval');
parameter= getfield ( opt_trials,'X');
[sort_result,result_num]=sort(result);
parameter=parameter(result_num,:);
opt_result=parameter(1,:);
% rng_num=min_rng_set;
rng_num=min_rng_set(result_num(1,:));
% select_num=23;
% train_data = func_fmri_train_AD_data();
% [train_data,testdata1] = func_fmri_train_structure_data();
% resize_set=[300;400;500;600;700];
% for rc_nn=1:length(resize_set)
% resSize=500;
rng(rng_num)
eig_rho =opt_result(1);
W_in_a =opt_result(2);
a = opt_result(3);
reg = opt_result(4);
density =opt_result(5);
% load S01_NCallData.mat
% p_num=3;
% train_data=allData(:, :,p_num);
% train_len=170;
% train_data=train_data(1:train_len,:);
% testdata1=train_data;
outSize = size(train_data,2);
inSize =  size(train_data,2); 
% perLen=10;
trainLen=size(train_data,1)-1;
test_Len=size(testdata1,1);
testlen=test_Len;
% trainLen=perLen*data_num;
% trainLen = perLen*data_num;
% intLen=1;
%  for i=1:data_num-1
%     Ydata((i-1)*perLen+1:i*perLen,:)=train_data((i-1)*tra_Len+1:(i-1)*tra_Len+perLen,:);
%     %testdata1((i-1)*testlen+1:i*testlen,1)=train_data((i-1)*tra_Len+perLen+1:(i-1)*tra_Len+perLen+testlen,1);
%  end   
%  Ydata((data_num-1)*perLen+1:data_num*perLen+exLen,:)=train_data((data_num-1)*tra_Len+1:(data_num-1)*tra_Len+perLen+exLen,:);
% testdata1((data_num-1)*testlen+1:data_num*testlen+exLen,:)=train_data((data_num-1)*tra_Len+perLen+1:(data_num-1)*tra_Len+perLen+testlen+exLen,:);
Ydata=train_data;
indata=Ydata(:,1:inSize);
outdata=Ydata(:,1:outSize);
% testdata1=train_data;
% Ydata=train_data;
% indata=Ydata;
% outdata=Ydata;
% resSize =400; % size of the reservoir nodes;  
Win=(2.0*rand(resSize, inSize)-1.0)*W_in_a;
%WW=sprandsym(resSize, density);
WW = zeros(resSize,resSize);
for i=1:resSize
    for j=i:resSize
           if (rand()<density)
           WW(i,j)=(2.0*rand()-1.0);
            WW(j,i)=WW(i,j);
           end
   end
end
rhoW = abs(eigs(WW,1));
W = WW .* (eig_rho /rhoW); % the spectral radius is 0.1:
% X = zeros(inSize+resSize,trainLen);
X = zeros(inSize+R_N,trainLen);
% set the corresponding target matrix directly
 Yt = outdata(2:trainLen+1,:)';
% Yt = outdata(1,2:trainLen+1);
% run the reservoir with the data and collect X
%x = (2.0*rand(resSize,1)-1.0)*0.5;
x = zeros(resSize,1);
% bias=ones(resSize,1);
for t = 1:trainLen
    u = indata(t,:)';
    x = (1-a)*x + a*tanh( Win*u + W*x );
    s = Psi*x;
    CS_x = Phi*s;
    X(:,t) = [u;CS_x];
%   X(:,t) = [x;x.^2];
%     X(:,t) = [u;x];
%     X(:,t) = x;
%     X(1:2:end,t) = X(1:2:end,t).^2;  
end
X_T = X';
% Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
Wout = Yt*X_T / (X*X_T + reg*eye(inSize+R_N));

M=testlen+1000;
num_lyaps=80;
u=testdata1(1,:)';
x = zeros(resSize,1);
delta=orth(rand(resSize+ inSize,num_lyaps));
% Q = eye(resSize + inSize); % 初始化Q为单位矩阵，尺寸应匹配J的尺寸
% Rdiag = zeros(resSize + inSize, M-1); % 准备接收R矩阵的对角线元素，M是迭代次数
% sum1 = 0; % 初始化用于计算第一个李雅普诺夫指数的和
% sum2 = 0; % 初始化用于计算第二个李雅普诺夫指数的和
% sum3 = 0; 
Sum=0;
norm_time=10;
% sum_record = zeros(3, ceil((M-1)/record_step)); % 预先分配记录数组
for i = 1:M-1         %迭代
    x = (1-a)*x + a*tanh( Win*u + W*x );
    X1(:,i)=x;
    s = Psi*x;
    CS_x = Phi*s;
    X_cs(:,i)=CS_x;
    y = Wout*[u;CS_x];
    Y1(:,i)=y;
    u=y;
    ys=Phi*Psi;
    df= 1-tanh(Win*u + W*x).*tanh(Win*u + W*x);
    J1=(1-a)*eye(resSize)+a*bsxfun(@times,df,W);
    J2=a*bsxfun(@times,df,Win);
    J3=(1-a)*Wout(:,inSize+1:end)*ys+a*Wout(:,inSize+1:end)*ys*bsxfun(@times,df,W);
    J4=Wout(:,1:inSize)+a*Wout(:,inSize+1:end)*ys*bsxfun(@times,df,Win);
    J=[J1,J2;J3,J4];
    delta=J*delta;
    if mod(i,norm_time)==0
        [Q,R] = qr(delta,0);
        delta = Q(:,1:num_lyaps);
%         Rdiag(:,i+1) = diag(R(1:num_lyaps,1:num_lyaps));   
        R_ii(:,i/norm_time) = log(diag(R(1:num_lyaps,1:num_lyaps)));  
        Sum=Sum+(R_ii(:,i/norm_time));
        sum_record(:,i/norm_time)=real(Sum)./(i);
    end
end
LE=real(Sum)./(M);
% le=sum1/(M*0.01);
% % LE(1) = sum1/M; LE(2) = sum2/M; LE(3) = sum3/M;
max_ly=max(LE);
correlationMatrix_real=corr(train_data);
correlationMatrix_rc=corr(Y1');
% frobenius_norm = norm(correlationMatrix_real - correlationMatrix_rc, 'fro');
similarity_metric = corr2(correlationMatrix_rc, correlationMatrix_real);

for i = 1:size(Y1, 1)
    for j = i:size(Y1, 1)
        % 计算两个脑区之间的PLV
        analytic_signal1 = hilbert(Y1(i, :));
        analytic_signal2 = hilbert(Y1(j, :));
        % 提取相位信息
        phase1 = angle(analytic_signal1);
        phase2 = angle(analytic_signal2);
        % 计算相位差
        phase_diff = phase1 - phase2;
        plv_matrix(i, j) = abs(mean(exp(1j*phase_diff)));
        plv_matrix(j, i) = plv_matrix(i, j); % 由于PLV是对称的，我们只需计算上三角部分
    end
end
PLV_ad=plv_matrix;

correlationMatrix_rcstate=corr(X1');
correlationMatrix_rcstate_ys=corr(X_cs');
threshold = 0.6; % 相似性阈值，用于确定网络中的边
adjacencyMatrix_rcstate = correlationMatrix_rcstate > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
adjacencyMatrix_rcstate_ys = correlationMatrix_rcstate_ys > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
adjMatrix=adjacencyMatrix_rcstate;
numNodes = size(adjMatrix, 1); % 节点数量
clusteringCoeffs = zeros(numNodes, 1); % 初始化聚类系数数组
for i = 1:numNodes
    neighbors = find(adjMatrix(i, :) > 0); % 找到节点i的所有邻居
    if length(neighbors) < 2
        continue; % 如果邻居少于两个，则聚类系数为0
    end
    subGraph = adjMatrix(neighbors, neighbors); % 提取邻居子图
    totalLinks = sum(subGraph(:)) / 2; % 计算子图中的边数
    maxLinks = length(neighbors) * (length(neighbors) - 1) / 2; % 计算最大可能的边数
    clusteringCoeffs(i) = totalLinks / maxLinks; % 计算聚类系数
end
avgClusteringCoeff = mean(clusteringCoeffs); % 计算平均聚类系数
avg_cluster_rc=avgClusteringCoeff;

adjMatrix=adjacencyMatrix_rcstate_ys;
numNodes = size(adjMatrix, 1); % 节点数量
clusteringCoeffs = zeros(numNodes, 1); % 初始化聚类系数数组
for i = 1:numNodes
    neighbors = find(adjMatrix(i, :) > 0); % 找到节点i的所有邻居
    if length(neighbors) < 2
        continue; % 如果邻居少于两个，则聚类系数为0
    end
    subGraph = adjMatrix(neighbors, neighbors); % 提取邻居子图
    totalLinks = sum(subGraph(:)) / 2; % 计算子图中的边数
    maxLinks = length(neighbors) * (length(neighbors) - 1) / 2; % 计算最大可能的边数
    clusteringCoeffs(i) = totalLinks / maxLinks; % 计算聚类系数
end
avgClusteringCoeff = mean(clusteringCoeffs); % 计算平均聚类系数
avg_cluster_rc_ys=avgClusteringCoeff;
end
% iterations = (1:length(sum_record))*record_step;
% plot(sum_record(1, :), 'r-', 'LineWidth', 2);
% hold on;
% plot(sum_record(2, :), 'g-', 'LineWidth', 2);
% plot(sum_record(3, :), 'b-', 'LineWidth', 2);
% hold off;
% % ylim([-0.02 0.055]);
% % 设置图形参数
% title('Liapunov Exponents Sum over Iterations');
% xlabel('Iteration');
% ylabel('Sum of Logarithms of Rdiag Elements');
% legend('le(1)', 'le(2)', 'le(3)');
% grid on;

% x=train_data(:,:);
% correlationMatrix = corr(x); % Step 1:使用corr函数计算相关系数矩阵
% % Step 2: 构建相似性网络 
% threshold = 0.6; % 相似性阈值，用于确定网络中的边
% adjacencyMatrix = correlationMatrix > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
% A=adjacencyMatrix;
% % A=correlationMatrix;
% figure
% imagesc(A);
% hTitle=title('real data');
% x1=Y1(:,:)';
% correlationMatrix = corr(x1); % Step 1:使用corr函数计算相关系数矩阵
% % Step 2: 构建相似性网络 
% threshold = 0.6; % 相似性阈值，用于确定网络中的边
% adjacencyMatrix = correlationMatrix > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
% A1=adjacencyMatrix;
% % A1=correlationMatrix;
% figure
% imagesc(A1);
% hTitle=title('RC');
% error_fc=abs(A-A1);
% figure
% imagesc(error_fc);
% hTitle=title('error');
% colorbar
% 
% figure;
% % 循环绘制每个时间序列
% hold on; % 保持图形窗口打开，以便将所有线条绘制在同一张图上
% for i = 1:size(Y1,1)
%     plot(Y1(i, :),'LineStyle', '-', 'linewidth',1.5); % 绘制第 i 个时间序列
% end
% hold off; % 关闭保持，图形窗口恢复正常行为
% % xlim([0 1050]);
% xlabel('step');
% ylabel('y');
% set(gca,'FontName','Times New Roman','FontSize',18);
% % axis tight; % 紧凑显示轴，减少空白区域
% box on; % 添加边框
% % % end
% %%
% Y1_first_part = Y1(:, 1:testlen);
% Y1_second_part = Y1(:, testlen+1:end);
% % Y1_second_part = Y1(:, testlen+1:end);
% dt=2;
% Fs = 1/dt;          % 采样频率（每秒采样点数）
% % T = ;              % 采样时长（秒）
% % t = 0:1/Fs:T-1/Fs;  % 时间向量
% % f1 = 5;             % 第一个频率成分（Hz）
% % f2 = 20;            % 第二个频率成分（Hz）
% % x = sin(2*pi*f1*t) + 0.5 * sin(2*pi*f2*t);  % 示例信号，包含两个频率成分
% figure
% for dim=1:size(Y1_first_part,1)
% x1=Y1_first_part(dim,:);
% x1=x1-mean(x1);
% % 进行傅里叶变换
% N = length(x1);      % 数据点数
% X = fft(x1);          % 执行快速傅里叶变换
% % 计算频谱
% f = (0:N-1)*(Fs/N);  % 频率范围
% f1=f(1:(ceil(N/2))+1);
% P = abs(X/N);        % 频域幅度
% P = P(1:N/2+1);      % 只保留正频谱部分
% Pow1=P.*conj(P);
% 
% subplot(2,1,1);
% plot(f1, Pow1,'linewidth',2);
% [pks1, locs1] = findpeaks(Pow1(:, :), 'SortStr', 'descend', 'NPeaks', 1);
% plot(f1(locs1), pks1, 'ro', 'MarkerFaceColor', 'r');
% 
% % title('node69');
% xlabel('f (Hz)');
% ylabel('Pow');
% % xlim([0 0.25])
% % ylim([-0.01 0.trainlen6])
% title(['FFT of trainlen Part ']);
% set(gca,'XColor','k','YColor','k','linewidth',3,'fontsize',8);
% set(gca,'FontSize',15,'Fontname', 'Times New Roman');
% set(gca,'box','on');
% hold on;
% 
% 
% x2=Y1_second_part(dim,:);
% x2=x2-mean(x2);
% % 进行傅里叶变换
% N = length(x2);      % 数据点数
% X= fft(x2);          % 执行快速傅里叶变换
% % 计算频谱
% f = (0:N-1)*(Fs/N);  % 频率范围
% f2=f(1:(ceil(N/2)));
% P = abs(X/N);        % 频域幅度
% P = P(1:N/2+1);      % 只保留正频谱部分
% Pow2=P.*conj(P);
% if any(isnan(Pow1)) || any(isnan(Pow2))
%     % 如果其中一个序列包含NaN，则设置distance为100
%     distance(dim) = 20;
% else
%     % 如果没有NaN，计算DTW距离
%     distance(dim) = dtw(Pow1,Pow2);
% end
% 
% subplot(2,1,2);
% plot(f2, Pow2,'linewidth',2);
% [pks2, locs2] = findpeaks(Pow2(:, :), 'SortStr', 'descend', 'NPeaks', 1);
% plot(f2(locs2), pks2, 'ro', 'MarkerFaceColor', 'r');
% 
% % title('node69');
% xlabel('f (Hz)');
% ylabel('Pow');
% % xlim([0 0.25])
% % ylim([-0.01 0.6])
% title(['FFT of auto Part ']);
% set(gca,'XColor','k','YColor','k','linewidth',3,'fontsize',8);
% set(gca,'FontSize',15,'Fontname', 'Times New Roman');
% set(gca,'box','on');
% hold on;
% end
% figure
% plot(distance);
%%
% 假设 Y1 和 testlen 已经定义
% 假设 Y1 的大小是 246*(testlen+1000)
% 假设 Y1 和 testlen 已经定义
% % 假设 Y1 的大小是 246*(testlen+1000)
% Fs = 0.5;  % 采样频率
% numDimensions = size(Y1, 1);  % 数据维度数
% 
% % 提取时间序列
% x = Y1(:, 1:testlen);
% y = Y1(:, testlen+1:end);
% 
% % 确保 x 和 y 的长度相同
% if size(y, 2) > size(x, 2)
%     y = y(:, 1:size(x, 2));
% elseif size(y, 2) < size(x, 2)
%     x = x(:, 1:size(y, 2));
% end
% % 计算Hilbert变换来获取瞬时相位
% phi_x = angle(hilbert(x));
% phi_y = angle(hilbert(y));
% 
% % 初始化PLV矩阵
% plv = zeros(1, numDimensions);
% % 初始化互信息矩阵
% mi = zeros(1, numDimensions);
% 
% % 计算每个维度的互信息
% for i = 1:numDimensions
%     % 离散化数据
%     plv(i) = abs(mean(exp(1i * (phi_x(i, :) - phi_y(i, :)))));
%     numBins = floor(sqrt(length(x(i, :))));  % 使用经验公式选择bins的数量
%     x_discrete = discretize(x(i, :), numBins);
%     y_discrete = discretize(y(i, :), numBins);
% 
%     % 计算互信息
%     mi(i) = mutualinfo(x_discrete, y_discrete);
% end
% 
% figure;
% plot(1:numDimensions, plv);
% title('Mutual Information Across Dimensions');
% xlabel('Dimension');
% ylabel('Mutual Information');
% grid on;
% 
% % 可视化互信息
% figure;
% plot(1:numDimensions, mi);
% title('Mutual Information Across Dimensions');
% xlabel('Dimension');
% ylabel('Mutual Information');
% grid on;
% function I = mutualinfo(x, y)
%     % Joint histogram
%     jointHist = histcounts2(x, y);
%     % Convert joint histogram to joint probability distribution
%     jointProb = jointHist / Sum(jointHist, 'all');
%     % Marginal probabilities
%     px = Sum(jointProb, 1);  % Sum over rows to get p(x)
%     py = Sum(jointProb, 2);  % Sum over columns to get p(y)
%     % Entropy calculations
%     Hx = -Sum(px .* log2(px + eps));  % Entropy of x
%     Hy = -Sum(py .* log2(py + eps));  % Entropy of y
%     Hxy = -Sum(jointProb .* log2(jointProb + eps), 'all');  % Joint entropy
%     % Mutual information
%     I = Hx + Hy - Hxy;
% end
% figure;
% plot(f, coherenceValues(:,1), 'b');
% xlabel('Frequency (Hz)');
% ylabel('Average Coherence');
% title('Average Magnitude-Squared Coherence Across Dimensions');
% grid on;
% figure;
% % 循环绘制每个时间序列
% % hold on; % 保持图形窗口打开，以便将所有线条绘制在同一张图上
% % for i = 1:size(Y1,1)
%     plot(Y1(2, :),'LineStyle', '-', 'linewidth',1.5); % 绘制第 i 个时间序列
% % end
% % hold off; % 关闭保持，图形窗口恢复正常行为
% % xlim([0 1050]);
% xlabel('step');
% ylabel('y');
% set(gca,'FontName','Times New Roman','FontSize',18);
% axis tight; % 紧凑显示轴，减少空白区域
% box on; % 添加边框
% %%
% x=train_data(:,:);
% correlationMatrix = corr(x); % Step 1:使用corr函数计算相关系数矩阵
% % Step 2: 构建相似性网络 
% threshold = 0.6; % 相似性阈值，用于确定网络中的边
% adjacencyMatrix = correlationMatrix > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
% A=adjacencyMatrix;
% % A=correlationMatrix;
% figure
% imagesc(A);
% hTitle=title('real data');
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 18); % 调整坐标轴字体和字号
% set(hTitle, 'FontSize', 20, 'FontWeight', 'bold'); % 调整标题字体大小和加粗
% colorbar
% 
% x1=Y1(:,:)';
% correlationMatrix1 = corr(x1); % Step 1:使用corr函数计算相关系数矩阵
% % Step 2: 构建相似性网络 
% threshold = 0.6; % 相似性阈值，用于确定网络中的边
% adjacencyMatrix1 = correlationMatrix1 > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
% A1=adjacencyMatrix1;
% % A=correlationMatrix;
% figure
% imagesc(A1);
% hTitle=title('RC');
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 18); % 调整坐标轴字体和字号
% set(hTitle, 'FontSize', 20, 'FontWeight', 'bold'); % 调整标题字体大小和加粗
% colorbar
% frobenius_norm = norm(correlationMatrix - correlationMatrix1, 'fro');
% 
% error_fc=abs(A-A1);
% % fc_error(rc_nn)=mean(error_fc,'all');
% 
% figure
% imagesc(error_fc);
% hTitle=title('error');
% set(gca, 'FontName', 'Times New Roman', 'FontSize', 18); % 调整坐标轴字体和字号
% set(hTitle, 'FontSize', 20, 'FontWeight', 'bold'); % 调整标题字体大小和加粗
% colorbar
%  %  hold on;
% %  plot(t4,Y1(1,229:end),'LineStyle', '-', 'linewidth',2.5,'Color',color_f1)
% %  hold on;
% %  plot(t1,testdata1(1:drive_num+1,1),'-', 'linewidth',2.5,'Color',color_f2);
% %  hold on;
% %  plot(t1,testdata1(1:drive_num+1,1),'.','Color',color_f4,'MarkerSize',20);
% performPCAandVisualize(X1, 'Data Set X1');
% performPCAandVisualize(Y1, 'Data Set Y1');
% function performPCAandVisualize(X, figureTitle)
%     % X是m*n维的数据矩阵，m为维度，n为样本数
%     % 步骤1: 数据中心化
%     X_mean = mean(X, 2); % 计算每个变量的平均值
%     X_centered = X - X_mean; % 中心化数据
% 
%     % 步骤2: 计算协方差矩阵
%     C = (X_centered * X_centered') / (size(X, 2) - 1); % 协方差矩阵
% 
%     % 步骤3: 特征值分解
%     [V, D] = eig(C); % V为特征向量，D为对角线上的特征值
% 
%     % 步骤4: 排序特征值和特征向量
%     [d, ind] = sort(diag(D), 'descend'); % 将特征值降序排序
%     V = V(:, ind); % 将特征向量按相应的特征值排序
% 
%     % 步骤5: 选择主成分
%     k = 3; % 选择前3个主成分
%     V_k = V(:, 1:k); % 选择前k个主成分
% 
%     % 步骤6: 将数据投影到主成分上
%     X_projected = V_k' * X_centered; % 数据投影
% 
%     % 显示结果
%     fprintf('前三个主成分的方差解释率 (%s):\n', figureTitle);
%     variance_explained = sum(d(1:k)) / sum(d) * 100;
%     fprintf('前 %d 个主成分解释了总方差的 %.2f%%\n', k, variance_explained);
% 
%     % 可视化投影结果
%     figure;
%     scatter3(X_projected(1, :), X_projected(2, :), X_projected(3, :), 'filled');
%     title(['PCA Projection - ', figureTitle]);
%     xlabel('Principal Component 1');
%     ylabel('Principal Component 2');
%     zlabel('Principal Component 3');
%     grid on; % 添加网格以便更好地观察数据点的分布
% end
% 
% %  set(hTitle,'FontName','Times New Roman','FontSize', 20)
% %  legend('Model','RC')
% % hXLabel = xlabel('$Time \:(s)$','interpreter','latex');
% % hYLabel = ylabel('$BOLD$','interpreter','latex');
% % xlim([0 229*3]);
% % hTitle=title('predicting stage');
% % legend('Real data','RC')
% % rmse_dynamics=mean(abs(Y1(:,1:end-1)-indata(2:trainLen-intLen,:)'),'all');
% % color_f1= addcolorplus(12);
% % color_f2= addcolorplus(11);
% % color_f3= addcolorplus(1);
% % color_f4= addcolorplus(164);
% % x=train_data(:,:);
% % correlationMatrix = corr(x); % Step 1:使用corr函数计算相关系数矩阵
% % % Step 2: 构建相似性网络 
% % threshold = 0.6; % 相似性阈值，用于确定网络中的边
% % adjacencyMatrix = correlationMatrix > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
% % % A=adjacencyMatrix;
% % A=correlationMatrix;
% % figure
% % imagesc(A);
% % hTitle=title('real data');
% 
% figure
% imagesc(Wout);
% colorbar
% hTitle=title('Wout for AD');
% figure
% heatmap(Wout);
% hold on;
% line([68,68],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([124,124],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([160,160],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([174,174],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([188,188],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([210,210],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([214,214],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([218,218],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([230,230],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([246,246],[0,263],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[68,68],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[124,124],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[160,160],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[174,174],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[188,188],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[210,210],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[214,214],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[218,218],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[230,230],'Color','black','LineWidth',3);
% hold on;
% line([0,263],[246,246],'Color','black','LineWidth',3);
%%

% x1=Y11(:,:)';
% correlationMatrix = corr(x1); % Step 1:使用corr函数计算相关系数矩阵
% % Step 2: 构建相似性网络 
% threshold = 0.6; % 相似性阈值，用于确定网络中的边
% adjacencyMatrix = correlationMatrix > threshold; % 构建邻接矩阵，大于阈值的相关系数对应的边为1，否则为0
% % A1=adjacencyMatrix;
% A1=correlationMatrix;
% figure
% imagesc(A1);
% hTitle=title('RC');
% error_fc=abs(A-A1);
% figure
% imagesc(error_fc);
% hTitle=title('error');
% colorbar
%%
% figure
% plot(indata(2:trainLen-2,100),'LineStyle', '-', 'linewidth',2.5,'Color',color_f3);
% hold on;
% plot(Y1(100,1:end-1),'LineStyle', '--', 'linewidth',2.5,'Color',color_f1);
% W_expanded = [W zeros(500, 3)];  % 在 A 的右侧添加 200 列零
% W_stru=W_expanded+Win*Wout;
% W_stru= W_stru > 0.0001;
% figure
% imagesc(W_stru);

% g1=digraph(W_stru);
% l1=g1.Edges;
% ff1=table2array(l1);
% csvwrite('edge_w_0.0001_train3_AD004_rc_500.csv',ff1);

% save w1_trainNC05.mat W_stru
%%
% W_stru= W_stru > 0.00000001;
% W= W > 0.0000000000001;
% W= W > 0.0000000000001;
% meanW = mean(Wout, 2); % 计算每行的均值，2代表按行操作
% stdW = std(Wout, 0, 2); % 计算每行的标准差，0代表使用N-1（样本标准差），2代表按行操作
% medianW = median(Wout, 2); % 计算每行的中位数，2代表按行操作
% % 计算相关系数
% corrW = corr(Wout'); % 计算权重之间的相关系数，转置是为了获得行与行之间的相关性
% corrW1=corr(corrW);
% corrW2=corr(corrW1);
% % AdjMatrix=corrW > 0.3;
% AdjMatrix=corrW2 > 0.3;

% figure
% imagesc(AdjMatrix);
% hTitle=title('FC of Wout');
% colorbar
% g2=digraph(AdjMatrix);
% l3=g2.Edges;
% ff3=table2array(l3);
% csvwrite('edge_new_wout_corr2_0.3_train3_AD004_rc_500.csv',ff3);
% csvwrite('edge_new_wout_0.3_train3_AD005_rc_500.csv',ff3);
% csvwrite('edge_new_wout_corr3_0.3_train2_MC006_rc_500.csv',ff3);
% csvwrite('edge_new_wout_corr3_0.3_train1_NC006_rc_500.csv',ff3);
% inDegree = sum(W_stru, 1);  % 列和表示入度
% outDegree = sum(W_stru, 2); % 行和表示出度
% % % 
% % % 计算度分布
% inDegreeDist = histcounts(inDegree, 'BinMethod', 'integers');
% outDegreeDist = histcounts(outDegree, 'BinMethod', 'integers');
% degree=outDegree;
% maxDegree = max(degree);
% % 初始化度分布数组
% degreeDistribution = zeros(maxDegree, 1);
% 
% % 计算度的分布
% for k = 1:maxDegree
%     degreeDistribution(k) = sum(degree == k);
% end
% 
% % 转换为概率密度
% probabilityDensity = degreeDistribution / length(degree);
% 
% % 绘制度的概率密度分布
% figure
% % loglog(1:maxDegree, probabilityDensity)
% bar(1:maxDegree, probabilityDensity)
% xlabel('Degree')
% ylabel('Probability Density')
% title('Degree Distribution Probability Density')
% %     绘制度分布图
%     figure;
%     subplot(1,2,1);
%     stem(0:length(inDegreeDist)-1, inDegreeDist, 'filled');
%     title('In-Degree Distribution');
%     xlabel('Degree');
%     ylabel('Frequency');
%     hold on;
%     subplot(1,2,2);
%     stem(0:length(outDegreeDist)-1, outDegreeDist, 'filled');
%     title('Degree Distribution');
%     xlabel('Degree');
%     ylabel('Frequency');
% % figure
% % imagesc(W_stru);
% % colormap hot
% % colorbar
% % W_stru= W_stru > 0.02;
% % g2=digraph(W_stru);
% % l3=g2.Edges;
% % ff3=table2array(l3);
% % csvwrite('edge_w_0.02_trainNC_08_rc_500.csv',ff3);
% csvwrite('edge_w_0.02_trainAD_01_rc_500.csv',ff3);
%%
% corrW= corrW> 0.4;
% 打印统计量结果
% disp('Mean of weights:');
% disp(meanW);
% disp('Standard deviation of weights:');
% disp(stdW);
% disp('Median of weights:');
% disp(medianW);
% figure
% imagesc(corrW);
% colorbar
% hTitle=title('MCI');
