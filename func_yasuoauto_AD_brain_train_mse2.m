function rmse =func_yasuoauto_AD_brain_train_mse2(hyperpara_set,rng_num,train_data,testdata1,resSize,Phi,Psi,R_N)
% return validating RMSE 
%parameter aware RC
rng(rng_num)
%rng(rng_num,'twister')
eig_rho = hyperpara_set(1);
W_in_a = hyperpara_set(2);
a = hyperpara_set(3);
reg = hyperpara_set(4);
density = hyperpara_set(5);
% tra_Len=340;
% testlen=tra_Len;
data_num=1;
% outSize = size(train_data,2);
outSize = size(testdata1,2);
inSize =  size(train_data,2); 
% perLen=10;
trainLen=size(train_data,1)-1;
test_Len=size(testdata1,1);
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
outdata=testdata1(:,1:outSize);
% resSize =500; % size of the reservoir nodes;  
Win=(rand(resSize, inSize)-0.5)*W_in_a;
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

% for i=0:data_num-1
%     X(:,i*perLen+1-i*intLen:i*perLen-i*intLen+intLen)=[];
%     Yt(:,i*perLen+1-i*intLen:i*perLen-i*intLen+intLen)=[];   
% end
%x1=X;
% randindex=randperm(size(X,2));
% X=X(:,randindex);
% Yt=Yt(:,randindex);
% train the output
X_T = X';
% Wout = Yt*X_T * inv(X*X_T + reg*eye(1+inSize+resSize));
Wout = Yt*X_T / (X*X_T + reg*eye(inSize+R_N));
% Wout = Yt*X_T / (X*X_T + reg*eye(resSize));
% Y1 = Wout*X;
% rmse_dynamics=mean(abs(Y1(:,1:end-1)-indata(2:trainLen-intLen,:)'),'all');
drive_num=1;
testlen=test_Len;
% test_Len1=1000;
Y1= zeros(outSize,test_Len);
% Y1= zeros(inSize,test_Len);
% mse_dynamics=zeros(data_num,1);
x = zeros(resSize,1);
for j=1:data_num
    u1=testdata1((j-1)*testlen+1,:)';
%   x=zeros(resSize,1);
for t = 1:test_Len+1000
    x = (1-a)*x + a*tanh( Win*u1 + W*x);
    s = Psi*x;
    CS_x = Phi*s;
    y = Wout*[u1;CS_x];
%   y(3)=testdata1((j-1)*testlen+1,3);
     Y1(:,t) = y;
    if t<=drive_num
    u1=testdata1(1+t,:)';
    else
    u1=y;
%     u1 = y(2:2:end);
    end
end
%mse_dynamics(j)=mean(abs(Y1(1:2,100+1:end)-testdata1((j-1)*testlen+100+2:(j-1)*testlen+1+test_Len,1:2)'),'all');
mse_dynamics(j)=mean(abs(Y1(:,1:testlen-1)-testdata1((j-1)*testlen+2:(j-1)*testlen+test_Len,:)'),'all');
% +mean(abs(Y1(2,300+1:end)-testdata1((j-1)*testlen+300+2:(j-1)*testlen+1+test_Len,2)'));
end
% rmse_dynamics=mean(mse_dynamics);
% if(mean(Y1(:,end-10:end)-Y1(:,end-20:end-10))<0.1)
%    error=10;
% else
%    error=0.1;
% end
correlationMatrix_real=corr(train_data);
correlationMatrix_rc=corr(Y1');
% frobenius_norm = norm(correlationMatrix_real - correlationMatrix_rc, 'fro');
similarity_metric = corr2(correlationMatrix_rc, correlationMatrix_real);
rmse=-similarity_metric;
if isnan(rmse)
     rmse=0;
end
if rmse==0
    rmse=1;
end
if rmse>20
    rmse=10;
end
% end