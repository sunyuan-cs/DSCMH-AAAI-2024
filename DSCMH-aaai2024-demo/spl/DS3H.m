function [ImgToTxt,TxtToImg] = DS3H(trainLabel, param, dataset)
seed = 0;
rng('default');
rng(seed);
%two-step
cmm=15;
X1 = dataset.XDatabase;
X2 = dataset.YDatabase;
XTest = dataset.XTest;
YTest = dataset.YTest;
testL = dataset.testL;
databaseL = dataset.databaseL;
databaseL= NormalizeFea(databaseL,1);
% top_K=1000;
[d,~] = size(X1');
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
beta = param.beta;
alpha = param.alpha;
% pace_rate1=param.pace1;
% pace_rate2=param.pace2;
numTrain = size(trainLabel, 1);
gamma1_SPLs=param.pace1;
gamma2_SPLs=param.pace1;
gamma_SPLb=param.pace2;


%--------------------------------initial------------------------------------

B= ones(bit, numTrain);
B(randn(bit, numTrain) < 0) = -1;


V1 = randn(bit, numTrain);
V2 = randn(bit, numTrain);
W1= randn(bit, d);
W2= randn(bit, d);
LOSS1 = zeros(size(X1'));
LOSS2 = zeros(size(X2'));
L1_SPLs = zeros(size(LOSS1,2),1);
R1_SPLs = ones(size(LOSS1,2),1);
L2_SPLs = zeros(size(LOSS2,2),1);
R2_SPLs = ones(size(LOSS2,2),1);
L_SPLb = zeros(size(LOSS1,1),1);
R_SPLb = ones(size(LOSS1,1),1);


tic
for epoch = 1:maxIter
    % Ut-Step
      SPLsXT1=bsxfun(@times,R1_SPLs,V1'); 
      ZU1 = (SPLsXT1'*X1);
      U1 = myOrth(ZU1)';
      SPLsXT2=bsxfun(@times,R2_SPLs,V2'); 
      ZU2 = (SPLsXT2'*X2);
      U2 = myOrth(ZU2)';
    % Vt-Step
    SPLsXB1=bsxfun(@times,R1_SPLs,X1); 
%     SPLbXU1=bsxfun(@times,R_SPLb,U1); 
    ZV1 = (alpha*SPLsXB1*U1) + bit * databaseL * (databaseL' *  beta*B');
    V1 = myOrth(ZV1)';
    SPLsXB2=bsxfun(@times,R2_SPLs,X2); 
%     SPLbXU2=bsxfun(@times,R2_SPLb,U2); 
    ZV2 = (alpha*SPLsXB2*U2) + bit * databaseL * (databaseL' *  beta*B');
    V2 = myOrth(ZV2)';
  
    % B-Step
    EB1=bsxfun(@times,R_SPLb,W1'); 
    EB2=bsxfun(@times,R_SPLb,W2');
    B = sign(EB1'*X1'+EB2'*X2'+bit*(beta*V1+beta*V2)*databaseL*databaseL');   
    % W-Step 
    W1 = (X1' * X1 + lambda * eye(d, d)) \ (X1' * B');W1=W1';
    W2 = (X2' * X2 + lambda * eye(d, d)) \ (X2' * B');W2=W2';
    
    
      % rs^t-Step
    LOSS1_s = zeros(size(X1'));
    LOSS1_s = LOSS1_s+X1'-U1*V1;
    for ind = 1:size(LOSS1_s,2)
        L1_SPLs(ind) = norm(LOSS1_s(:,ind),'fro');
    end
    L1_SPLs = L1_SPLs';
    L1_SPLs = mapminmax(L1_SPLs, 0, 1)*cmm;
    L1_SPLs= L1_SPLs';
%     loss_each_Image(epoch) = sum(L1_SPLs);
      
    for ind = 1:size(L1_SPLs,1)
        me = (1+exp(-1*gamma1_SPLs));
        de = (1+exp(L1_SPLs(ind)-gamma1_SPLs));
        R1_SPLs(ind) = me/de;
    end
%      gamma1_SPLs=gamma1_SPLs*pace_rate1;
    
    LOSS2_s = zeros(size(X2'));
    LOSS2_s = LOSS2_s+X2'-U2*V2;
%     LOSS2_s=X2'-U2*V2;
    for ind = 1:size(LOSS2_s,2)
        L2_SPLs(ind) = norm(LOSS2_s(:,ind),'fro');
    end
    L2_SPLs = L2_SPLs';
    L2_SPLs = mapminmax(L2_SPLs, 0, 1)*cmm;
    L2_SPLs= L2_SPLs';
%    loss_each_text(epoch) = sum(L2_SPLs) ;
       
    for ind = 1:size(L2_SPLs,1)
        me = (1+exp(-1*gamma2_SPLs));
        de = (1+exp(L2_SPLs(ind)-gamma2_SPLs));
        R2_SPLs(ind) = me/de;
    end
%      gamma2_SPLs=gamma2_SPLs*pace_rate1;
    
    % rb^t-Step
    LOSS_b = zeros(size(X1'));
    LOSS_b = LOSS_b+X1'-U1*V1+X2'-U2*V2;
    for ind = 1:size(LOSS_b,1)
        L_SPLb(ind) = norm(LOSS_b(:,ind),'fro');
    end
    L_SPLb = L_SPLb';
    L_SPLb = mapminmax(L_SPLb, 0, 1)*cmm;
    L_SPLb= L_SPLb';
    
%     loss_each_text(epoch) = sum(L1_SPLb)
    for ind = 1:size(L_SPLb,1)
        me = (1+exp(-1*gamma_SPLb));
        de = (1+exp(L_SPLb(ind)-gamma_SPLb));
        R_SPLb(ind) = me/de;
    end
%      gamma1_SPLb=gamma1_SPLb*pace_rate2;
     

end



% for epoch = 1:maxIter
%       % P-Step
%    
% %     fun(epoch)=norm(B-B_old,'fro')/norm(B_old,'fro');
% end

toc




%    %real-time evaluation
tBX = sign(W1*XTest');
tBY = sign(W2*YTest');
B(B<0) = 0;
B = compactbit(B');
tBX(tBX<0) = 0;
tBX = compactbit(tBX');
tBY(tBY<0) = 0;
tBY = compactbit(tBY');
Dhamm1 = hammingDist(tBY, B);
[~, HammingRank]=sort(Dhamm1,2);
out.MAP_test(1) = cal_mAP(databaseL,testL,HammingRank);
[out.Image_VS_Text_precision, out.Image_VS_Text_recall] = precision_recall(HammingRank',databaseL,testL);
% out.Image_To_Text_Precision = precision_at_k(HammingRank', databaseL,testL,top_K);
Dhamm2 = hammingDist(tBX, B);
[~, HammingRank]=sort(Dhamm2,2);
out.MAP_test(2) = cal_mAP(databaseL,testL,HammingRank);
[out.Text_VS_Image_precision, out.Text_VS_Image_recall] = precision_recall(HammingRank', databaseL,testL);
% out.Text_To_Image_Precision = precision_at_k(HammingRank', databaseL,testL,top_K);
ImgToTxt=out.MAP_test(1);
TxtToImg=out.MAP_test(2);

% Image_VS_Text_precision=out.Image_VS_Text_precision;
% Image_VS_Text_recall=out.Image_VS_Text_recall;
% Image_To_Text_Precision=out.Image_To_Text_Precision;
% Text_VS_Image_precision=out.Text_VS_Image_precision;
% Text_VS_Image_recall=out.Text_VS_Image_recall;
% Text_To_Image_Precision=out.Text_To_Image_Precision;

% save('HCCH_result_flickr_cur.mat','Image_VS_Text_precision','Image_VS_Text_recall','Text_VS_Image_precision','Text_VS_Image_recall');




end