function [ImgToTxt,TxtToImg] = DSPL(trainLabel, param, dataset)
seed = 0;
rng('default');
rng(seed);
%norm2(Xt-Ut*V) 
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
[d1,~] = size(X1');
[d2,~] = size(X2');
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
beta = param.beta;
alpha = param.alpha;


numTrain = size(trainLabel, 1);
gamma1_SPLs=param.pace1;
gamma2_SPLs=param.pace1;
gamma1_SPLb=param.pace2;
gamma2_SPLb=param.pace2;

%--------------------------------initial------------------------------------

V= ones(bit, numTrain);
V(randn(bit, numTrain) < 0) = -1;




LOSS1 = zeros(size(V));
LOSS2 = zeros(size(V));
L1_SPLs = zeros(size(LOSS1,2),1);
R1_SPLs = ones(size(LOSS1,2),1);
L2_SPLs = zeros(size(LOSS2,2),1);
R2_SPLs = ones(size(LOSS2,2),1);
L1_SPLb = zeros(size(LOSS1,1),1);
R1_SPLb = ones(size(LOSS1,1),1);
L2_SPLb = zeros(size(LOSS2,1),1);
R2_SPLb = ones(size(LOSS2,1),1);

tic
for epoch = 1:maxIter
%     B_old=B;
    % Ut-Step
    ZU0 = (alpha*V*databaseL);
    U0 = myOrth(ZU0);    
    
    ZU1 = (beta*V*X1);
    U1 = myOrth(ZU1);
    
    ZU2 = (lambda*V*X2);
    U2 = myOrth(ZU2);
    
    % Vt-Step
    V=sign(alpha*U0*databaseL'+beta*U1*X1'+lambda*U2*X2');
  
    
end
% lambda=1e-1;
    P1 = (X1' * X1 + lambda * eye(d1, d1)) \ (X1' * V');P1=P1';
    P2 = (X2' * X2 + lambda * eye(d2, d2)) \ (X2' * V');P2=P2';
% for epoch = 1:maxIter
%       % P-Step
%       SPLsXT1=bsxfun(@times,R1_SPLs,X1);       
%       P1 = V*SPLsXT1*pinv(X1'*SPLsXT1);
%       SPLsXT2=bsxfun(@times,R2_SPLs,X2);       
%       P2 = V*SPLsXT2*pinv(X2'*SPLsXT2);
% 
%     % rs^t-Step
%     LOSS1_s = zeros(size(V));
%     LOSS1_s = LOSS1_s+bsxfun(@times,R1_SPLb,V-P1*X1');
%     for ind = 1:size(LOSS1_s,2)
%         L1_SPLs(ind) = norm(LOSS1_s(:,ind),'fro');
%     end
%     L1_SPLs = L1_SPLs';
%     L1_SPLs = mapminmax(L1_SPLs, 0, 1)*cmm;
%     L1_SPLs= L1_SPLs';
%     loss_each_Image(epoch) = sum(L1_SPLs);
%       
%     for ind = 1:size(L1_SPLs,1)
%         me = (1+exp(-1*gamma1_SPLs));
%         de = (1+exp(L1_SPLs(ind)-gamma1_SPLs));
%         R1_SPLs(ind) = me/de;
%     end
% %     gamma1_SPLs=gamma1_SPLs*pace1;
%     
%     LOSS2_s = zeros(size(V));
%     LOSS2_s = LOSS2_s+bsxfun(@times,R2_SPLb,V-P2*X2');
% %     LOSS2_s=X2'-U2*V2;
%     for ind = 1:size(LOSS2_s,2)
%         L2_SPLs(ind) = norm(LOSS2_s(:,ind),'fro');
%     end
%     L2_SPLs = L2_SPLs';
%     L2_SPLs = mapminmax(L2_SPLs, 0, 1)*cmm;
%     L2_SPLs= L2_SPLs';
%    loss_each_text(epoch) = sum(L2_SPLs) ;
%        
%     for ind = 1:size(L2_SPLs,1)
%         me = (1+exp(-1*gamma2_SPLs));
%         de = (1+exp(L2_SPLs(ind)-gamma2_SPLs));
%         R2_SPLs(ind) = me/de;
%     end
% %     gamma2_SPLs=gamma2_SPLs*pace1;
%     
%     % rb^t-Step
%     LOSS1_b = zeros(size(V));
%     LOSS1_b = LOSS1_b+bsxfun(@times,V-P1*X1',R1_SPLs');
%     for ind = 1:size(LOSS1_b,1)
%         L1_SPLb(ind) = norm(LOSS1_b(:,ind),'fro');
%     end
%     L1_SPLb = L1_SPLb';
%     L1_SPLb = mapminmax(L1_SPLb, 0, 1)*cmm;
%     L1_SPLb= L1_SPLb';
%     
% %     loss_each_text(epoch) = sum(L1_SPLb)
%     for ind = 1:size(L1_SPLb,1)
%         me = (1+exp(-1*gamma1_SPLb));
%         de = (1+exp(L1_SPLb(ind)-gamma1_SPLb));
%         R1_SPLb(ind) = me/de;
%     end
% %     gamma1_SPLb=gamma1_SPLb*pace2;
%      
%     LOSS2_b = zeros(size(V));
%     LOSS2_b = LOSS2_b+bsxfun(@times,V-P2*X2',R2_SPLs');
%     for ind = 1:size(LOSS2_b,1)
%         L2_SPLb(ind) = norm(LOSS2_b(:,ind),'fro');
%     end
%     L2_SPLb = L2_SPLb';
%     L2_SPLb = mapminmax(L2_SPLb, 0, 1)*cmm;
%     L2_SPLb= L2_SPLb';
%     for ind = 1:size(L2_SPLb,1)
%         me = (1+exp(-1*gamma2_SPLb));
%         de = (1+exp(L2_SPLb(ind)-gamma2_SPLb));
%         R2_SPLb(ind) = me/de;
%     end
% %     gamma2_SPLb=gamma2_SPLb*pace2;
%     
% %     fun(epoch)=norm(V-B_old,'fro')/norm(B_old,'fro');
% end

toc




%    %real-time evaluation
tBX = sign(P1*XTest');
tBY = sign(P2*YTest');
V(V<0) = 0;
V = compactbit(V');
tBX(tBX<0) = 0;
tBX = compactbit(tBX');
tBY(tBY<0) = 0;
tBY = compactbit(tBY');
Dhamm1 = hammingDist(tBY, V);
[~, HammingRank]=sort(Dhamm1,2);
out.MAP_test(1) = cal_mAP(databaseL,testL,HammingRank);
[out.Image_VS_Text_precision, out.Image_VS_Text_recall] = precision_recall(HammingRank',databaseL,testL);
% out.Image_To_Text_Precision = precision_at_k(HammingRank', databaseL,testL,top_K);
Dhamm2 = hammingDist(tBX, V);
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




