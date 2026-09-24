function [ImgToTxt,TxtToImg] = DSCMH_random_opp(trainLabel, param, dataset)
seed = 2023;
% rng('default');
rng(seed);
cmm=15;
X1 = dataset.XDatabase;
X2 = dataset.YDatabase;
XTest = dataset.XTest;
YTest = dataset.YTest;
testL = dataset.testL;
truth_L = dataset.databaseL;
databaseL= NormalizeFea(truth_L,1);
top_K=1000;
[d1,~] = size(X1');
[d2,~] = size(X2');
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
alpha = param.alpha;
pace=1.2;

numTrain = size(trainLabel, 1);

gamma1_SPLs=8;
gamma1_SPLb=8;
gamma2_SPLb=8;
%--------------------------------initial------------------------------------
B= ones(bit, numTrain);
B(randn(bit, numTrain) < 0) = -1;
V1 = randn(bit, numTrain);
LOSS1 = zeros(size(X1'));
LOSS2 = zeros(size(X2'));
R1_SPLs = ones(size(LOSS1,2),1);
R1_SPLb = ones(size(LOSS1,1),1);
R2_SPLb = ones(size(LOSS2,1),1);
SPL1_s=[];
SPL1_b=[];
SPL2_b=[];
tic
for epoch = 1:maxIter
    % Ut-Step
      SPLsXT1=bsxfun(@times,R1_SPLs,V1');   
      U1=(X1'*SPLsXT1)/(V1*SPLsXT1);
      U2=(X2'*SPLsXT1)/(V1*SPLsXT1);
      
    % Vt-Step
    SPLsXB1=bsxfun(@times,R1_SPLs,X1); 
    SPLbXU1=bsxfun(@times,R1_SPLb,U1); 
    SPLbXU2=bsxfun(@times,R2_SPLb,U2); 
    ZV1 = (SPLsXB1*SPLbXU1+SPLsXB1*SPLbXU2) + bit * databaseL * (databaseL' *  alpha*B');
    V1 = myOrth(ZV1)';
 
    % B-Step
    B = sign(bit*(alpha*V1)*databaseL*databaseL');   
   
    % rs^t-Step
    LOSS1_s = bsxfun(@times,R1_SPLb,X1'-U1*V1);
    for ind = 1:size(LOSS1_s,2)
        L1_SPLs(ind) = norm(LOSS1_s(:,ind),'fro');
    end
    L_SPLs = mapminmax(L1_SPLs, 0, 1)*cmm; 
    for ind = 1:size(L_SPLs,2)
        me = (1+exp(-1*gamma1_SPLs));
        de = (1+exp(L_SPLs(ind)-gamma1_SPLs));
        R_SPLs(ind) = me/de;
    end
    gamma1_SPLs=pace*gamma1_SPLs;
    SPL1_s=1-[SPL1_s,R_SPLs'];
    
%     num(epoch)=length(find(L1_SPLs<gamma1_SPLs));
%     sample(epoch)=num(epoch)/length(L1_SPLs)*100;
    

    
    % rb^t-Step
%     LOSS1_b =bsxfun(@times,X1'-U1*V1,R1_SPLs');
%     for ind = 1:size(LOSS1_b,1)
%         L1_SPLb(ind) = norm(LOSS1_b(:,ind),'fro');
%     end
%     L1_SPLb = mapminmax(L1_SPLb, 0, 1)*cmm;
%     for ind = 1:size(L1_SPLb,2)
%         me = (1+exp(-1*gamma1_SPLb));
%         de = (1+exp(L1_SPLb(ind)-gamma1_SPLb));
%         R1_SPLb(ind) = me/de;
%     end
%         gamma1_SPLb=pace*gamma1_SPLb;
%         SPL1_b=[SPL1_b,R1_SPLb];
% 
%     LOSS2_b = bsxfun(@times,X2'-U2*V1,R1_SPLs');
%     for ind = 1:size(LOSS2_b,1)
%         L2_SPLb(ind) = norm(LOSS2_b(:,ind),'fro');
%     end
%     L2_SPLb = mapminmax(L2_SPLb, 0, 1)*cmm;    
%     for ind = 1:size(L2_SPLb,2)
%         me = (1+exp(-1*gamma2_SPLb));
%         de = (1+exp(L2_SPLb(ind)-gamma2_SPLb));
%         R2_SPLb(ind) = me/de;
%     end
%        gamma2_SPLb=pace*gamma2_SPLb;
%        SPL2_b=[SPL2_b,R2_SPLb];
end
% save('DSCMH_flickr_weight.mat','SPL1_s','SPL1_b','SPL2_b'); 

%    save('DSCMH_flickr_8_rate.mat','sample'); 
    P1 = (X1' * X1 + lambda * eye(d1, d1)) \ (X1' * B');P1=P1';
    P2 = (X2' * X2 + lambda * eye(d2, d2)) \ (X2' * B');P2=P2';

toc

%real-time evaluation
tBX = sign(P1*XTest');
tBY = sign(P2*YTest');
B(B<0) = 0;
B = compactbit(B');
tBX(tBX<0) = 0;
tBX = compactbit(tBX');
tBY(tBY<0) = 0;
tBY = compactbit(tBY');
Dhamm1 = hammingDist(tBY, B);
[~, HammingRank]=sort(Dhamm1,2);
out.MAP_test(1) = cal_mAP(truth_L,testL,HammingRank);
[out.Image_VS_Text_precision, out.Image_VS_Text_recall] = precision_recall(HammingRank',truth_L,testL);
 out.Image_To_Text_Precision = precision_at_k(HammingRank', truth_L,testL,top_K);
Dhamm2 = hammingDist(tBX, B);
[~, HammingRank]=sort(Dhamm2,2);
out.MAP_test(2) = cal_mAP(truth_L,testL,HammingRank);
[out.Text_VS_Image_precision, out.Text_VS_Image_recall] = precision_recall(HammingRank', truth_L,testL);
out.Text_To_Image_Precision = precision_at_k(HammingRank', truth_L,testL,top_K);
ImgToTxt=out.MAP_test(1);
TxtToImg=out.MAP_test(2);

% Image_VS_Text_precision=out.Image_VS_Text_precision;
% Image_VS_Text_recall=out.Image_VS_Text_recall;
% Image_To_Text_Precision=out.Image_To_Text_Precision;
% Text_VS_Image_precision=out.Text_VS_Image_precision;
% Text_VS_Image_recall=out.Text_VS_Image_recall;
% Text_To_Image_Precision=out.Text_To_Image_Precision;
% save('DSCMH_result_nuswide_cur.mat','Image_VS_Text_precision','Image_VS_Text_recall','Text_VS_Image_precision','Text_VS_Image_recall','Image_To_Text_Precision','Text_To_Image_Precision');


end