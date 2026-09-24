function [ImgToTxt,TxtToImg] = DSCMH_ab(trainLabel, param, dataset)
seed = 2023;
% rng('default');
rng(seed);

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
beta = param.alpha;
alpha = param.alpha;

numTrain = size(trainLabel, 1);

%--------------------------------initial------------------------------------

B= ones(bit, numTrain);
B(randn(bit, numTrain) < 0) = -1;
V1 = randn(bit, numTrain);
V2 = randn(bit, numTrain);

tic
for epoch = 1:maxIter
    % Ut-Step    
      U1=X1'/V1;
      U2=X2'/V2;     
    % Vt-Step
    ZV1 = (X1*U1) + bit * databaseL * (databaseL' *  alpha*B');
    V1 = myOrth(ZV1)';
    ZV2 = (X2*U2) + bit * databaseL * (databaseL' *  beta*B');
    V2 = myOrth(ZV2)'; 
    % B-Step
    B = sign(bit*(alpha*V1+beta*V2)*databaseL*databaseL');      

end
toc
    P1 = (X1' * X1 + lambda * eye(d1, d1)) \ (X1' * B');
    P1=P1';
    P2 = (X2' * X2 + lambda * eye(d2, d2)) \ (X2' * B');
    P2=P2';

%    %real-time evaluation
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