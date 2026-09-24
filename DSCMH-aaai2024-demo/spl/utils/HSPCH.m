function [ImgToTxt,TxtToImg] = HSPCH(trainLabel, param, dataset)
seed = 0;
rng('default');
rng(seed);
X1 = dataset.XDatabase;
X2 = dataset.YDatabase;
XTest = dataset.XTest;
YTest = dataset.YTest;
testL = dataset.testL;
databaseL = dataset.databaseL;

% top_K=1000;
[d1,~] = size(X1');
[d2,~] = size(X2');
bit = param.bit;
maxIter = param.maxIter;
lambda = param.lambda;
beta = param.beta;
alpha = param.alpha;
r = param.r;

numTrain = size(trainLabel, 1);

%--------------------------------initial------------------------------------

B1= ones(numTrain, bit); 
B1(randn(numTrain, bit) < 0) = -1;
B2=B1;
H=B1;

% P1=eye(d1,d1);
% P2=eye(d2,d2);
% R1=randn(d1,bit);
% R2=randn(d2,bit);
P1=randn(d1,r);
P2=randn(d2,r);
R1=randn(r,bit);
R2=randn(r,bit);

XTX1=X1'*X1;
XTX2=X2'*X2;
% YYT=trainLabel*trainLabel';


for epoch = 1:maxIter
tic 
   %--------- B-step
   B1=sgn(X1*P1*R1+alpha*bit*trainLabel*(trainLabel'*H));
   B2=sgn(X2*P2*R2+beta*bit*trainLabel*(trainLabel'*H));
%    B1_TMP=(X1*P1*R1+alpha*bit*trainLabel*(trainLabel'*H))';
%    B1=hash_const(B1_TMP,bit,numTrain)';  
%    B2_TMP=(X2*P2*R2+beta*bit*trainLabel*(trainLabel'*H))'; 
%    B2=hash_const(B2_TMP,bit,numTrain)'; 
   H_C = (alpha*bit*trainLabel*(trainLabel'*B1)+beta*bit*trainLabel*(trainLabel'*B2))';
   H=hash_const(H_C,bit,numTrain)';  
     
   %--------- PR-step  
   PR1 = P1*R1;
   PR2 = P2*R2;
   v1  = sqrt(sum(PR1.*PR1,2)+eps);
   v2  = sqrt(sum(PR2.*PR2,2)+eps);
   M1  = diag(1./(2*v1));
   M2  = diag(1./(2*v2)); 
   St1 = XTX1+lambda*M1;
   St2 = XTX2+lambda*M2;
   Sb1 = X1'*B1*B1'*X1;
   Sb2 = X2'*B2*B2'*X2;
   [P1, ~, ~]=eig1((St1\Sb1), r, 1);
   [P2, ~, ~]=eig1((St2\Sb2), r, 1);
   R1 = (P1'*St1*P1)\(P1'*X1'*B1);
   R2 = (P2'*St2*P2)\(P2'*X2'*B2); 
   
    

end
%  ytrain=dataset.databaseL;
% Xtrain=X1*P1;Ytrain=X2*P2;
% save('train_2.mat','Xtrain','Ytrain','ytrain');
% Xtrain=B1;Ytrain=B2;
% save('train_3.mat','Xtrain','Ytrain','ytrain');
toc
%    %real-time evaluation
    tBX = sign(XTest * P1*R1);
    tBY = sign(YTest * P2*R2);
%     sim_it = B1 * tBY';
%     sim_ti = B2 * tBX';
%     R = size(H,1); 
% 
%     ImgToTxt = mAP(sim_ti,databaseL,testL,R);%
%     TxtToImg= mAP(sim_it,databaseL,testL,R);%
%     fprintf('...iter:%d,   i2t:%.4f,   t2i:%.4f\n',epoch, ImgToTxt, TxtToImg)

B1(B1<0) = 0;
B1 = compactbit(B1); 
B2(B2<0) = 0;
B2 = compactbit(B2); 
tBX(tBX<0) = 0;
tBX = compactbit(tBX);    
tBY(tBY<0) = 0;
tBY = compactbit(tBY); 
Dhamm1 = hammingDist(tBY, B1);
[~, HammingRank]=sort(Dhamm1,2);
out.MAP_test(1) = cal_mAP(databaseL,testL,HammingRank);
[out.Image_VS_Text_precision, out.Image_VS_Text_recall] = precision_recall(HammingRank',databaseL,testL);
% out.Image_To_Text_Precision = precision_at_k(HammingRank', databaseL,testL,top_K);


Dhamm2 = hammingDist(tBX, B2);
[~, HammingRank]=sort(Dhamm2,2);
out.MAP_test(2) = cal_mAP(databaseL,testL,HammingRank);
[out.Text_VS_Image_precision, out.Text_VS_Image_recall] = precision_recall(HammingRank', databaseL,testL);
% out.Text_To_Image_Precision = precision_at_k(HammingRank', databaseL,testL,top_K);

ImgToTxt=out.MAP_test(1);
TxtToImg=out.MAP_test(2);

Image_VS_Text_precision=out.Image_VS_Text_precision;
Image_VS_Text_recall=out.Image_VS_Text_recall;
% Image_To_Text_Precision=out.Image_To_Text_Precision;
Text_VS_Image_precision=out.Text_VS_Image_precision;
Text_VS_Image_recall=out.Text_VS_Image_recall;
% Text_To_Image_Precision=out.Text_To_Image_Precision;

save('HCCH_result_flickr_cur.mat','Image_VS_Text_precision','Image_VS_Text_recall','Text_VS_Image_precision','Text_VS_Image_recall');
    
    
    

end




