close all;
clear;
warning off
addpath(genpath('./utils/'));
addpath(genpath('./data/'));
seed = 2023;
rng('default');
rng(seed);

db = {'wikiData','flickr_IT_half_0.5_5','flickr-25k','flickr_index','iapr-tc12','nus-wide-clear'};
hashmethods = {'DSCMH'};
bits = [8 16 32 64];%8 16 32 64
nb = numel(bits);
% param.top_K = 1000;
param.bits = bits;
param.maxIter = 5;
fprintf('......%s start...... \n\n', 'DSCMH');

for dbi =4:4
    % load dataset
    db_name = db{dbi}; param.db_name = db_name;
    dataset = load_data(db_name);
    n_anchors1 = 1500;
    n_anchors2 = 1500;
    alpha =1e-3;%[1e-4,1e-3,1e-2,1e-1,1]; %,1e1,1e2,1e3,1e4
    lambda =1e-2;%[1e-4,1e-3,1e-2,1e-1,1];%1e-4;%1,1e1,1e2,1e3,1e4
   
    par_1 = numel(alpha);
    par_3 = numel(lambda);

    %     rbf2
    [n, ~] = size(dataset.YDatabase);
    anchor_image = dataset.XDatabase(randsample(n, n_anchors1),:);
    anchor_text = dataset.YDatabase(randsample(n, n_anchors2),:);
    dataset.XDatabase = RBF_fast(dataset.XDatabase',anchor_image');
    dataset.XTest = RBF_fast(dataset.XTest',anchor_image');
    dataset.YDatabase = RBF_fast(dataset.YDatabase',anchor_text');
    dataset.YTest = RBF_fast(dataset.YTest',anchor_text');   
    %     Xtrain=dataset.XDatabase;
    %     save('Image_train.mat','Xtrain');    
    total_res=[];
    % run algorithm
    for i = 1: nb
        for ij=1:par_1
            for jjj=1:par_3
                param.bit = bits(i);
                param.alpha = alpha(ij);
                param.lambda= lambda(jjj)
                trainL = dataset.databaseL;
                [ImgToTxt,TxtToIm] = DSCMH_3(trainL, param, dataset)% param.beta,
                total_res = [total_res;ImgToTxt,TxtToIm, param.bit, param.alpha,param.lambda];               
            end
        end
    end
    total_name = [db_name '_result' '.mat'];
%     save(total_name,'total_res');
%     clear total_res
end



