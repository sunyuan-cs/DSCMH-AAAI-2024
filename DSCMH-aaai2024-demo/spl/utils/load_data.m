function dataset = load_data(dataname)
switch dataname
     case 'wikiData'
        load data/wikiData.mat I_te I_tr T_te T_tr L_te L_tr;
        dataset.XTest = I_te;
        dataset.YTest = T_te;
        dataset.XDatabase = I_tr;
        dataset.YDatabase = T_tr;
        dataset.testL = L_te;
        dataset.databaseL = L_tr;
    case 'nus-vgg'
        load data/mynus_cnn.mat I_te I_tr T_te T_tr L_te L_tr;
        dataset.XTest = I_te;
        dataset.YTest = T_te;
        dataset.XDatabase = I_tr;
        dataset.YDatabase = T_tr;
        dataset.testL = L_te;
        dataset.databaseL = L_tr;
    case 'flickr-25k'
        load data/flickr-25k.mat XTest YTest XDatabase YDatabase testL databaseL;
       
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'flickr_index'
        load data/flickr_index.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = double(XTest);
        dataset.YTest = YTest;
        dataset.XDatabase = double(XDatabase);
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
       case 'flickr_IT_half_0.5_5'
        load data/flickr_IT_half_0.5_5.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'flickr_occlusion_0.6'
        load data/flickr_occlusion_0.6.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
     case 'flickr_noise_0.4_base'
        load data/flickr_noise_0.4_base.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = double(XTest);
        dataset.YTest = YTest;
        dataset.XDatabase = double(XDatabase);
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'flickr_noise_0.4'
        load data/flickr_noise_0.4.mat XTest YTest XDatabase YDatabase testL databaseL;
        dataset.XTest = double(XTest);
        dataset.YTest = YTest;
        dataset.XDatabase = double(XDatabase);
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'flickr-25k-vgg'
        load data/flickr-25k.mat VTest YTest VDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),10000);
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
    case 'iapr-tc12'
        load data/iapr-tc12.mat XTest YTest XDatabase YDatabase testL databaseL;
        
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'iapr-tc12-vgg'
        load data/iapr-tc12.mat VTest YTest VDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),10000);
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'nus-wide-clear'%nus-wide-tc10
        load data/nus-wide-clear.mat XTest YTest XDatabase YDatabase testL databaseL;
%         inx = randperm(size(databaseL,1),3000);
%         dataset.XTest = XTest;
%         dataset.YTest = YTest;
%         dataset.XDatabase = XDatabase(inx,:);
%         dataset.YDatabase = YDatabase(inx,:);
%         dataset.testL = testL;
%         dataset.databaseL = databaseL(inx,:);
   
        dataset.XTest = XTest;
        dataset.YTest = YTest;
        dataset.XDatabase = XDatabase;
        dataset.YDatabase = YDatabase;
        dataset.testL = testL;
        dataset.databaseL = databaseL;
    case 'nus-wide-tc10-vgg'
        load data/nus-wide.mat VTest YTest VDatabase YDatabase testL databaseL;
        inx = randperm(size(databaseL,1),15000);
        dataset.XTest = VTest;
        dataset.YTest = YTest;
        dataset.XDatabase = VDatabase(inx,:);
        dataset.YDatabase = YDatabase(inx,:);
        dataset.testL = testL;
        dataset.databaseL = databaseL(inx,:);
end
end

