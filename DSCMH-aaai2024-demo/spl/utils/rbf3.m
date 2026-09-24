% [n, ~] = size(dataset.YDatabase);
% anchor_image = dataset.XDatabase(randsample(n, n_anchors1),:); 
% anchor_text = dataset.YDatabase(randsample(n, n_anchors2),:);
% dataset.XDatabase = RBF_fast(dataset.XDatabase',anchor_image'); 
% dataset.XTest = RBF_fast(dataset.XTest',anchor_image'); 
% dataset.YDatabase = RBF_fast(dataset.YDatabase',anchor_text');  
% dataset.YTest = RBF_fast(dataset.YTest',anchor_text'); 



anchor_idx = randsample(size(dataset.XDatabase,1), n_anchors1);
anchor_image = dataset.XDatabase(anchor_idx,:);
anchor_idx = randsample(size(dataset.YDatabase,1), n_anchors2);
anchor_text = dataset.YDatabase(anchor_idx,:);

[dataset.XDatabase,dataset.XTest]=Kernel_Feature(dataset.XDatabase,dataset.XTest,anchor_image);
[dataset.YDatabase,dataset.YTest]=Kernel_Feature(dataset.YDatabase,dataset.YTest,anchor_text);