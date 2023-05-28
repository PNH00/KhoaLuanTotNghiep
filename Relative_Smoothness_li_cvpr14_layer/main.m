clc;
clear;

%% Scan Images

logpath = fopen('results_both.txt','w');
fprintf(logpath,'LOG\n');
disp('DEMO');

path_rmrf = [pwd,'/actual_data/'];

list_rmrf = dir([path_rmrf,'*.jpg']);

num_rmrf = length(list_rmrf);


%% Reflection Removal 
for i=1:num_rmrf
    
    fprintf(logpath,'\nProcessing reflection_removal-%d...\n',i); 
    path = [path_rmrf,list_rmrf(i).name]; 
    I = im2double(imread(path));

    %%Mặc định
    %[H W D] = size(I);
    %[LB LR time] = septRelSmo(I, 50, zeros(H,W,D), I);

    %%Áp dụng phương pháp cải thiện thứ nhất
    %x = size(I);
    %I_resize = imresize(I,[x(1)/2 x(2)/2]); 
    %[H W D] = size(I_resize);
    %[LB LR time] = septRelSmo(I_resize, 50, zeros(H,W,D), I_resize);
    %tic
    %LR = imresize(LR,[x(1) x(2)]); 
    %LB = I - LR;
    %tt = toc;

    %Áp dụng hai phương pháp cải thiện tốc độ
    x = size(I);
    I_resize = imresize(I,[x(1)/2 x(2)/2]); 
    [H W D] = size(I_resize);
    [LB LR time] = septRelSmo(I_resize, 100, zeros(H,W,D), I_resize);
    tic
    LR = imresize(LR,[x(1) x(2)]); 
    LB = I - LR;
    tt = toc;

    fprintf(logpath,'Time consumption: %.4fs\n',time + tt);
 
    STR = ['Ref_Rem_ex',int2str(i)];
    disp(['DONE!......',STR]);
    
    
    
    imwrite(LB,"results_both_optimization/LB_"+num2str(i)+".png");
    imwrite(LR,"results_both_optimization/LR_"+num2str(i)+".png");

end

fclose(logpath);
disp('ALL DONE!');
disp('Pls open results_images & results_log to check the results!')