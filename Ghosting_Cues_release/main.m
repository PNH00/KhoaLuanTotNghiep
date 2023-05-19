I=im2double(imread("test_6.png"));
tic;
deghost(I);
t = toc;
disp(t);