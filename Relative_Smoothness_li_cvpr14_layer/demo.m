close all;

%% Reflection Separation Using Focus 
disp('Reflection Removal Example');
I2 = im2double(imread('reflection_in.jpg')); 
[H W D] = size(I2);
[LB LR] = septRelSmo(I2, 50, zeros(H,W,D), I2);
figure(2),
subplot 131, imshow(I2) , title('input');
subplot 132, imshow(LB*2), title('background'); 
subplot 133, imshow(LR*2), title('reflection');
