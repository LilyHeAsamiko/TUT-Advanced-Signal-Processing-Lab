I1 = imread('natural.jpg');
I2 = imread('outoffocus.jpg');
figure,
imshow(I1);
figure,
imshow(I2);
%Demosaic: 4 types of mosaic (Bayer pattter):
%'grbg' = G R G
%         B G B
%         G R G
%
%'gbrg' = G B G
%         R G R
%         G B G
%
%'rggb' = R G R
%         G B G
%         R G R
%
%'bggr' = B G B
%         G R G
%         B G B
%A bayer filter mosaic is a color filter arranging RGB color filters on a
%square grid. It's pattern is 50% green, %25 red and %25 blue.Hence it is
%also called BGGR, RGBG, RGGB. It's based on using twice as many green(luminance-sensitive)
%elements as red or blue(chrminance-sensitive) to mimic the physiology of
%the human eye.
%Sampsa values after interpolation become image pixels.

%Separate the images into subchannels
I11 = I1(:,:,1);
I12 = I1(:,:,2);
I13 = I1(:,:,3);

I21 = I2(:,:,1);
I22 = I2(:,:,2);
I23 = I2(:,:,3);
figure,
subplot(2,3,1)
imshow(I11);
subplot(2,3,2)
imshow(I12);
subplot(2,3,3)
imshow(I13);
subplot(2,3,4)
imshow(I21);
subplot(2,3,5)
imshow(I22);
subplot(2,3,6)
imshow(I23);

%%implement RGGB
BI11 = RGGB (I11);
BI12 = RGGB (I12);
BI13 = RGGB (I13);


BI21 = RGGB (I21);
BI22 = RGGB (I22);
BI23 = RGGB (I23);
figure,
subplot(3,1,1)
imshow(BI11);
subplot(3,1,2)
imshow(BI12);
subplot(3,1,3)
imshow(BI13);

figure,
subplot(3,1,1)
imshow(BI21);
subplot(3,1,2)
imshow(BI22);
subplot(3,1,3)
imshow(BI23);
% mean-variance of the sliding window on the out-of-focus 
% analysis on subchannel

[m11,v11] = slidewindow(I11,13,77);
[m12,v12] = slidewindow(I12,13,77);
[m13,v13] = slidewindow(I13,13,77);
figure,
scatter([m11 m12 m13], [v11 v12 v13]);
x = [m11 m12 m13]';
y = [v11 v12 v13]';
ft = fittype('a*x-b');
f = fit(x, y, ft,'StartPoint',[1 2])
plot(f,x,y);
%     General model:
%     f(x) = a1*x-b1
%     Coefficients (with 95% confidence bounds):
       a1 =   7.9731e-3  %(-8.165e-3, 8.36e-3)
       b1 =   1.196  %(-1.225, 1.255)

[m21,v21] = slidewindow(I21,12,66);
[m22,v22] = slidewindow(I22,12,66);
[m23,v23] = slidewindow(I23,12,66);
figure,
scatter([m21 m22 m23], [v21 v22 v23]);
x = [m21 m22 m23]';
y = [v21 v22 v23]';
ft = fittype('a*x-b');
f = fit(x, y, ft,'StartPoint',[1 2])
plot(f,x,y);
%     General model:
%     f(x) = a2*x-b2
%     Coefficients (with 95% confidence bounds):
      a2 =  1.568e-5  %(-1.15e-5, 1.743e-5)
      b2 =  1.142  %(-1.061, 1.33)
%%apply the dct
Id11 = dct2((I11));
Id12 = dct2((I12));
Id13 = dct2((I13));

Id21 = dct2((I21));
Id22 = dct2((I22));
Id23 = dct2((I23));
figure,
subplot(3,1,1)
imshow(Id11);
subplot(3,1,2)
imshow(Id12);
subplot(3,1,3)
imshow(Id13);

figure,
subplot(3,1,1)
imshow(Id21);
subplot(3,1,2)
imshow(Id22);
subplot(3,1,3)
imshow(Id23);
%%apply the transformation
I1n = 2*(I1/a1+3/8+b1/(a1)^2);
I2n = 2*(I2/a2+3/8+b2/(a2)^2);

I2n(:,:,1)= I2n1;
I2n(:,:,2)= I2n2;
I2n(:,:,3)= I2n3;
[m21,v21] = slidewindow(I21,12,66);
[m22,v22] = slidewindow(I22,12,66);
[m23,v23] = slidewindow(I23,12,66);
figure,
scatter([m21 m22 m23], [v21 v22 v23]);

%%Apply the inverse transformation
I2i = a2*(1/4*I2n.^2+1/4*sqrt(3/2)/I2n-11/8/I2n.^2+5/8*sqrt(3/2)/I2n.^3-1/8-b2/a2.^2);
figure,
subplot(3,1,1)
imshow(I2)
subplot(3,1,2)
imshow(I2n)
subplot(3,1,3)
imshow(I2i)

%%demosaic the images
[X,Y] = meshgrid(0:255,0:255);
Z = BI11;
[Xq,Yq] = meshgrid(0:255);
Zq = interp2(X,Y,Z,Xq,Yq,'cubic');
figure,
surf(Xq, Yq, Zq);
title('Cubic Interpolation Over Finer Grid');

%%Compose a color image
B = uint8(BI11+BI12+BI13);
imshow(B)

%%white balance
% project to YUV
xfm =   [0.299 0.587 0.144; ...
        -0.299 -0.587 0.886; ...
        0.701 -0.587 -0.114];
%rgbImage = B*xfm; %convert to YUV
rgbImage(:,:,1) = B(:,:,3);
rgbImage(:,:,2) = B(:,:,1);
rgbImage(:,:,3) = B(:,:,2);
% To get the highest luminance place white equals to get the mean of the
% luminance of the three colors the same.
grayImage = rgb2gray(rgbImage); % Convert to gray so we can get the mean luminance.
correctedImage = imadjust(rgbImage,[],[],0.5); % adjust the contrast, apecifying a gamma 

% Extract the individual red, green, and blue color channels.
redChannel = rgbImage(:, :, 1);
greenChannel = rgbImage(:, :, 2);
blueChannel = rgbImage(:, :, 3);
meanR = mean2(redChannel);
meanG = mean2(greenChannel);
meanB = mean2(blueChannel);
meanGray = mean2(grayImage);
meanCorrected = mean2(correctedImage);
% Make all channels have the same mean
redChannel = uint8(double(redChannel) * meanGray / meanR);
greenChannel = uint8(double(greenChannel) * meanGray / meanG);
blueChannel = uint8(double(blueChannel) * meanGray / meanB);
% Recombine separate color channels into a single, true color RGB image.
rgbImage = cat(3, redChannel, greenChannel, blueChannel);
imshow(rgbImage);
%the result of the one after gamma correction
redChannel1 = uint8(double(redChannel) * meanCorrected / meanR);
greenChannel1 = uint8(double(greenChannel) * meanCorrected / meanG);
blueChannel1 = uint8(double(blueChannel) * meanCorrected / meanB);
meanCorrected = mean2(correctedImage);
figure,
rgbImage1 = cat(3, redChannel1, greenChannel1, blueChannel1);
imshow(rgbImage1);
%
