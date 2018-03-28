%--------------------------------------------------------------------------------------------%
%  Reading Input of a Image from train-Med-img Folder
%  The image can be changed by changing the name of the file.
%--------------------------------------------------------------------------------------------%
A = imread('../train-Med-img/05.jpg');
A = imresize(A,[300 300]);
%figure


%-------------------------------------------------------------------------------------------%
%  Converting into Grayscale Image
%-------------------------------------------------------------------------------------------%
Agray = rgb2gray(A);
%surf(Agray);
%imshow(Agray);
%figure
%imshow(A);
imageSize = size(A);
numRows = imageSize(1);
numCols = imageSize(2);

%------------------------------------------------------------------------------------------%
%  Creating Gabor Filter according to it's standard method of taking
%  wavelength and orientation 
%------------------------------------------------------------------------------------------%

wavelengthMin = 4/sqrt(2);
wavelengthMax = hypot(numRows,numCols);
n = floor(log2(wavelengthMax/wavelengthMin));
wavelength = 2.^(0:(n-2)) * wavelengthMin;

deltaTheta = 45;
orientation = 0:deltaTheta:(180-deltaTheta);

g = gabor(wavelength,orientation);

gabormag = imgaborfilt(Agray,g);

%-----------------------------------------------------------------------------------------%
%  Applying Gaussian Filter on Gabor Filter
%-----------------------------------------------------------------------------------------%

for i = 1:length(g)
    sigma = 0.2*g(i).Wavelength;
    K = 2;
    gabormag(:,:,i) = imgaussfilt(gabormag(:,:,i),K*sigma); 
end
%figure
%imshow(gabormag);
X = 1:numCols;
Y = 1:numRows;
[X,Y] = meshgrid(X,Y);
featureSet = cat(3,gabormag,X);
featureSet = cat(3,featureSet,Y);

numPoints = numRows*numCols;
X = reshape(featureSet,numRows*numCols,[]);

%----------------------------------------------------------------------------------------%
%  Normalizing the Data
%----------------------------------------------------------------------------------------%

X = bsxfun(@minus, X, mean(X));
X = bsxfun(@rdivide,X,std(X));

coeff = pca(X);
feature2DImage = reshape(X*coeff(:,1),numRows,numCols);
%figure
%imshow(feature2DImage,[])
%figure
%surf(feature2DImage);

[a,b]=size(feature2DImage);
%figure
%[count,x]=imhist(feature2DImage,a*b);
%plot(x,count);
%stem(count,x);
%---------------------------------------------------------------------------------------%
%  Segmenting the Image into 'N' parts using K-Mean Segmentation
%---------------------------------------------------------------------------------------%

N=2;                   % Vary the value of N to segment into large number of segments

L = kmeans(X,N,'Replicates',5);

L = reshape(L,[numRows numCols]);
figure
imshow(label2rgb(L))
%figure
%surf(L);
rgb_label = repmat(L,[1 1 3]);
%figure
%surf(rgb_label);
segmented_images = cell(1,N);

%---------------------------------------------------------------------------------------%
%  Storing Segmented Image into segmented image array
%---------------------------------------------------------------------------------------%

for k = 1:N
    color = A;
    color(rgb_label ~= k) = 0;
    segmented_images{k} = color; 
end
%ishandle(segmented_images{1});
%ishandle(segmented_images{2});
%mkdir('processedimage');

%--------------------------------------------------------------------------------------%
%  If N>2 remove comment sign i.e., from 3rd and 4th line or can add more
%  line same as 1 and 2
%  Segmented Image will be saved in the same directory i.e., in which these files are saved.
%--------------------------------------------------------------------------------------%


imwrite(segmented_images{1},'Fig1.jpg');
imwrite(segmented_images{2},'Fig2.jpg');
%imwrite(segmented_images{3},'Fig3.png');
%imwrite(segmented_images{4},'Fig4.png');
%subplot(2,1,1);
%imshow(segmented_images{1}), title('cluster 1');
%subplot(2,1,2);
%imshow(segmented_images{2}), title('cluster 2');
%subplot(2,2,3);
%imshow(segmented_images{3}), title('cluster 3');
%subplot(2,2,4);
%imshow(segmented_images{4}), title('cluster 4');

%Aseg1 = zeros(size(A),'like',A);
%Aseg2 = zeros(size(A),'like',A);
%BW = L == 2;
%BW = repmat(BW,[1 1 3]);
%Aseg1(BW) = A(BW);
%Aseg2(~BW) = A(~BW);
%figure
%imshowpair(Aseg1,Aseg2,'montage');