%---------------------------------------------------------------------------------------------------------%
%  READING THE IMAGES FROM THE train-Med-img DIRECTORY in srcFiles
%---------------------------------------------------------------------------------------------------------%

srcFiles = dir('..\train-Med-img\*.jpg');
A = zeros(190,200);
A = extractLBPFeatures(A);
J = A;

%---------------------------------------------------------------------------------------------------------%
%  READING ALL IMAGES FROM srcFiles in I .
%  1. Then converting it to a Grayscale image from which we will extract features.
%  2. Then resizing all image to a standard size here I have taken it 190 x 200.
%  3. Extracting the LBP features using 'extractLBPFeatures(I)' will give me a 59 x 1 matrix.
%  4. If the data after extraction of LBP Features is not normalised we can normalise it.
%  5. Adding all images features to 'J'.
%  6. There are total 99 images in the training set.
%---------------------------------------------------------------------------------------------------------%
for i = 1 : length(srcFiles)
  filename = strcat('..\train-Med-img\',srcFiles(i).name);
  I = imread(filename);
  I = rgb2gray(I);
  I = imresize(I,[190 200]);
  I = extractLBPFeatures(I);
  %I = bsxfun(@rdivide,I,sum(I));
  J=[J ; I];
end

%--------------------------------------------------------------------------------------------------------%
%  Writing the whole Training output in 'y'
%--------------------------------------------------------------------------------------------------------%
y = [0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 1 1];
y= y';


%--------------------------------------------------------------------------------------------------------%
%  Training the SVM Classifier using Gausssian Kernel and storing the
%  classifer in 'cl'
%--------------------------------------------------------------------------------------------------------%

cl= fitcsvm(J, y, 'KernelFunction','rbf','BoxConstraint',30);


%--------------------------------------------------------------------------------------------------------%
%  READING THE IMAGES FROM THE test-Med-img Directory in srctestFiles
%--------------------------------------------------------------------------------------------------------%

srctestFiles = dir('..\test-Med-img\*.jpg');
Test= A;

%--------------------------------------------------------------------------------------------------------%
% DOING THE SAME THING THAT WE HAVE DONE FOR THE TRAINING DATASET
% READING ALL IMAGES FROM srctestFiles in A .
%  1. Then converting it to a Grayscale image from which we will extract features.
%  2. Then resizing all image to a standard size here I have taken it 190 x 200.
%  3. Extracting the LBP features using 'extractLBPFeatures(A)' will give me a 59 x 1 matrix.
%  4. If the data after extraction of LBP Features is not normalised we can normalise it.
%  5. Adding all images features to 'Test'.
%  6. There are total 32 images in the test data set.
%--------------------------------------------------------------------------------------------------------%

for i = 1 : length(srctestFiles)
    test_filename= strcat('..\test-Med-img\',srctestFiles(i).name);
    A = imread(test_filename);
    A = rgb2gray(A);
    A = imresize(A, [190 200]);
    A = extractLBPFeatures(A);
    Test = [Test; A];
end
%--------------------------------------------------------------------------------------------------------%
%  Setting the correct output to be match with
%--------------------------------------------------------------------------------------------------------%
ans= [0 1 1 1 1 1 1 0 1 0 0 1 1 1 1 1 1 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1];
%ans= ans';

%--------------------------------------------------------------------------------------------------------%
%  Getting the predicted output by the classifier
%--------------------------------------------------------------------------------------------------------%
pred_out= 0;
for i = 2: 33
    k = cl.predict(Test(i,:));
    %disp(Test(i,:));
    pred_out = [pred_out; k];
end;

%--------------------------------------------------------------------------------------------------------%
%  Percentage Accuracy 
%--------------------------------------------------------------------------------------------------------%
pred_out= pred_out';
c=0;
for i = 2:33
    if ans(1,i-1)==pred_out(1,i)
        c = c + 1;
    end;
end;

fprintf('Accuracy: %i\n',c/32 *100);

%--------------------------------------------------------------------------------------------------------%
%  Accuracy is about:  46.87 %
%--------------------------------------------------------------------------------------------------------%