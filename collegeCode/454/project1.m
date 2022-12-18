%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project 1 - EE 454
% Group members:
%   Luke Trumpbour
%   Evan Soisson
%   Noah Muthler
%   Noah Webb
% Date: 10/2/2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% loading the test images to be used
load('cifar10testdata.mat');
% classlabels -> look up table for class labels
% imageset -> all of the images (access by imageset(:,:,:,i)
% trueclass -> target values
% loading the filter weights for the network
load('CNNparameters.mat');
% biasvectors -> vector of the bias values
% filterbanks -> all of the filters needed
% layertypes -> list of layers in CNN

correct = 0;
incorrect = 0;
% create a 2D array for the confusion matrix
tablearray = zeros(10,10);

% variables the calculation of the average prob when there is a correct prediction
airplanePerc = 0;
airplaneCount = 0;
autoPerc = 0;
autoCount = 0;
birdPerc = 0;
birdCount = 0;
catPerc = 0;
catCount = 0;
deerPerc = 0;
deerCount = 0;
dogPerc = 0;
dogCount = 0;
frogPerc = 0;
frogCount = 0;
horsePerc = 0;
horseCount = 0;
shipPerc = 0;
shipCount = 0;
truckPerc = 0;
truckCount = 0;


% %% full test %%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=1:10000
    % 1 normalize
    image1 = apply_imnormalize(imageset(:,:,:,i));
    % 2 convolve
    image2 = apply_convolve(image1, filterbanks{1,2}, biasvectors{1,2});
    % 3 relu
    image3 = apply_relu(image2);
    % 4 convolve
    image4 = apply_convolve(image3, filterbanks{1,4}, biasvectors{1,4});
    % 5 relu
    image5 = apply_relu(image4);
    % 6 maxpool
    image6 = apply_maxpool(image5);
    % 7 convolve
    image7 = apply_convolve(image6, filterbanks{1,7}, biasvectors{1,7});
    % 8 relu
    image8 = apply_relu(image7);
    % 9 convolve
    image9 = apply_convolve(image8, filterbanks{1,9}, biasvectors{1,9});
    % 10 relu
    image10 = apply_relu(image9);
    % 11 maxpool
    image11 = apply_maxpool(image10);
    % 12 convolve
    image12 = apply_convolve(image11, filterbanks{1,12}, biasvectors{1,12});
    % 13 relu
    image13 = apply_relu(image12);
    % 14 convolve
    image14 = apply_convolve(image13, filterbanks{1,14}, biasvectors{1,14});
    % 15 relu
    image15 = apply_relu(image14);
    % 16 maxpool
    image16 = apply_maxpool(image15);
    % 17 fullconnect
    image17 = apply_fullconnect(image16, filterbanks{1,17}, biasvectors{1,17});
    % 18 softmax
    output = apply_softmax(image17);
    % value of index is the predicted class
    [max_ index] = max(output);
    %fprintf('estimated class is %s with probability %.4f\n',...
    %classlabels{index},max_);
    %fprintf('actual output is %s\n',classlabels{trueclass(i)});
    %fprintf('Iteration: %i\n', i);
    if(index == trueclass(i))
        correct = correct + 1;
        % add the prob to the total var for each class
        % increase the total number of the correct guesses for the class
        if (index == 1)
            airplanePerc = airplanePerc + max_;
            airplaneCount = airplaneCount + 1;
        elseif (index == 2)
            autoPerc = autoPerc + max_;
            autoCount = autoCount + 1;
        elseif (index == 3)
            birdPerc = birdPerc + max_;
            birdCount = birdCount + 1;
        elseif (index == 4)
            catPerc = catPerc + max_;
            catCount = catCount + 1;
        elseif (index == 5)
            deerPerc = deerPerc + max_;
            deerCount = deerCount + 1;
        elseif (index == 6)
            dogPerc = dogPerc + max_;
            dogCount = dogCount + 1;
        elseif (index == 7)
            frogPerc = frogPerc + max_;
            frogCount = frogCount + 1;
        elseif (index == 8)
            horsePerc = horsePerc + max_;
            horseCount = horseCount + 1;
        elseif (index == 9)
            shipPerc = shipPerc + max_;
            shipCount = shipCount + 1;
        elseif (index == 10)
            truckPerc = truckPerc + max_;
            truckCount = truckCount + 1;
        end
    else
        incorrect = incorrect + 1;
    end
    % build the confusion matrix
    % add 1 to the node representing the actual class (x) and estimated class (y)
    tablearray(trueclass(i), index) = tablearray(trueclass(i), index) + 1;
end

% calculate each of the average probs of a correct prediction per class
airplaneCor = airplanePerc / airplaneCount;
autoCor = autoPerc / autoCount;
birdCor = birdPerc / birdCount;
catCor = catPerc / catCount;
deerCor = deerPerc / deerCount;
dogCor = dogPerc / dogCount;
frogCor = frogPerc / frogCount;
horseCor = horsePerc / horseCount;
shipCor = shipPerc / shipCount;
truckCor = truckPerc / truckCount;

% display the results from the calculations above
fprintf('Average Prob when a correct predicition for airplanes is made = %.4f\n', airplaneCor);
fprintf('Average Prob when a correct predicition for automobiles is made = %.4f\n', autoCor);
fprintf('Average Prob when a correct predicition for birds is made = %.4f\n', birdCor);
fprintf('Average Prob when a correct predicition for cats is made = %.4f\n', catCor);
fprintf('Average Prob when a correct predicition for deer is made = %.4f\n', deerCor);
fprintf('Average Prob when a correct predicition for dogs is made = %.4f\n', dogCor);
fprintf('Average Prob when a correct predicition for frogs is made = %.4f\n', frogCor);
fprintf('Average Prob when a correct predicition for horses is made = %.4f\n', horseCor);
fprintf('Average Prob when a correct predicition for ships is made = %.4f\n', shipCor);
fprintf('Average Prob when a correct predicition for truck is made = %.4f\n', truckCor);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Debugging %%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load('debuggingTest.mat');
% load('CNNparameters.mat');
% 
% % 1 normalize
% image1 = apply_imnormalize(imrgb);
% % 2 convolve
% image2 = apply_convolve(image1, filterbanks{1,2}, biasvectors{1,2});
% % 3 relu
% image3 = apply_relu(image2);
% % 4 convolve
% image4 = apply_convolve(image3, filterbanks{1,4}, biasvectors{1,4});
% % 5 relu
% image5 = apply_relu(image4);
% % 6 maxpool
% image6 = apply_maxpool(image5);
% % 7 convolve
% image7 = apply_convolve(image6, filterbanks{1,7}, biasvectors{1,7});
% % 8 relu
% image8 = apply_relu(image7);
% % 9 convolve
% image9 = apply_convolve(image8, filterbanks{1,9}, biasvectors{1,9});
% % 10 relu
% image10 = apply_relu(image9);
% % 11 maxpool
% image11 = apply_maxpool(image10);
% % 12 convolve
% image12 = apply_convolve(image11, filterbanks{1,12}, biasvectors{1,12});
% % 13 relu
% image13 = apply_relu(image12);
% % 14 convolve
% image14 = apply_convolve(image13, filterbanks{1,14}, biasvectors{1,14});
% % 15 relu
% image15 = apply_relu(image14);
% % 16 maxpool
% image16 = apply_maxpool(image15);
% % 17 fullconnect
% image17 = apply_fullconnect(image16, filterbanks{1,17}, biasvectors{1,17});
% % 18 softmax
% output = apply_softmax(image17);
% [max_ index] = max(output);
% 
% 
% %loading this file defines imrgb and layerResults
% load('debuggingTest.mat');
% %sample code to show image and access expected results
% figure; imagesc(imrgb); truesize(gcf,[64 64]);
% for d = 1:length(layerResults)
% result = layerResults{d};
% fprintf('layer %d output is size %d x %d x %d\n',...
% d,size(result,1),size(result,2), size(result,3));
% end
% %find most probable class
% classprobvec = squeeze(layerResults{end});
% [maxprob,maxclass] = max(classprobvec);
% %note, classlabels is defined in ’cifar10testdata.mat’
% fprintf('estimated class is %s with probability %.4f\n',...
% classlabels{maxclass},maxprob);
% fprintf('Predicted class is %s with probability %.4f\n', classlabels{index}, max_);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Testing our own image
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I = imread('cat_picture.jpeg');
% I = imresize(I, [32 32]);
% % 1 normalize
% image1 = apply_imnormalize(I);
% % 2 convolve
% image2 = apply_convolve(image1, filterbanks{1,2}, biasvectors{1,2});
% % 3 relu
% image3 = apply_relu(image2);
% % 4 convolve
% image4 = apply_convolve(image3, filterbanks{1,4}, biasvectors{1,4});
% % 5 relu
% image5 = apply_relu(image4);
% % 6 maxpool
% image6 = apply_maxpool(image5);
% % 7 convolve
% image7 = apply_convolve(image6, filterbanks{1,7}, biasvectors{1,7});
% % 8 relu
% image8 = apply_relu(image7);
% % 9 convolve
% image9 = apply_convolve(image8, filterbanks{1,9}, biasvectors{1,9});
% % 10 relu
% image10 = apply_relu(image9);
% % 11 maxpool
% image11 = apply_maxpool(image10);
% % 12 convolve
% image12 = apply_convolve(image11, filterbanks{1,12}, biasvectors{1,12});
% % 13 relu
% image13 = apply_relu(image12);
% % 14 convolve
% image14 = apply_convolve(image13, filterbanks{1,14}, biasvectors{1,14});
% % 15 relu
% image15 = apply_relu(image14);
% % 16 maxpool
% image16 = apply_maxpool(image15);
% % 17 fullconnect
% image17 = apply_fullconnect(image16, filterbanks{1,17}, biasvectors{1,17});
% % 18 softmax
% output = apply_softmax(image17);
% [max_ index] = max(output);
% fprintf('Real class is cat\n');
% fprintf('Predicted class is %s with probability %.4f\n',...
%     classlabels{index},max_);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%

% Here is the decleration of each function/layer type
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Normalization layer
function outarray = apply_imnormalize(inarray)
    % In this layer, a non-normalized NxMx3 image is inputted. Each element
    % of the image is divided by 255 and then 0.5 is subtracted. This
    % process normalizes the image and restricts the output range from -0.5
    % to 0.5. The output is NxMx3.
    inarray = double(inarray);
    [n m d] = size(inarray);
    outarray = zeros([n m d]);
    for i=1:n
        for j=1:m
            for k=1:d
                outarray(i,j,k) = inarray(i,j,k)/255.0 - 0.5;
            end
        end
    end
end

% ReLU layer
function outarray = apply_relu(inarray)
    % This layer is the activation function for each neuron in the network.
    % This allows the network to be non-linear. This function reads an
    % NxMxD image and zeros all of the negative values in the image. If a value
    % is positive it stays the same. The output is NxMxD.
    [n m d] = size(inarray);
    outarray = zeros([n m d]);
    for k=1:d
        for j=1:m
            for i=1:n
                if inarray(i,j,k) > 0
                    outarray(i,j,k) = inarray(i,j,k);
                end
            end
        end
    end
end

% Maxpool layer
function outarray = apply_maxpool(inarray)
    % This function defines the maxpool layer. The input is a 2Nx2MxD
    % image. The maxpooling layer takes each 2x2 square in the image and
    % returns the largest value in the patch. This process reduces the
    % output image size to NxMxD.
    [n m d] = size(inarray);
    outarray = zeros([n/2 m/2 d]);
    for k = 1:d
        for i = 1:n/2
            for j = 1:m/2
                outarray(i,j,k) = max(max(inarray(2*i-1:2*i,2*j-1:2*j,k)));
            end
        end
    end
end

% Convolution layer
function outarray = apply_convolve(inarray, filterbank, biasvals)
    % This function defines the convolution layers. The inputs are a
    % NxMxD1 image, a RxCxD1xD2 filter bank, and D2 length vecotor of bias
    % values. This function utilitzes the 2D imfilter() function from the
    % image processing toolbox. This function iterates over the size of the
    % image a convolves it with the given filters. This is done D2 times so
    % the output image is of size NxMxD2.
    [R C D1 D2] = size(filterbank);
    [n m d] = size(inarray);
    outarray = zeros([n m D2]);
    for i=1:D2
        for j=1:D1
            outarray(:,:,i) = outarray(:,:,i) + imfilter(inarray(:,:,j), filterbank(:,:,j,i),'conv');
        end
        outarray(:,:,i) = outarray(:,:,i) + biasvals(i);
    end
end

% Fully connected layer
function outarray = apply_fullconnect(inarray, filterbank, biasvals)
    % This function defines the fully connected layer. The inputs are a
    % NxMxD1 image, a NxMxD1xD2 filterbank, and a vecotor of length D2
    % called biasvals. The fully connected layer stems from a traditional
    % neural network. Since the inarray and filterbank have the same first
    % two dimensions, the fully connected layer acts as a dot product
    % operation between the two matricies. The fully connected layer
    % multiplies each element of the inarray and filter and sums each of
    % them up, and finally adds a bias value to the summation. This is done
    % D2 times. Therefore, the output is 1x1xD2.
    [N M D1 D2] = size(filterbank);
    outarray = zeros([1 1 D2]);
    for l=1:D2
        for i=1:N
            for j=1:M
                for k=1:D1
                    outarray(1,1,l) = outarray(1,1,l) + filterbank(i,j,k,l) * inarray(i,j,k);
                end
            end
        end
        outarray(1,1,l) = outarray(1,1,l) + biasvals(l);
    end
end

% Softmax layer
function outarray = apply_softmax(inarray)
    % input and output arrays are 1 x 1 x D
    % This function defines the softmax layer. The softmax layer readys in
    % the output of the fully connected layer and returns the probability
    % of classification for each element of the vector. This is done by the
    % following eqution: 
    %   e^(inarray(i) - max of innary) / sum(e^(inarray(k) - max) 
    % This is performed D2 times, resulting in a 1x1xD2 output.
    [n m d] = size(inarray);
    % get denominator for the outarray calculations
    alpha = max(inarray);
    denom = 0;
    outarray = zeros([1 1 d]);
    for i=1:d
        denom = denom + exp(inarray(1,1,i) - alpha);
    end
    for j=1:d
        outarray(1,1,j) = exp(inarray(1,1,j) - alpha) / denom;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
