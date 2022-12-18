%% Read in the images
classes = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'};

trainFolder = 'cifar10Train';
testFolder = 'cifar10Test';
flippedFolder = 'flippedImages';
%imds = imageDatastore(fullfile(folder, classes,'LabelSource','foldernames'));
imds = imageDatastore(fullfile(trainFolder, classes),'LabelSource', 'foldernames');
imdsTest = imageDatastore(fullfile(testFolder, classes),'LabelSource', 'foldernames');

inputSize = [227 227];
imds.ReadFcn = @(loc)imresize(imread(loc),inputSize);
imdsTest.ReadFcn = @(loc)imresize(imread(loc),inputSize);

layers = [
    imageInputLayer([227 227 3])
	
    convolution2dLayer(11,96,'Stride',4, ...
    'WeightsInitializer', @(sz) rand(sz) * 0.0001, ...
    'BiasInitializer', @(sz) rand(sz) * 0.0001)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(3,'Stride',2)
	
    convolution2dLayer(5,256,'Stride', 1, 'Padding', 'same', ...
    'WeightsInitializer', @(sz) rand(sz) * 0.0001, ...
    'BiasInitializer', @(sz) rand(sz) * 0.0001)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(3,'Stride',2)
	
    convolution2dLayer(3,384,'Stride', 1, 'Padding', 'same', ...
    'WeightsInitializer', @(sz) rand(sz) * 0.0001, ...
    'BiasInitializer', @(sz) rand(sz) * 0.0001)
    reluLayer
    batchNormalizationLayer

    convolution2dLayer(3,384,'Stride', 1, 'Padding', 'same', ...
    'WeightsInitializer', @(sz) rand(sz) * 0.0001, ...
    'BiasInitializer', @(sz) rand(sz) * 0.0001)
    reluLayer
    batchNormalizationLayer

    convolution2dLayer(3,256,'Stride',1,'Padding','same', ...
    'WeightsInitializer', @(sz) rand(sz) * 0.0001, ...
    'BiasInitializer', @(sz) rand(sz) * 0.0001)
    reluLayer
    batchNormalizationLayer
    maxPooling2dLayer(3,'Stride',2)
    
    dropoutLayer(0.5)
    fullyConnectedLayer(4096)
    reluLayer
    dropoutLayer(0.5)
    fullyConnectedLayer(4096)
    reluLayer

    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

opts = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.005, ...
    'ValidationData', imdsTest, ...
    'ValidationFrequency', 5, ...
    'LearnRateSchedule', 'none', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 200, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment','gpu');

%% Run the net
[net, info] = trainNetwork(imds, layers, opts);

% determine accuracy
YPred = classify(net, imds);
YTrain = imds.Labels;

accuracyTrain = sum(YPred == YTrain) / numel(YTrain);

YPred = classify(net, imdsTest);
YValidation = imdsTest.Labels;

accuracyTest = sum(YPred == YValidation) / numel(YValidation);