clear;
clc;
close all;

load('dataTable.mat') % labelData

resizeLabel = labelData;
shuffledIndices = randperm(height(resizeLabel));
idx = floor(0.8 * length(shuffledIndices) );

trainingIdx = 1:10000;
trainingDataTbl = resizeLabel(shuffledIndices(trainingIdx),:);

validationIdx = 10001:12500;
validationDataTbl = resizeLabel(shuffledIndices(validationIdx),:);

testIdx = 12501:13239;
testDataTbl = resizeLabel(shuffledIndices(testIdx),:);

imdsTrain = imageDatastore(trainingDataTbl{:,6});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,1:5));

imdsValidation = imageDatastore(validationDataTbl{:,6});
bldsValidation = boxLabelDatastore(validationDataTbl(:,1:5));

imdsTest = imageDatastore(testDataTbl{:,6});
bldsTest = boxLabelDatastore(testDataTbl(:,1:5));

trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);


networkInputSize = [416 416 3];
rng(0)
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);

area = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(area, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

numClasses = 5;

classNames = ["car","truck","pedestrian","biker","trafficLight"];

featureExtractionNetwork = resnet50;
featureLayer = 'activation_40_relu';
new(1:3, :) = anchorBoxes{1,1};
new(4:6, :) = anchorBoxes{2,1};
anchorBoxes = new;

lgraph = yolov2Layers([416 416 3],numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

options = trainingOptions('sgdm', ...
    'InitialLearnRate', 0.001, ...
    'ValidationData', validationData, ...
    'ValidationFrequency', 5, ...
    'LearnRateSchedule', 'none', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 8, ...
    'L2Regularization', 0.004, ...
    'MaxEpochs',5, ...
    'MiniBatchSize',20, ...
    'Verbose', true, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch', ...
    'ExecutionEnvironment','gpu');

[detector,info] = trainYOLOv2ObjectDetector(trainingDataForEstimation,lgraph,options);

I = imread(testData.UnderlyingDatastores{1, 1}.Files{1, 1});
I = imresize(I, [416 416]);
[bboxes,scores] = detect(detector,I);

I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
figure
imshow(I)
