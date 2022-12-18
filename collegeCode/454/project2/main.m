%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Project 2 - EE 454 Fall 2022
% Group members: Noah Muthler, Luke Trumpbour, Noah Webb, Evan Soisson
%
% Each part of the project is split up into different functions, they are
% shown below.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;
clc;
close all;

image1 = imread('im1corrected.jpg');
image2 = imread('im2corrected.jpg');

parameters1 = load('Parameters_V1_1.mat');
parameters2 = load('Parameters_V2_1.mat');

mocapPoints3D = load('mocapPoints3D.mat');
pts3D = mocapPoints3D.pts3D;

% part 1 - Projecting 3D mocap points into 2D pixel locations
[pixelPoints2D_img1] = task3_1(pts3D, parameters1);
[pixelPoints2D_img2] = task3_1(pts3D, parameters2);

figure(1);
imshow(image1);
hold on;
scatter(pixelPoints2D_img1(1,:),pixelPoints2D_img1(2,:));

figure(2);
imshow(image2);
hold on;
scatter(pixelPoints2D_img2(1,:),pixelPoints2D_img2(2,:));

% part 2 - Triangulation to recover 3D mocap points from two views
[estimated3DPoints] = task3_2(pixelPoints2D_img1,parameters1,pixelPoints2D_img2,parameters2);

% mean squared error
error = 0;
for i = 1:39
    error = error + norm((estimated3DPoints(:,i) - pts3D(:,i)))^2;
end
error = error / 39;
fprintf("Mean squared error between estimated and given 3D world points is: %d\n", error);

% part 3 - Triangulation to make measurements about the scene
[floorPlane, floorEq, wallPlane, wallEq, doorHeight, personHeight, cameraCoords] = task3_3(image1,parameters1,image2,parameters2);

% part 4 - Compute the Fundamental matrix from known camera calibration parameters
F_compute = task3_4(parameters1,parameters2);
draw_epipolar_lines(F_compute, image1, image2, pixelPoints2D_img1, pixelPoints2D_img2);
% 
% % part 5 - Compute the Fundamental matrix using the eight-point algorithm
F_eightPoint = task3_5(image1,image2);

% part 6 - Quantitative evaluation of F matrices
retVals = task3_6(pixelPoints2D_img1, pixelPoints2D_img2, F_compute, F_eightPoint);

% part 7 - Generate a similarity-accurate top-down view of the floor plane
dest = task3_7(image1, parameters1);

