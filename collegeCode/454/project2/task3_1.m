function [pixelPoints2D] = task3_1(pts3D, parameters)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Mocap3DToPixel2D - converts 3D world coordinates into corresponding
    % 2D image coordinates
    %
    % inputs:   pts3D - vector of 3D world coordinates
    %           parameters - struct including camera calibration parameters
    %
    % outputs:  pixelPoints2D - matrix of corresponding 2D pixel
    %                           coordinates
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    pts3D(4,:) = ones([1 size(pts3D,2)]);
    % extract K and P matrix from parameters
    K = parameters.Parameters.Kmat;
    P = parameters.Parameters.Pmat;
    pixelPoints2D = zeros([3 size(pts3D,2)]);
    % mulitpy each 3D point by K and P matrices in order to find
    % its corresponding 2D image location
    for i = 1:size(pts3D,2)
        pixelPoints2D(:,i) = K * P * pts3D(:,i);
        pixelPoints2D(:,i) = pixelPoints2D(:,i) / pixelPoints2D(3,i);
    end
end

