function retVals = task3_6(pixelPoints2D_img1, pixelPoints2D_img2, F_compute, F_eightPoint)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % evalFmatrices - computes analytical comparison between the two
    % fundatmental matrices found in this project
    %
    % inputs:   pixelPoints2D_img1 - corresponding pixel points from image
    %                                1
    %           pixelPoints2D_img1 - corresponding pixel points from image
    %                                2
    %           F_compute - Fundamental matrix found by the camera
    %                       parameters
    %           F_eightPoint - Fundamental matrix found by the eight point
    %                          algorithm
    %
    % outputs:  retVals - analytical comparison between two fundamental
    % matrices 
    %   (retVals(1) = sedErrorComputed & retVals(2) = sedErrorEightPoint)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    sedMeanComputed = 0;
    sedMeanEightPoint = 0;
   
    % for computed F matrix
    for i=1:39
        img1x = pixelPoints2D_img1(1, i);
        img1y = pixelPoints2D_img1(2, i);
        img2x = pixelPoints2D_img2(1, i);
        img2y = pixelPoints2D_img2(2, i);
       
        pointLeft = [img1x; img1y; 1];
        rightLine = F_compute * pointLeft;
        distancePointRight = (rightLine(1) * img2x) + (rightLine(2) * img2y) + rightLine(3);
        distancePointRight = distancePointRight * distancePointRight;
        divDistPntRight = (rightLine(1) * rightLine(1)) + (rightLine(2) * rightLine(2));
        addMeanRight = distancePointRight / divDistPntRight;
        sedMeanComputed = sedMeanComputed + addMeanRight;
   
        pointRight = [img2x; img2y; 1];
        leftLine = transpose(F_compute) * pointRight;
        distancePointLeft = (leftLine(1) * img1x) + (leftLine(2) * img1y) + leftLine(3);
        distancePointLeft = distancePointLeft * distancePointLeft;
        divDistPntLeft = (leftLine(1) * leftLine(1)) + (leftLine(2) * leftLine(2));
        addMeanLeft = distancePointLeft / divDistPntLeft;
        sedMeanComputed = sedMeanComputed + addMeanLeft;
   
    end
   
    % for eight point F matrix
    for i=1:39
        img1x = pixelPoints2D_img1(1, i);
        img1y = pixelPoints2D_img1(2, i);
        img2x = pixelPoints2D_img2(1, i);
        img2y = pixelPoints2D_img2(2, i);
       
        pointLeft = [img1x; img1y; 1];
        rightLine = F_eightPoint * pointLeft;
        distancePointRight = (rightLine(1) * img2x) + (rightLine(2) * img2y) + rightLine(3);
        distancePointRight = distancePointRight * distancePointRight;
        divDistPntRight = (rightLine(1) * rightLine(1)) + (rightLine(2) * rightLine(2));
        addMeanRight = distancePointRight / divDistPntRight;
        sedMeanEightPoint = sedMeanEightPoint + addMeanRight;
   
        pointRight = [img2x; img2y; 1];
        leftLine = transpose(F_eightPoint) * pointRight;
        distancePointLeft = (leftLine(1) * img1x) + (leftLine(2) * img1y) + leftLine(3);
        distancePointLeft = distancePointLeft * distancePointLeft;
        divDistPntLeft = (leftLine(1) * leftLine(1)) + (leftLine(2) * leftLine(2));
        addMeanLeft = distancePointLeft / divDistPntLeft;
        sedMeanEightPoint = sedMeanEightPoint + addMeanLeft;
   
    end
   
    sedErrorComputed = sedMeanComputed / 78;
    sedErrorEightPoint = sedMeanEightPoint / 78;
   
    retVals(1) = sedErrorComputed;
    retVals(2) = sedErrorEightPoint;
    fprintf("The SED error for camera parameter F: %d, SED error for eight-point algorithm F: %f\n", retVals(1), retVals(2));
end