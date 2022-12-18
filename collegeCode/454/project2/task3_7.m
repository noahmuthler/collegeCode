function dest = task3_7(image, parameters)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % homography - reads in an image and its camera's parameters and
    % provides a top down view of the floor plane (code adapted from
    % professor)
    %
    % inputs:   image - the image
    %           parameters - camera calibration parameters
    %
    % outputs:  dest - final top down homography image
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    source1 = image;
    [nr,nc,nb] = size(source1);
    
    %make new image
    dest = zeros(size(source1));

    % define four points on the floor
    point1_world = [-1650; 1950; 0];
    point2_world = [2250; 1950; 0];
    point3_world = [2250; -2250; 0];
    point4_world = [-1650; -2250; 0];
    worldPoints = [point1_world point2_world point3_world point4_world];

    % find the length and height of the square
    length = abs(point2_world(1) - point4_world(1));
    height = abs(point2_world(2) - point4_world(2));
    
    % find the image location of the world points
    imgPoints = task3_1(worldPoints, parameters);
    xpts = imgPoints(1,:); ypts = imgPoints(2,:);

    % set scale factor to translate from millimeters to pixels
    scale_factor = 11000;
    
    % determine size of square in the final image
    xp1 = 400;
    yp1 = 300;
    xp2 = xp1 + height/scale_factor * nc;
    yp2 = yp1 + length/scale_factor * nr;
    xprimes = [xp1 xp2 xp2 xp1]';
    yprimes = [yp1 yp1 yp2 yp2]';

    %compute homography (from im2 to im1 coord system)
    p1 = xpts'; p2 = ypts';
    p3 = xprimes; p4 = yprimes;
    vec1 = ones(size(p1,1),1);
    vec0 = zeros(size(p1,1),1);
    Amat = [p3 p4 vec1 vec0 vec0 vec0 -p1.*p3 -p1.*p4; vec0 vec0 vec0 p3 p4 vec1 -p2.*p3 -p2.*p4];
    bvec = [p1; p2];
    h = Amat \ bvec;
    fprintf(1,'Homography:');
    fprintf(1,' %.2f',h); fprintf(1,'\n');
    
    %warp im1 forward into im2 coord system 
    [xx,yy] = meshgrid(1:size(dest,2), 1:size(dest,1));
    denom = h(7)*xx + h(8)*yy + 1;
    hxintrp = (h(1)*xx + h(2)*yy + h(3)) ./ denom;
    hyintrp = (h(4)*xx + h(5)*yy + h(6)) ./ denom;
    for b = 1:nb
     dest(:,:,b) = interp2(double(source1(:,:,b)),hxintrp,hyintrp,'linear')/255.0;
    end
    
    %display result
    figure;
    imagesc(dest);
end

