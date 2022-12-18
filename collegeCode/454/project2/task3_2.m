function [estimated3DPoints] = task3_2(pixelPoints2D_img1,parameters1,pixelPoints2D_img2,parameters2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % pixel2DtoWorld3D - converts corresponding 2D image coordiantes into
    % 3D world coordinates via triangularization
    % 
    %
    % inputs:   pixelPoints2D_img1 - vector of 2D coordinates for image 1
    %           parameters1 - struct including camera 1 calibration parameters
    %           pixelPoints2D_img2 - vector of 2D coordinates for image 2
    %           parameters2 - struct including camera 2 calibration parameters
    %
    % outputs:  estimated3DPoints - vector of corresponding 3D world
    %                               coordinates
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    estimated3DPoints = zeros(size(pixelPoints2D_img1));
    % extracting P and K matrices from parameters structs
    % extracting center of cameras
    Pmat1 = parameters1.Parameters.Pmat;
    Pmat2 = parameters2.Parameters.Pmat;
    R1 = Pmat1(:,1:3);
    t1 = Pmat1(:,4);
    R2 = Pmat2(:,1:3);
    t2 = Pmat2(:,4);
    Kmat1 = parameters1.Parameters.Kmat;
    Kmat2 = parameters2.Parameters.Kmat;
    c1 = -1*R1'*t1;
    c2 = -1*R2'*t2;

    % utilizing triangularization to estimate 3D world coordinates
    for i = 1:size(estimated3DPoints,2)
        % calculating vectors pointing from camera 1 and 2 to the point in
        % 3D space
        u1 = R1'*inv(Kmat1)*pixelPoints2D_img1(:,i);
        u1 = u1 / norm(u1);
        
        u2 = R2'*inv(Kmat2)*pixelPoints2D_img2(:,i);
        u2 = u2 / norm(u2);
        
        % finding the third vector via cross product
        u3 = cross(u1,u2);
    
        % finding coeffiecents a,d,b in equation below:
        %   a*u1 + d*u3 - b*u2 = c2 - c1
        C = c2 - c1;
        
        U = [u1 u3 u2];
        a_d_b = linsolve(U,C);

        a = a_d_b(1);
        d = a_d_b(2);
        b = a_d_b(3) * -1;
        
        % constructing p1 and p2 vectors pointing from camera 1 and 2 to
        % point in 3D space
        p1 = c1 + a * u1;
        p2 = c2 + b * u2;

        % determining the estimated 3D coordinate
        estimated3DPoints(:,i) = (p1 + p2) / 2;
    end
end

