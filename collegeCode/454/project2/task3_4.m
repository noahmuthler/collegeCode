function F = task3_4(parameters1,parameters2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % computeFundamentalMatrix - computes fundamental matrix between two
    % cameras using known camera parameter values
    %
    % inputs:   parameters1 - struct including camera 1 calibration 
    %                         parameters
    %           parameters2 - struct including camera 2 calibration 
    %                         parameters
    %
    % outputs:  F - fundamental matrix
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    % extracting all of the need information from the parameters structs
    rot_1 = parameters1.Parameters.Rmat;
    rot_2 = parameters2.Parameters.Rmat;
    camera2_world = parameters2.Parameters.position';
    camera2_world(4) = 1;
    Pmat = parameters1.Parameters.Pmat;
    Kmat1 = parameters1.Parameters.Kmat;
    Kmat2 = parameters2.Parameters.Kmat;
    
    % determining tx,ty,tz needed in S matrix
    tx_ty_tz = Pmat * camera2_world; % camera 2 position in world coordinates
    tx = tx_ty_tz(1);
    ty = tx_ty_tz(2);
    tz = tx_ty_tz(3);
    
    % constructing skew symmetric S matrix
    S = [0 -tz ty; tz 0 -tx; -ty tx 0];

    % constructing rotation matrix from camera 1 to camera 2
    R = rot_2 * inv(rot_1); 
    
    % constructing E matrix from the R and S matrices
    E = R * S;

    % constructing final F matrix from E and the K matrices from each
    % camera
    F = inv(Kmat2)' * E * inv(Kmat1); % constructing F from E
end

