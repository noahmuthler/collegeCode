function [floorPlane, floorEq, wallPlane, wallEq, doorHeight, personHeight, cameraCoords] = task3_3(image1,parameters1,image2,parameters2)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % distancesOfImageObjects - Reads into two images and their camera
    % calibration parameters to make measurements in the scene including
    % the plane of the floor, the plane of the striped wall, the height of
    % the door, the height of the person, and the coordinate of the camera
    % located by the striped wall's center.
    %
    % inputs:   image1 - the first image
    %           image2 - the second image
    %           parameters1 - struct including camera 1's calibration 
    %                         parameters
    %           parameters2 - struct including camera 2's calibration
    %                         parameters
    %
    % outputs:  floorPlane - plane of the floor in world coordinates
    %           floorEq - equation of floor plane
    %           wallPlane - plane of the striped wall in world coordinates
    %           wallEq - equation of the striped wall's plane
    %           doorHeight - height of the door in meters
    %           personHeight - height of the person in meters
    %           cameraCoordinates - world coordiantes of the camera along
    %                               the striped wall
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % reading in three corresponding points to determine floor plane
    fprintf("Click three corresponding points on the floor.\n")
    worldPoints_floor = determineDistance(image1,parameters1,image2,parameters2);
    floorPlane = worldPoints_floor / 1000;
    
    % reading in three corresponding points to determine strped wall plane
    fprintf("Click three corresponding points on the striped wall.\n")
    worldPoints_wall = determineDistance(image1,parameters1,image2,parameters2);
    wallPlane = worldPoints_wall / 1000;
    
    % reading in two corresponding points to determine door's height
    fprintf("Click a point at the bottom of the door and the top of the door.\n")
    worldPoints_door = determineDistance(image1,parameters1,image2,parameters2);
    doorHeight = worldPoints_door(3,2)/1000 - worldPoints_door(3,1)/1000;
    
    % reading in two corresponding points to determine person's height
    fprintf("Click a point at the bottom of the person and the top of the person.\n")
    worldPoints_person = determineDistance(image1,parameters1,image2,parameters2);
    personHeight = worldPoints_person(3,2)/1000 - worldPoints_person(3,1)/1000;
    
    % reading in one corresponding point to determine camera's world
    % coordiantes
    fprintf("Click on the center of the camera along the striped wall.\n")
    cameraCoords = determineDistance(image1,parameters1,image2,parameters2);
    cameraCoords = cameraCoords / 1000;

    % calculating the equation of the floor and wall
    % printing out all information
    [a1,b1,c1,d1] = calcPlane(floorPlane);
    floorEq = [a1,b1,c1,d1];
    fprintf("Floor plane: %fx + %fy + %fz + %f = 0\n", a1,b1,c1,d1)
    [a2,b2,c2,d2] = calcPlane(wallPlane);
    wallEq = [a2,b2,c2,d2];
    fprintf("Wall plane: %fx + %fy + %fz + %f = 0\n", a2,b2,c2,d2)
    fprintf("Door height: %f\n", doorHeight);
    fprintf("Person height: %f\n", personHeight);
    fprintf("Camera coordinates: (%f,%f,%f)\n", cameraCoords(1), cameraCoords(2), cameraCoords(3));
    
    
    % determining all of the pixel coordinates needed for the figure
    points_floor = task3_1(floorPlane * 1000, parameters1);
    points_wall = task3_1(wallPlane * 1000, parameters1);
    points_man = task3_1(worldPoints_person, parameters1);
    points_door = task3_1(worldPoints_door, parameters1);
    point_camera = task3_1(cameraCoords * 1000, parameters1);
    
    % displaying the floor's plane, the wall's plane, the height of the
    % door, the height of the person, and the location of the camera
    figure;
    imshow(image1); hold on; 
    fill3(points_floor(1,:),points_floor(2,:),points_floor(3,:),'r'); alpha(0.5); 
    fill3(points_wall(1,:),points_wall(2,:),points_wall(3,:),'b'); alpha(0.5)
    plot(points_man(1,:), points_man(2,:),'LineWidth',5)
    plot(points_door(1,:), points_door(2,:),'LineWidth',5)
    plot(point_camera(1), point_camera(2), 'r*')
end

