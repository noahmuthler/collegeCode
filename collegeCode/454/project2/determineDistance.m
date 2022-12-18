function worldPoints = determineDistance(image1,parameters1,image2,parameters2)
    % reads in images and their parameters, allows users to choose
    % corresponding points in the two images, and returns the points' 3D
    % locations

    figure(1); imagesc(image1); axis image; drawnow;
    figure(2); imagesc(image2); axis image; drawnow;
    
    figure(1); [x1,y1] = getpts;
    figure(1); imagesc(image1); axis image; hold on
    for i=1:length(x1)
       h=plot(x1(i),y1(i),'*'); set(h,'Color','g','LineWidth',2);
       text(x1(i),y1(i),sprintf('%d',i));
    end
   
    hold off
    drawnow;
    
    figure(2); imagesc(image2); axis image; drawnow;
    [x2,y2] = getpts;
    figure(2); imagesc(image2); axis image; hold on
    
    for i=1:length(x2)
       h=plot(x2(i),y2(i),'*'); set(h,'Color','g','LineWidth',2);
       text(x2(i),y2(i),sprintf('%d',i));
    end

    image1_pixel = [x1'; y1'];
    image1_pixel(3,:) = ones([1 size(x1,1)]);

    image2_pixel = [x2'; y2'];
    image2_pixel(3,:) = ones([1 size(x1,1)]);

    worldPoints = task3_2(image1_pixel,parameters1,image2_pixel,parameters2);
end

