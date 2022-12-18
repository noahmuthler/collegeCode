function draw_epipolar_lines(F_compute, image1, image2, pixelPoints2D_img1, pixelPoints2D_img2)
    %overlay epipolar lines on image2
    F = F_compute;

    P = randperm(39,10);
    x1 = pixelPoints2D_img1(1, P)';
    y1 = pixelPoints2D_img1(2, P)';
    x2 = pixelPoints2D_img1(1, P)';
    y2 = pixelPoints2D_img2(2, P)';

    colors =  'bgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmykbgrcmyk';

    L = F * [x1' ; y1'; ones(size(x1'))];
    [nr,nc,nb] = size(image2);
    figure(2); clf; imagesc(image2); axis image;
    hold on; hold off
    for i=1:length(L)
        a = L(1,i); b = L(2,i); c=L(3,i);
        if (abs(a) > (abs(b)))
           ylo=0; yhi=nr;  
           xlo = (-b * ylo - c) / a;    
           xhi = (-b * yhi - c) / a;  
           hold on
           h=plot([xlo; xhi],[ylo; yhi]);
           set(h,'Color',colors(i),'LineWidth',2);
           hold off
           drawnow;  
        else
           xlo=0; xhi=nc;  
           ylo = (-a * xlo - c) / b; 
           yhi = (-a * xhi - c) / b;
           hold on  
           h=plot([xlo; xhi],[ylo; yhi],'b');
           set(h,'Color',colors(i),'LineWidth',2);  
           hold off  
           drawnow;
        end
    end

    %overlay epipolar lines on im1
    L = ([x2' ; y2'; ones(size(x2'))]' * F)' ;  
    [nr,nc,nb] = size(image1); 
    figure(1); clf; imagesc(image1); axis image;
    hold on; hold off 
    for i=1:length(L)
        a = L(1,i); b = L(2,i); c=L(3,i); 
        if (abs(a) > (abs(b)))
           ylo=0; yhi=nr;
           xlo = (-b * ylo - c) / a;
           xhi = (-b * yhi - c) / a;
           hold on 
           h=plot([xlo; xhi],[ylo; yhi],'b');
           set(h,'Color',colors(i),'LineWidth',2);
           hold off
           drawnow;
        else
           xlo=0; xhi=nc;
           ylo = (-a * xlo - c) / b;
           yhi = (-a * xhi - c) / b;
           hold on
           h=plot([xlo; xhi],[ylo; yhi],'b');
           set(h,'Color',colors(i),'LineWidth',2);  
           hold off
           drawnow;
        end 
    end
end

