function [a,b,c,d] = calcPlane(points)
    % reads in 3 coplanar points and returns the plane that contains all
    % three points
    
    point1 = points(:,1);
    point2 = points(:,2);
    point3 = points(:,3);

    a1 = point2(1) - point1(1);
    b1 = point2(2) - point1(2);
    c1 = point2(3) - point1(3);
    a2 = point3(1) - point1(1);
    b2 = point3(2) - point1(2);
    c2 = point3(3) - point1(3);
    a = b1 * c2 - b2 * c1;
    b = a2 * c1 - a1 * c2;
    c = a1 * b2 - b1 * a2;
    d = (-a * point1(1) - b * point1(2) - c * point1(3));
end

