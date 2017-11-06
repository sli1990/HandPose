function [ u, v ,d ] = xyz2uvd( x,y,z )
%XYZ2UVD Summary of this function goes here
%   Detailed explanation goes here
    u0 = 315.944855;    v0 = 245.287079;
    fx = 475.065948;    fy = 475.065857;
    u = x*fx./z + u0; 
    v = y*fy./z + v0;
    d = z;
    u = u+1;
    v = v+1;

end

