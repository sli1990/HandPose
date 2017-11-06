function [ x,y,z ] = uvd2xyz( u,v,d )
%UVD2XYZ Summary of this function goes here
%   Detailed explanation goes here
    u0 = 315.944855;    v0 = 245.287079;
    fx = 475.065948;    fy = 475.065857;
    x = ((u-1-u0)).*d./fx;
    y = ((v-1-v0)).*d./fy;
    z = d;
end

