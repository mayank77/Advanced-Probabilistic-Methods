function [ phi_x ] = phiX( x, d )
%PHI_X Summary of this function goes here
%   Detailed explanation goes here
    phi_x = ((x*ones(1,d+1)).^(0:d))';
end

