function angle=Ang(v1,v2)
% This is a program to calculate the angle between
% each row of matrics v1 and v2
% where v1 and v2 have the same dimension

CosTheta=dot(v1,v2,2)./(vecnorm(v1,2,2).*vecnorm(v2,2,2));
CosTheta=max(min(CosTheta,1),-1);
angle=acos(CosTheta');