function [eigval,eigfun] = inteq(xcov, arg, noeig)
% This is a program to calculate the integral equation, 
% which appeared at FPCA context
% note:   trapezoid method will be used to approximate the integration

% Input
% xcov:   a symmetric matrix, which corresponding to a kernel function taking
%         value at a set of discrete points
% arg:    a set of discrete time point
% noeig:  number of eigenvalues that needed

% Output
% eigval: a set of eigenvalues
% eigfun: a set of eigenfunctions

ngrid = length(arg);
arg1 = [arg(2:end),arg(ngrid)];
arg2 = [arg(1),arg(1:(ngrid-1))];
weight = (arg1 - arg2)/2;
V = diag(sqrt(weight));
xcov1 = V * xcov * V;
[eigen, d] = eigs((xcov1+xcov1')/2,noeig,'lm');
eigval = diag(d); % resulted eigenvalue
eigfun = eigen;
eigfun = diag(1./sqrt(weight)) * eigfun;
for i = 1: noeig
    eigfun(:,i) = eigfun(:,i)/sqrt(trapz(arg,eigfun(:,i).^2));
    if eigfun(2,i) < eigfun(1,i)
       eigfun(:,i) = -eigfun(:,i);
    end
end