function Ps=proj(u,arg)
% This is a program to calculate the projection operator on S
% S is spaned by u_1,u_2,...,u_d

% Output: the kernel function (matrix) of the operator

d=size(u,2);
A=zeros(d,d);
for j=1:d
    for k=1:d
        A(j,k)=trapz(arg,u(:,j).*u(:,k));
    end
end
Ps=u/A*u';