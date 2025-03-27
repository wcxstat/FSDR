function SImat=FSAVE(Y,X,arg,H)
% functional SAVE (Lian & Li,2014,JMVA)

% Input:
% Y: 1*n response
% X: n*length(arg) matrix, dense and regular function data
% H: the number of slices

n=length(Y); % sample size
ngrid=length(arg); % number of sample point
delta=arg(2)-arg(1);
X_mean=mean(X,1); % sample mean function
X_cen=X-X_mean; %center
xcov=X_cen'*X_cen/n; % covariance

point=linspace(min(Y),max(Y),H+1);
SImat=zeros(ngrid,ngrid);
for h=1:H
    if h==H
        index=point(h)<=Y & Y<=point(h+1);
    else
        index=point(h)<=Y & Y<point(h+1);
    end
    pr=mean(index);
    if pr>0
        scov=X_cen(index,:)'*X_cen(index,:)/sum(index);
        SImat=SImat+pr*(xcov-scov)^2*delta;
    end
end