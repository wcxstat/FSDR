function SImat=FSIR(Y,X,H)
% functional sliced inverse regression (Ferre & Yao, 2003)

% Input:
% Y: 1*n response
% X: n*length(arg) matrix, dense and regular function data
% H: the number of slices

X_mean=mean(X,1); % sample mean function
point=linspace(min(Y),max(Y),H+1);
SImat=0;
for h=1:H
    if h==H
        index=point(h)<=Y & Y<=point(h+1);
    else
        index=point(h)<=Y & Y<point(h+1);
    end
    pr=mean(index);
    if pr>0
        smean=mean(X(index,:),1)-X_mean;
        SImat=SImat+smean'*smean*pr;
    end
end