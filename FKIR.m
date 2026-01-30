function KImat=FKIR(Y,X,H)
% functional k-means inverse regression (Wang et al.,2014,CSDA)

% Input:
% Y: 1*n response
% X: n*length(arg) matrix, dense and regular function data
% H: the number of clusters

X_mean=mean(X,1); % sample mean function
idx=kmeans(Y,H);
KImat=0;
for h=1:H
    index=(idx==h);
    pr=mean(index);
    cmean=mean(X(index,:),1)-X_mean;
    KImat=KImat+cmean'*cmean*pr;
end