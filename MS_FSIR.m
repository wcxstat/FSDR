function SImat=MS_FSIR(Y,X,H)
% Multivariate slicing based on functional sliced inverse regression

% Input:
% Y: n*p multivariate response
% X: n*length(arg) matrix, dense and regular function data
% H: the number of slices

[n,p]=size(Y);
X_mean=mean(X,1); % sample mean function
ms_index={1:n};
for j=1:p
    Yj=Y(:,j);
    point=linspace(min(Yj),max(Yj),H+1);
    index_new=cell(1,H*length(ms_index));
    for k=1:length(ms_index)
        index_k=ms_index{k};
        for h=1:H
            if h==H
                index=point(h)<=Yj & Yj<=point(h+1);
            else
                index=point(h)<=Yj & Yj<point(h+1);
            end
            index_new{(k-1)*H+h}=intersect(index_k,find(index));
        end
    end
    ms_index=index_new(cellfun(@(x) ~isempty(x), index_new));
end

SImat=0;
for h=1:length(ms_index)
    index_h=ms_index{h};
    pr=length(index_h)/n;
    smean=mean(X(ms_index{h},:),1)-X_mean;
    SImat=SImat+smean'*smean*pr;
end

