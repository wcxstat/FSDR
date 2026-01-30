function yhat = ks(x,y,opt,h,xnew)

% ks  local constant/local linear kernel regression

% x: n-by-k matrix
% y: n-by-1 vector
% h: a positive scalar
% opt: 'kr' or [] if (local constant) kernel regression, 'krll' if local linear kernel regression

% yhat = ks(x,y,opt,h) implements a multi-dimensional nonparametric regression, 
% i.e., kernel regression (the default) or local linear regression using the Quartic kernel. 
% yhat = ks(x,y,opt,h) returns the fitted value yhat at each obs. of x.

% yhat = ks(x,y) performs kernel regression using the optimal bandwidth estimated by cross validation (calls function opt_h).
% yhat = ks(x,y,opt) performs the specified (by opt) nonparametric regression using the optimal bandwidth estimated by cross validation.
% yhat = ks(x,y,opt,h) performs the specified (by opt) nonparametric regression using the provided bandwidth, h.

% Copyright: Yingying Dong at Boston College, July, 2008.

%%

[nt,k] = size(xnew); b=zeros(k,nt); yhat = zeros(nt,1);

for i = 1:nt
    dis = bsxfun(@minus, x, xnew(i,:));
    u = bsxfun(@rdivide, dis, std(dis))/h;
    % Kernel = @(u)15/16*(1-u.^2).^2.*(abs(u)<=1);
    Kernel = @(u)1/sqrt(2*pi)*exp(-u.^2/2);
    w = prod(Kernel(u),2);
    sw = sum(w);
    if strcmp(opt, 'krll')
        t1 = bsxfun(@times, dis, w)';
        t2 = sum(t1,2);
        b(:,i)= (sw*t1*dis - t2*t2')\(sw*t1*y - t2*w'*y);
        yhat(i) = w'*(y - dis*b(:,i))/sw;
    else
        yhat(i) = w'*y/sw;
    end
end

%fprintf('\n%4.0f fitted values are generated  \n', n);