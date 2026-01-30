function opth = opt_h(x,y,opt,h0)

% x: n-by-k matrix
% y: n-by-1 vector
% h0: a positive scalar
% opt: 'kr' or [] if (local constant) kernel regression, 'krll' if local linear regression

% opth = opt_h(x,y,opt,h0) returns the optimal bandwidth for a multi-dimensional nonparametric regression,
% i.e., kernel regression (the default) or local linear regression using the Quartic kernel.
% It is calculated based on cross-validation

% opth = opt_h(x,y) returns the optimal bandwidth for a kernel regression of y on x.
% opth = opt_h(x,y,opt) returns the optimal bandwidth for a specified (in opt) nonparametric regression of y on x.
% opth = opt_h(x,y,opt,h0) returns returns the optimal bandwidth for a specified (in opt) nonparametric regression
% of y on x, using the provided starting value of the bandwidth.

% Copyright: Yingying Dong at Boston College, July, 2008.

%% ------------------Step 0: Check input and output---------------------
warning('off','all')

error(nargchk(2,4,nargin));
error(nargchk(1,1,nargout));

if nargin < 3
    opt = [];
elseif ~(strcmp(opt, 'kr')||strcmp(opt,'krll')||isempty(opt))
    error('opt must be ''kr'', ''krll'' or [].')
end

%% -------------Step 1: get the starting value for h--------------------
if nargin < 4

  %-----initial grid search--------
    hLin = 0+0.5*(1:10);
    mseFun = zeros(size(hLin));
    for i = 1:length(hLin)
        mseFun(i) = MSE(x,y,opt,hLin(i));
    end
    %     plot(hLin, mseFun,'b'); ylim([0 10]);xlabel('bandwidth'); ylabel('mse');
    h00 = hLin(mseFun==min(mseFun));

    %-----finer grid search----------
    hLin = h00-0.5+0.05*(0:20);
    mseFun = zeros(size(hLin));
    for j = 1:length(hLin)
        mseFun(j) = MSE(x,y,opt,hLin(j));
    end
    %     hold on;
    %     plot(hlin, mseFun, 'r'); ylim([0 10]);xlabel('bandwidth'); ylabel('mse');
    if sum(mseFun==min(mseFun))==1
        h0 = hLin(mseFun==min(mseFun));
    else
        h0 = mean(hLin(mseFun==min(mseFun)));
    end
%fprintf ('\nThe starting value of the bandwidth is %6.4f.', h0);
elseif ~isscalar(h0)
    error('The provided bandwidth must be a scalar.')
end

%% ------------Step 2: find the optimal value for h--------------------

options=optimset('MaxFunEvals', 1000, 'MaxIter', 1000,'TolX',1e-10,'TolFun',1e-10, 'Display', 'off' );
if nargin < 4
    [opth]= fmincon(@(h)MSE(x,y,opt,h),h0, [],[], [],[], h0-0.05, h0+0.05,[],options);
else
    [opth]= fmincon(@(h)MSE(x,y,opt,h), h0, [],[], [],[], 0, 10,[], options);
end

%fprintf ('\nThe optimal bandwidth is %6.4f.', opth);

%% MSE function
function value = MSE(x,y,opt,h)
[n k]= size(x);
b = zeros(k,n-1);
yhat = nan(n,1);

%----calculate the cross validation criterion function------
    for i = 1:n
        % remove ith observation
        xo = x([1:i-1 i+1:end],:);
        yo = y([1:i-1 i+1:end],:);

        % calculate kernel function
        dis = bsxfun(@minus, xo, x(i,:));
        u = bsxfun(@rdivide, dis, std(dis)) / h;
        % Kernel = @(u) 15/16 * (1-u.^2).^2.*(abs(u)<=1);
        Kernel = @(u)1/sqrt(2*pi)*exp(-u.^2/2);

        % calculate weights
        w = prod(Kernel(u),2);
        sw = sum(w);

        % calculate yhat, using kernel or local linear regression
        if strcmp(opt, 'krll')
            t1 = bsxfun(@times, dis, w)';
            t2 = sum(t1, 2);
            b(:,i) = (sw*t1*dis - t2*t2')\(sw*t1*yo - t2*w'*yo);
            yhat(i) = w'*(yo - dis*b(:,i))/sw;
            %if isnan(yhat(i))||isinf(yhat(i))
            %yhat(i)= w'*yo/sw;
            %end
        else
            yhat(i) = w'*yo/sw;
        end
    end
    
flag = ~(yhat~=yhat);
yhat(~flag)= 1e+308;

value = sum((flag.*(y-yhat)).^2)/sum(flag);
value(isnan(value)|isinf(value)) = 1e+308;
