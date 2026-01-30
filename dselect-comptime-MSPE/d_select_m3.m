%%%%%%%%%% Simulation codes for structural dimension selection %%%%%%%%%%
% Proposed seven estimators for Model 7.3 with n=200
% K=5, d_max=5, and s_max=7
% The result is "resmat", a 7-by-6 matrix, whose each row
% is the percentages of d_hat taking vlaues in {0,1,...,5}
% The true d=1

alpha=1.1;
n=200;
H=5;
arg=0:0.01:1;
L=length(arg);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)]; % 50-by-L matrix
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta=PHI'*beta1_coef'; % true slope function

dmat=zeros(7,200);
parfor rep=1:200
    fprintf("%d ", rep);
    %%% Normally distributed Z_k' %%%
    Z=normrnd(0,1,n,50);

    %%% Multivariate t distributed Z_k' %%%
    % z=normrnd(0,1,n,50);
    % tau=chi2rnd(5,[1,n]);
    % Z=z./repmat(sqrt(tau'/(5-2)),1,50);

    %%% Uniformly distributed Z_k' %%%
    % Z=unifrnd(-sqrt(3),sqrt(3),[n,50]);

    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';

    p=4;
    mu=zeros(1,p);
    Y=zeros(n,p);
    for i=1:n
        Sigma=blkdiag([1,Xbeta1(i)/sqrt(1+Xbeta1(i)^2);Xbeta1(i)/sqrt(1+Xbeta1(i)^2),1],0.1*eye(2));
        epsilon=mvnrnd(mu,Sigma,1);
        Y(i,:)=[epsilon(1),epsilon(2:end)];
    end

    CV=KCV_simu(arg,X,Y,1:7,0:5,5,H);
    dmat(:,rep)=CV(:,1);
end

resmat=zeros(7,6);
for d=1:6
    resmat(:,d)=sum(dmat==(d-1),2);
end