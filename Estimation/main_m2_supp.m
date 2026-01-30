%%%%%%%%%%%% Simulation codes for Model 7.2 %%%%%%%%%%%%
%%%%%%%%%%%%      Last two estimators       %%%%%%%%%%%%

alpha=1.1;
n=200;
% n=500;

%%% H is the number of slices: H=5 for n=200; H=10 for n=500 %%%
%%% H1 is the number of slices for FMS %%%
H=5;H1=5;
% H=10;H1=5;

arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)]; % 50-by-L matrix
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef']; % two true slope functions
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec),2);
MSE2=zeros(1000,length(svec),2);
for rep=1:1000
    Lambda=zeros(L,L,2);
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
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    p=4;
    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta1')),zeros(n,2)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    %%%%%%%%%%%%%%%% FMS %%%%%%%%%%%%%%%%
    
    Lambda(:,:,1)=MS_FSIR(Y,X,H1);
    
    %%%%%%%%%%%%%%%% FMC %%%%%%%%%%%%%%%%
    
    Lambda_FMC=0;
    for j=1:p
        Lambda_FMC=Lambda_FMC+FSIR(Y(:,j)',X,H);
    end
    Lambda(:,:,2)=Lambda_FMC/p;
    
    for k=1:2
        [~,psi_est]=inteq(Lambda(:,:,k),arg,d);
        for sn=1:length(svec)
            beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
            Ps_est=proj(beta_est,arg);
            MSE1(rep,sn,k)=trapz2((Ps_est-Ps).^2,arg,arg);
            MSE2(rep,sn,k)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
        end
    end
end

MSEopt=zeros(2,6);
for k=1:2
    [min_mse1,pos1]=min(mean(MSE1(:,:,k)));
    MSEopt(k,1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1,k))];
    [min_mse2,pos2]=min(mean(MSE2(:,:,k)));
    MSEopt(k,4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2,k))];
end