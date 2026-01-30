%%%%%%%%%% Simulation codes for computation times %%%%%%%%%%
% Ten estimators of Model 7.4 with normally distributed Z_k' are considered
% 1000 repetitions for each setting
% Output: timevec (in minutes)

n=200; % n=200 or 500

%%% H is the number of slices: H=5 for n=200; H=10 for n=500 %%%
%%% H1 is the number of slices for FMS %%%
H=5;H1=5;
% H=10;H1=5;

p=20; % p=20, 50, or 100

%%%%%%%%%%%%%%%% 1 PFSIR %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    mn=ceil(n^(3/2));
    G=normrnd(0,1,mn,p);
    Lambda_PFSIR=0;
    for j=1:mn
        W=G(j,:)/norm(G(j,:));
        Lambda_PFSIR=Lambda_PFSIR+FSIR(W*Y',X,H);
    end
    Lambda=Lambda_PFSIR/mn;
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime1=toc;


%%%%%%%%%%%%%%%% 2 PFSAVE %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    G=normrnd(0,1,mn,p);
    Lambda_PFSAVE=0;
    for j=1:mn
        W=G(j,:)/norm(G(j,:));
        Lambda_PFSAVE=Lambda_PFSAVE+FSAVE(W*Y',X,arg,H);
    end
    Lambda=Lambda_PFSAVE/mn;
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime2=toc;


%%%%%%%%%%%%%%%% 3 PFCS %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    Lad_PFCS=0;
    for i=1:(n-1)
        for j=(i+1):n
            index=setdiff(1:n,[i,j]);
            Yij=Y(index,:);
            angdist_ij=sum(Ang(Y(i,:)-Yij,Y(j,:)-Yij))/(pi*n);
            Lad_PFCS=Lad_PFCS+angdist_ij*X_cen(i,:)'*X_cen(j,:);
        end
    end
    Lambda=-(Lad_PFCS+Lad_PFCS')/(2*n^2);
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime3=toc;


%%%%%%%%%%%%%%%% 4 Dist-l2 %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;

    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    Lad_Sq=0;
    for i=1:(n-1)
        for j=(i+1):n
            dist_ij=norm(Y(i,:)-Y(j,:));
            Lad_Sq=Lad_Sq+dist_ij*X_cen(i,:)'*X_cen(j,:);
        end
    end
    Lambda=-(Lad_Sq+Lad_Sq')/(n^2);
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime4=toc;



%%%%%%%%%%%%%%%% 5 Dist-l1 %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    Lad_Abs=0;
    for i=1:(n-1)
        for j=(i+1):n
            Lad_Abs=Lad_Abs+norm(Y(i,:)-Y(j,:),1)*X_cen(i,:)'*X_cen(j,:);
        end
    end
    Lambda=-(Lad_Abs+Lad_Abs')/(n^2);
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime5=toc;



%%%%%%%%%%%%%%%% 6 Dist-GK %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    sigma=median(pdist(Y));
    Lad_GK=0;
    for i=1:(n-1)
        for j=(i+1):n
            dist_ij=norm(Y(i,:)-Y(j,:));
            Lad_GK=Lad_GK+(2-2*exp(-dist_ij^2/(2*sigma^2)))*X_cen(i,:)'*X_cen(j,:);
        end
    end
    Lambda=-(Lad_GK+Lad_GK')/(n^2);
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime6=toc;



%%%%%%%%%%%%%%%% 7 Dist-LK %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    sigma=median(pdist(Y));
    Lad_LK=0;
    for i=1:(n-1)
        for j=(i+1):n
            dist_ij=norm(Y(i,:)-Y(j,:));
            Lad_LK=Lad_LK+(2-2*exp(-dist_ij/sigma))*X_cen(i,:)'*X_cen(j,:);
        end
    end
    Lambda=-(Lad_LK+Lad_LK')/(n^2);
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime7=toc;



%%%%%%%%%%%%%%%% 8 FKIR %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    KImat=FKIR(Y,X,H);
    for sn=1:length(svec)
        s=svec(sn);
        xcov_inv_sq=eigfun(:,1:s)*diag(kappa(1:s).^(-1/2))*eigfun(:,1:s)';
        Gamma1=xcov_inv_sq*TR*KImat*TR*xcov_inv_sq;
        Gamma=(Gamma1+Gamma1')/2;
        [eigen,eigv]=eigs(Gamma,d,'lm');
        beta_FKIR=xcov_inv_sq*eigen;
        Ps_FKIR=proj(beta_FKIR,arg);
        MSE1(rep,sn)=trapz2((Ps_FKIR-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_FKIR-Ps)*TR*(Ps_FKIR-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime8=toc;



%%%%%%%%%%%%%%%% 9 FMS %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    Lambda=MS_FSIR(Y,X,H1);
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime9=toc;



%%%%%%%%%%%%%%%% 10 FMC %%%%%%%%%%%%%%%%
tic;
alpha=1.1;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)];
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef'];
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec));
MSE2=zeros(1000,length(svec));
for rep=1:1000
    Z=normrnd(0,1,n,50);
    eigvalsq=(-1).^(2:51).*(1:50).^(-alpha/2);
    score=Z.*repmat(eigvalsq,n,1);
    X=score*PHI;
    Xbeta1=(beta1_coef.*eigvalsq)*Z';
    Xbeta2=(beta2_coef.*eigvalsq)*Z';

    mu=zeros(1,p);
    Sigma=0.5.^(toeplitz(0:(p-1)));
    epsilon=0.1*mvnrnd(mu,Sigma,n);
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta2')),9./(1+exp(Xbeta1')),zeros(n,p-3)]+epsilon;
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    Lambda_FMC=0;
    for j=1:p
        Lambda_FMC=Lambda_FMC+FSIR(Y(:,j)',X,H);
    end
    Lambda=Lambda_FMC/p;
    
    [~,psi_est]=inteq(Lambda,arg,d);
    for sn=1:length(svec)
        beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
        Ps_est=proj(beta_est,arg);
        MSE1(rep,sn)=trapz2((Ps_est-Ps).^2,arg,arg);
        MSE2(rep,sn)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
    end
end

MSEopt=zeros(1,6);
[min_mse1,pos1]=min(mean(MSE1));
MSEopt(1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1))];
[min_mse2,pos2]=min(mean(MSE2));
MSEopt(4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2))];
etime10=toc;

timevec=[etime1,etime2,etime3,etime4,etime5,...
    etime6,etime7,etime8,etime9,etime10]/60