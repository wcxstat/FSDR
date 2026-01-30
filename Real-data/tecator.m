%%%%%%% Codes for Tecator dataset analysis with three-dim response %%%%%%%
% The dataset consists of 215 meat samples. The first 172 samples are used
% as training set, and the remaining 43 samples are used as the test set
% Ten estimators are considered for a comparison
% We use five-fold CV to select (d, sn), where d_max=7 and s_max=15
% We calculate the MSPE on the test set


absorp=importdata('absorp.txt');
covarite=importdata('y.txt');
X=absorp.data(:,26:89);
U=covarite.data/100; % fat, water, and protein content
Y=log(U./(1-U));
[n,p]=size(Y);
L=size(X,2);
arg=linspace(902,1028,64);
h=arg(2)-arg(1);
TR=diag([0.5,ones([1,L-2]),0.5]*h);

index_train=1:172;
index_test=setdiff(1:n,index_train);
X_train=X(index_train,:);X_test=X(index_test,:);
Y_train=Y(index_train,:);Y_test=Y(index_test,:);
n_train=size(Y_train,1);n_test=size(Y_test,1);

H=5;H1=5;
mn=2000;
CV=KCV(arg,X_train,Y_train,1:15,0:7,5,H);

no_pca=15;
[kappa,eigfun,xi_est]=FPCA_bal(X_train,arg,no_pca);
Lambda=zeros(L,L,9);
MSPE=zeros(1,10);
 
%%%%%%%%%%%%%%%% PFSIR %%%%%%%%%%%%%%%%
G=normrnd(0,1,mn,p);
Lambda_PFSIR=0;
for j=1:mn
    W=G(j,:)/norm(G(j,:));
    Lambda_PFSIR=Lambda_PFSIR+FSIR(W*Y_train',X_train,H);
end
Lambda(:,:,1)=Lambda_PFSIR/mn;
 
%%%%%%%%%%%%%%%% PFSAVE %%%%%%%%%%%%%%%%
G=normrnd(0,1,mn,p);
Lambda_PFSAVE=0;
for j=1:mn
    W=G(j,:)/norm(G(j,:));
    Lambda_PFSAVE=Lambda_PFSAVE+FSAVE(W*Y_train',X_train,arg,H);
end
Lambda(:,:,2)=Lambda_PFSAVE/mn;
 
%%%%%%%%%%%%%%%% PFCS and Distance %%%%%%%%%%%%%%%%
sigma=median(pdist(Y_train));
Xc=X_train-mean(X_train);
Lad_PFCS=0;
Lad_Sq=0;Lad_Abs=0;Lad_GK=0;Lad_LK=0;
for i=1:(n_train-1)
    for j=(i+1):n_train
        index=setdiff(1:n_train,[i,j]);
        Yij=Y_train(index,:);
        angdist_ij=sum(Ang(Y_train(i,:)-Yij,Y_train(j,:)-Yij))/(pi*n_train);
        Lad_PFCS=Lad_PFCS+angdist_ij*Xc(i,:)'*Xc(j,:);
        
        dist_ij=norm(Y_train(i,:)-Y_train(j,:));
        Lad_Sq=Lad_Sq+dist_ij*Xc(i,:)'*Xc(j,:);
        Lad_Abs=Lad_Abs+norm(Y_train(i,:)-Y_train(j,:),1)*Xc(i,:)'*Xc(j,:);
        Lad_GK=Lad_GK+(2-2*exp(-dist_ij^2/(2*sigma^2)))*Xc(i,:)'*Xc(j,:);
        Lad_LK=Lad_LK+(2-2*exp(-dist_ij/sigma))*Xc(i,:)'*Xc(j,:);
    end
end
Lambda(:,:,3)=-(Lad_PFCS+Lad_PFCS')/(2*n_train^2);
Lambda(:,:,4)=-(Lad_Sq+Lad_Sq')/(n_train^2);
Lambda(:,:,5)=-(Lad_Abs+Lad_Abs')/(n_train^2);
Lambda(:,:,6)=-(Lad_GK+Lad_GK')/(n_train^2);
Lambda(:,:,7)=-(Lad_LK+Lad_LK')/(n_train^2);

%%%%%%%%%%%%%%%% FMS %%%%%%%%%%%%%%%%
Lambda(:,:,8)=MS_FSIR(Y_train,X_train,H1);

%%%%%%%%%%%%%%%% FMC %%%%%%%%%%%%%%%%
Lambda_FMC=0;
for j=1:p
    Lambda_FMC=Lambda_FMC+FSIR(Y_train(:,j)',X_train,H);
end
Lambda(:,:,9)=Lambda_FMC/p;

beta_d=cell(1,10);
for k=1:9
    if k==8||k==9
        d=CV(k+1,1);sn=CV(k+1,2);
    else
        d=CV(k,1);sn=CV(k,2);
    end
    [~,psi_est]=inteq(Lambda(:,:,k),arg,d);
    beta_est=linv(arg,kappa,eigfun,psi_est,sn);
    if k==8||k==9
        beta_d{1,k+1}=beta_est;
    else
        beta_d{1,k}=beta_est;
    end
    bX_train=zeros(n_train,d);
    bX_test=zeros(n_test,d);
    for dd=1:d
        bX_train(:,dd)=trapz(arg,X_train'.*repmat(beta_est(:,dd),1,n_train))';
        bX_test(:,dd)=trapz(arg,X_test'.*repmat(beta_est(:,dd),1,n_test))';
    end
    ypred=zeros(n_test,p);
    for j=1:p
        h1=opt_h(bX_train,Y_train(:,j),'kr');
        ypred(:,j)=ks(bX_train,Y_train(:,j),'kr',h1,bX_test);
    end
    if k==8||k==9
        MSPE(k+1)=mean(sum((Y_test-ypred).^2,2));
    else
        MSPE(k)=mean(sum((Y_test-ypred).^2,2));
    end
end
 
%%%%%%%%%%%%%%%% FKIR %%%%%%%%%%%%%%%%
KImat=FKIR(Y_train,X_train,H);
d=CV(8,1);sn=CV(8,2);
xcov_inv_sq=eigfun(:,1:sn)*diag(kappa(1:sn).^(-1/2))*eigfun(:,1:sn)';
Gamma1=xcov_inv_sq*TR*KImat*TR*xcov_inv_sq;
Gamma=(Gamma1+Gamma1')/2;
[eigen,eigv]=eigs(Gamma,d,'lm');
beta_FKIR=xcov_inv_sq*eigen;
beta_d{1,8}=beta_FKIR;
bX_train=zeros(n_train,d);
bX_test=zeros(n_test,d);
for dd=1:d
    bX_train(:,dd)=trapz(arg,X_train'.*repmat(beta_FKIR(:,dd),1,n_train))';
    bX_test(:,dd)=trapz(arg,X_test'.*repmat(beta_FKIR(:,dd),1,n_test))';
end
ypred=zeros(n_test,p);
for j=1:p
    h1=opt_h(bX_train,Y_train(:,j),'kr');
    ypred(:,j)=ks(bX_train,Y_train(:,j),'kr',h1,bX_test);
end
MSPE(8)=mean(sum((Y_test-ypred).^2,2));

[CV, MSPE']
save('regfun.mat','beta_d');