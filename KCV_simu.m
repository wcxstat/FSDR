function CV=KCV_simu(arg,X,Y,svec,dvec,K,H)
% Using K-fold cross-validation to select the structural dimension d
% Proposed seven estimators

% Input
% svec: candidate truncation parameters
% dvec: candidate structural dimensions
% H: number of slices for PFSIR and PFSAVE

% Output
% CV, where d_hat=CV(:,1), s_hat=CV(:,2), and minCV=CV(:,3)

% Copyright: Wenchao Xu at SUIBE, Sept., 2025.

mn=2000;
L=length(arg);
[n,p]=size(Y);
M=floor(n/K);
MSPE=10000*ones(length(svec),length(dvec),K,7);
for kk=1:K
    if kk==K
        index=((K-1)*M+1):n;
    else
        index=((kk-1)*M+1):(kk*M);
    end
    index1=setdiff(1:n,index);
    X_train=X(index1,:);X_test=X(index,:);
    Y_train=Y(index1,:);Y_test=Y(index,:);
    n_train=size(Y_train,1);n_test=size(Y_test,1);
    
    no_pca=15;
    [kappa,eigfun,~]=FPCA_bal(X_train,arg,no_pca);
    Lambda=zeros(L,L,7);
    
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
    
    for k=1:7
        for d1=2:length(dvec)
            d=dvec(d1);
            [~,psi_est]=inteq(Lambda(:,:,k),arg,d);
            for sn=d:length(svec)
                beta_est=linv(arg,kappa,eigfun,psi_est,sn);
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
                MSPE(sn,d1,kk,k)=mean(sum((Y_test-ypred).^2,2));
            end
        end
        MSPE(:,1,kk,k)=repmat(mean(sum((Y_test-mean(Y_train)).^2,2)),length(svec),1);
    end
end

CV=zeros(7,3);
for k=1:7
    Mt=mean(MSPE(:,:,:,k),3);
    minValue=min(Mt(:));
    [rr,cc]=find(Mt==minValue);
    CV(k,:)=[cc(1)-1,rr(1),minValue];
end