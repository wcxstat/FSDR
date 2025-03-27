alpha=1.1;
n=500;
H=10;
arg=0:0.01:1;
h=arg(2)-arg(1);
L=length(arg);
TR=diag([0.5,ones([1,L-2]),0.5]*h);
PHI=[ones(1,L);sqrt(2)*cos((1:49)'*arg*pi)]; % 50-by-L matrix
beta1_coef=4*(-1).^(2:51).*(1:50).^(-2);
beta2_coef=4*(1:50).^(-2);
beta=PHI'*[beta1_coef',beta2_coef']; % true slope function
% beta=PHI'*beta1_coef';
d=size(beta,2);
Ps=proj(beta,arg);

svec=d:(d+5);
MSE1=zeros(1000,length(svec),8);
MSE2=zeros(1000,length(svec),8);
for rep=1:1000
    Lambda=zeros(L,L,7);
    % Z=normrnd(0,1,n,50);
    z=normrnd(0,1,n,50);
    tau=chi2rnd(5,[1,n]);
    Z=z./repmat(sqrt(tau'/(5-2)),1,50);
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
    % epsilon=0.1*mvtrnd(Sigma,3,n);
    % epsilon=0;
    % Y=[Xbeta1',Xbeta2',zeros(n,2)]+epsilon;
    % Y=sin(Xbeta1')+Xbeta1'+epsilon;
    % Y=[sin(Xbeta1')+(Xbeta1').^1,(Xbeta1').^3+Xbeta2',zeros(n,2)]+epsilon;
    % Y=[Xbeta1'./(10+(Xbeta2'+1).^2),Xbeta2'./(12+(Xbeta1'+1).^2),zeros(n,2)]+epsilon;
    % Y=Xbeta1'./(10+(Xbeta2'+1).^2)+0.01*normrnd(0,1,[n,1]);
    % Y=[10*Xbeta1'./(10+(Xbeta2'+1).^2),10./(4+(Xbeta1'+0.5).^2),zeros(n,98)]+epsilon;
    Y=[10*Xbeta1'./(9+(Xbeta2'+1).^2),10./(4+sin(Xbeta1')),zeros(n,2)]+epsilon;
    % Y=[Xbeta1'+epsilon(:,1),sin(Xbeta1')+exp(Xbeta2').*epsilon(:,2),epsilon(:,3:4)];
    % Y=Xbeta1'+exp(Xbeta1').*normrnd(0,1,[n,1]);
    % Y=zeros(n,p);
    % for i=1:n
    %     Sigma=blkdiag([1,1/(1+exp(Xbeta1(i)));1/(1+exp(Xbeta1(i))),1],eye(2));
    %     epsilon=mvnrnd(mu,Sigma,1);
    %     Y(i,:)=[epsilon(1),epsilon(2:end)];
    % end

    X_cen=X-mean(X,1); %center
    xcov=X_cen'*X_cen/n; % covariance
    no_pca=15;
    [kappa,eigfun,xi_est]=FPCA_bal(X,arg,no_pca);

    %%%%%%%%%%%%%%%% PFSIR %%%%%%%%%%%%%%%%
    % mn=2*ceil(n*log(n));
    mn=ceil(n^(3/2));
    G=normrnd(0,1,mn,p);
    Lambda_PFSIR=0;
    for j=1:mn
        W=G(j,:)/norm(G(j,:));
        Lambda_PFSIR=Lambda_PFSIR+FSIR(W*Y',X,H);
    end
    Lambda(:,:,1)=Lambda_PFSIR/mn;

    %%%%%%%%%%%%%%%% PFSAVE %%%%%%%%%%%%%%%%
    G=normrnd(0,1,mn,p);
    Lambda_PFSAVE=0;
    for j=1:mn
        W=G(j,:)/norm(G(j,:));
        Lambda_PFSAVE=Lambda_PFSAVE+FSAVE(W*Y',X,arg,H);
    end
    Lambda(:,:,2)=Lambda_PFSAVE/mn;

    %%%%%%%%%%%%%%%% PFCS and Distance %%%%%%%%%%%%%%%%
    sigma=median(pdist(Y));
    Xc=X-mean(X);
    Lad_PFCS=0;
    Lad_Sq=0;Lad_Abs=0;Lad_GK=0;Lad_LK=0;
    for i=1:(n-1)
        for j=(i+1):n
            index=setdiff(1:n,[i,j]);
            Yij=Y(index,:);
            angdist_ij=sum(Ang(Y(i,:)-Yij,Y(j,:)-Yij))/(pi*n);
            Lad_PFCS=Lad_PFCS+angdist_ij*Xc(i,:)'*Xc(j,:);
            %
            dist_ij=norm(Y(i,:)-Y(j,:));
            Lad_Sq=Lad_Sq+dist_ij*Xc(i,:)'*Xc(j,:);
            Lad_Abs=Lad_Abs+norm(Y(i,:)-Y(j,:),1)*Xc(i,:)'*Xc(j,:);
            Lad_GK=Lad_GK+(2-2*exp(-dist_ij^2/(2*sigma^2)))*Xc(i,:)'*Xc(j,:);
            Lad_LK=Lad_LK+(2-2*exp(-dist_ij/sigma))*Xc(i,:)'*Xc(j,:);
        end
    end
    Lambda(:,:,3)=-(Lad_PFCS+Lad_PFCS')/(2*n^2);
    Lambda(:,:,4)=-(Lad_Sq+Lad_Sq')/(n^2);
    Lambda(:,:,5)=-(Lad_Abs+Lad_Abs')/(n^2);
    Lambda(:,:,6)=-(Lad_GK+Lad_GK')/(n^2);
    Lambda(:,:,7)=-(Lad_LK+Lad_LK')/(n^2);
    for k=1:7
        [~,psi_est]=inteq(Lambda(:,:,k),arg,d);
        for sn=1:length(svec)
            beta_est=linv(arg,kappa,eigfun,psi_est,svec(sn));
            Ps_est=proj(beta_est,arg);
            MSE1(rep,sn,k)=trapz2((Ps_est-Ps).^2,arg,arg);
            MSE2(rep,sn,k)=inteq((Ps_est-Ps)*TR*(Ps_est-Ps),arg,1);
        end
    end

    %%%%%%%%%%%%%%%% FKIR %%%%%%%%%%%%%%%%
    KImat=FKIR(Y,X,H);
    for sn=1:length(svec)
        s=svec(sn);
        xcov_inv_sq=eigfun(:,1:s)*diag(kappa(1:s).^(-1/2))*eigfun(:,1:s)';
        Gamma1=xcov_inv_sq*TR*KImat*TR*xcov_inv_sq;
        Gamma=(Gamma1+Gamma1')/2;
        [eigen,eigv]=eigs(Gamma,d,'lm');
        beta_FKIR=xcov_inv_sq*eigen;
        % phi=linv(arg,kappa.^(1/2),eigfun,eigen(:,1:d),sn);
        Ps_FKIR=proj(beta_FKIR,arg);
        MSE1(rep,sn,8)=trapz2((Ps_FKIR-Ps).^2,arg,arg);
        MSE2(rep,sn,8)=inteq((Ps_FKIR-Ps)*TR*(Ps_FKIR-Ps),arg,1);
    end
end

MSEopt=zeros(8,6);
for k=1:8
    [min_mse1,pos1]=min(mean(MSE1(:,:,k)));
    MSEopt(k,1:3)=[svec(pos1),min_mse1,std(MSE1(:,pos1,k))];
    [min_mse2,pos2]=min(mean(MSE2(:,:,k)));
    MSEopt(k,4:6)=[svec(pos2),min_mse2,std(MSE2(:,pos2,k))];
end