function bhat=linv(arg,kappa,eigfun,psi,sn)
% This is a program to calculate Gamma^(-1)*g

d=size(psi,2);
bhat=zeros(length(arg),d);
for k=1:d
    g=trapz(arg,repmat(psi(:,k),1,sn).*eigfun(:,1:sn));
    bhat(:,k)=eigfun(:,1:sn)*(g./kappa(1:sn))';
end