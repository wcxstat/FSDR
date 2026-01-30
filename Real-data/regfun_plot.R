library(R.matlab)

betahat<-readMat('regfun.mat')
beta1<-betahat$beta.d
arg<-seq(902,1028,by=2)

pdf(file='regfun.pdf',width=10,height=3.3)
par(mfrow=c(1,3))
par(mai=c(0.4,0.42,0.3,0.1),mgp=c(1.8,0.6,0))
# PFSIR
beta2<-beta1[[1]][[1]]
matplot(arg,beta2,type="l",col=c("black","red","blue","green2"),lty=c(1,2,4,5),
        lwd=1.5,xlab="Wavelength (nm)",ylab=expression(widehat(beta)),
        main="PFSIR")

#PFCS
beta2<-beta1[[3]][[1]]
matplot(arg,beta2,type="l",col=c("black","red","blue"),lty=c(1,2,4),
        lwd=1.5,xlab="Wavelength (nm)",ylab=expression(widehat(beta)),
        main="PFCS")

#Dist-GK
beta2<-beta1[[6]][[1]]
matplot(arg,beta2,type="l",col=c("black","red","blue"),lty=c(1,2,4),
        lwd=1.5,xlab="Wavelength (nm)",ylab=expression(widehat(beta)),
        main="Dist-GK")

par(mfrow=c(1,1))
dev.off()