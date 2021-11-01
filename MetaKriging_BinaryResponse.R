library(spBayes)
library(MBA)
library(mcmc)
library(MASS)
library(MCMCpack)
library(parallel)
library(doParallel)
library(foreach)


#################################################################
##### Partition of Spatial data
#################################################################

n.sample <- n  ## Sample size
n.core <- 20  ## Number of Clusters  
per.core <- floor(n.sample/n.core)   ## Observations per cluster
n.part <- c(rep(per.core,n.core-1),n.sample-per.core*(n.core-1)) ### Vector of number of observations per cluster

a <- 1:n.sample 
index.part <- list() 
X1.part <- list()
Y1.part <- list()
X2.part <- list()
Y2.part <- list()
coords.part <- list()


for(i in 1:length(n.part )){
  beg<-Sys.time()
  index.part[[i]] <- sample(a,n.part[i],replace=FALSE) 
  ## Response in i th subset 
  Y1.part[[i]] <- y.1[index.part[[i]]]  
  Y2.part[[i]] <- y.2[index.part[[i]]]  
  ## Predictor in i th subset 
  X1.part[[i]]<- x.1[index.part[[i]],]
  X2.part[[i]]<- x.2[index.part[[i]],]
  ## Coordinates in i th subset       
  coords.part[[i]] <- coords[index.part[[i]],]  
  a <- setdiff(a,index.part[[i]])
}




partitioned_spMvGLM <- function(i, y, x, weight, n, q, n.part, coords.test, x.test){
  ## Model fitting
  
  #source("spMvGLM_DC.R")
  library(spBayes)
  
  #fit <- glm((y/weight)~x-1, weights=rep(10*weight, n*q), family="binomial")
  fit <- glm((y/weight)~x-1, weights=rep(weight, n*q), family="binomial")
  beta.starting <- coefficients(fit)
  beta.tuning <- t(chol(vcov(fit)))
  A.starting <- diag(1,q)[lower.tri(diag(1,q), TRUE)]
  n.batch <- 100
  batch.length <- 50
  n.samples <- n.batch*batch.length
  starting <- list("beta"=beta.starting, "phi"=rep(3/0.5,q), "A"=A.starting, "w"=0)
  tuning <- list("beta"=beta.tuning, "phi"=rep(1,q), "A"=rep(0.1,length(A.starting)),
                 "w"=0.5)
  priors <- list("beta.Flat", "phi.Unif"=list(rep(3/0.75,q), rep(3/0.25,q)),
   "K.IW"=list(q, diag(0.1,q)))
  

  YY1 <- Y1.part[[i]]
  YY2 <- Y2.part[[i]]

  
  ## Predictor in subset i    
  XX1 <- X1.part[[i]] 
  XX2 <- X2.part[[i]] 
  ## Coordinates in subset i      
  CC <- coords.part[[i]]  
  
  
  ## GP computation in each subset
  
  m.1 <- spMvGLM(formula=list(YY1~XX1-1, YY2~XX2-1),
                 coords=CC, weights= matrix(weight,n.part[i],q),
                 starting=starting, tuning=tuning, priors=priors,
                 amcmc=list("n.batch"=n.batch,"batch.length"=batch.length,"accept.rate"=0.43),
                 cov.model="exponential", n.report=10)  
  burn.in <- 0.75*n.samples
  # sub.samps <- burn.in:n.samples
  m.s.pred<- spPredict(m.1, coords.test, x.test, start=burn.in, end=n.samples)  
  allquant<- function(x){return(quantile(x, probs= seq(0.005, 1, 0.005) ))}
  param<-list("parameters"=apply(window(m.1$p.beta.theta.samples, start=burn.in), 2, allquant), "w.predict"= apply(m.s.pred$p.w.predictive.samples,1,allquant))

  ## Garbage cleaning  
  gc()          
  
  
  return(param)
}



####### Parallelization ########
## Number of clusters for parallel implementation
cl<-makeCluster(n.core)  
registerDoParallel(cl)

## Start time
strt<-Sys.time()
## Parallelized subset computation of GP in different cores
obj <- foreach(i=1:n.core) %dopar% partitioned_spMvGLM(i, y, x, weight, n, q, n.part, coords.test, x.test)  
## Total time for parallelized inference

final.time <- Sys.time()-strt  
#stopCluster(cl)

on.exit(stopCluster(cl))




#############################
####  Combining Results
#############################

result<- obj[[1]]$parameters
for(k in 2: length(n.part)){
  result<- result +  obj[[k]]$parameters
}
result<- result/length(n.part)

result2<- obj[[1]]$w.predict
for(k in 2: length(n.part)){
  result2<- result2 +  obj[[k]]$w.predict
}
result2<- result2/length(n.part)


####################################################
### Sampling from the parameter distribution
####################################################
samplesize<- 1000
Xout<-seq(0.005,1,0.001)
sampleparIndex<- sample(seq(1, length(Xout), 1), samplesize, replace=TRUE)
funInterpo<- function(x){return(approx(seq(0.005, 1, 0.005),x,xout=Xout)$y)}
Result.Inter<- apply(result, 2, funInterpo)
Result.Inter2<- apply(result2, 2, funInterpo)
SamplePar<- Result.Inter[sampleparIndex,]
Samplew<- Result.Inter2[sampleparIndex,]
####"traceplots"
plot(SamplePar[,1], type="l")
plot(Samplew[,1], type="l")



##########################################################
###### Prediction (Calculation of p(y=1|...))

p.sample<- array(NA, dim=c(samplesize, q*(n.extra-n)))
for(j in 1:samplesize)
{
  B.s<- SamplePar[j,1:4]
  p.sample[j,] <- 1/(1+exp(-(x.test%*%B.s+Samplew[j,])))
}

quant.pred<- function(x){return(quantile(x, probs=c( 0.5, 0.025,0.975)))}
w.quant<-apply(Samplew, 2,quant.pred)
param.quant<-apply(SamplePar, 2, quant.pred)
















