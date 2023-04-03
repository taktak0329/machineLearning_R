#problem5

set.seed(1)

#1)
X <- matrix(rnorm(200*2),200,2)

#2)
Euclid <- matrix(0,200,1)
for(i in 1:nrow(X)){
  Euclid[i,] <- sqrt(sum((X[i,])^2))
}

#3)
X <- data.frame(cbind(X,Euclid))
colnames(X) <- c('x1','x2','distance')
for(i in 1:nrow(X)){
  if(X$distance[i]>1){
    X[i,1:2] <- X[i,1:2]*5
  }
}

#4)
library(caret)
st_mod <- preProcess(X[,1:2],method=c('center','scale'))
X_st <- predict(st_mod,X[,1:2])

#plotting
plot(X_st$x2~X_st$x1,xlab='x1',ylab='x2')

#k-means
data <- X[,1:2]
colours <- c(rep('red',100),rep('blue',100))
plot(data,col=colours)
km_2 <- kmeans(data,centers=2,nstart=1000)
km_4 <- kmeans(data,centers=4,nstart=1000)
km_6 <- kmeans(data,centers=6,nstart=1000)
par(mfrow = c(2, 2))
plot(data,col=km_2$cluster)
plot(data,col=km_4$cluster)
plot(data,col=km_6$cluster)

