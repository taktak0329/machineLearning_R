#problem3

companies_price <- read.csv('companies-price.csv',header = F)
companies_date <- read.csv('companies-dates.csv',header = F)
companies_info <- read.csv('companies-info.csv',header = F)
emergency_data <- read.csv('emergency-data.csv',header = F)


#Getting the missing data





#Network analysis
#you may use functions we used in the slides and problem sets.
#once you have assembled the complete data set of the time series,
#your task is below

#assemble data and input emergency data
companies_price[,1:10] <- emergency_data
rownames(companies_price) <- companies_date[,1]
colnames(companies_price) <- companies_info[,1]

#a)
n <- ncol(companies_price)#num of companies
t <- nrow(companies_price)#num of terms

logret <- data.frame(matrix(0,nrow=t-1,ncol=n))
for(i in 1:n){
  for(j in 1:t-1){
    logret[j,i] = 100*(log((companies_price[j+1,i])-log(companies_price[j,i]))/log(companies_price[j,i]))
  }
}

#b)
market_fac <- rowMeans(logret)
logret_res <- matrix(0,t-1,n)
for(i in 1:n){
  regmodel <- lm(logret[,i] ~ market_fac)
  logret_res[,i] <- regmodel$resid
}
corr1 <- cor(logret)#without filtering
corr2 <- cor(logret_res)#with filtering
heatmap(corr1,Rowv=NA,Colv=NA,scale='none')
heatmap(corr2,Rowv=NA,Colv=NA,scale='none')

#c)
lambda.range <- seq(0.1,1,0.1)
bic_vec <- rep(0,length(lambda.range))
p_hat_list <- list()

#sourcing in the relevant functions to help find the BIC
source('bic_space.R')
source('hush.R')
Y = logret_res

library(space)

for(i in 1:length(lambda.range)){
  a <- hush(space.joint(Y,lam1=lambda.range[i]*(t-1)))
  bic_vec[i] <- bic_space(a)
  p_hat_list[[i]] <- a$ParCor
}

lambda_opt_ind <- which(bic_vec == min(bic_vec))
lambda.range[lambda_opt_ind]

P_hat <- p_hat_list[[lambda_opt_ind]]

#d)
library(igraph)
A_hat <- 1*(P_hat != 0) - diag(rep(1,n))
net_hat <- graph.adjacency(A_hat,mode='undirected')

deg <- degree(net_hat)
hist(deg,30,col='blue',main='',xlab='degrees')


#e)
#fiding the laplacian matrix
A <- as.matrix(A_hat)
D <- as.matrix(diag(deg))
L <- D-A
L_n <- solve(D^(1/2))%*% L %*% solve(D^(1/2))
plot(eigen(L_n)$val)
evec <- eigen(L_n)$vec

K <- 3
U <- evec[,(n-K+1):n]
N <- diag(1/(sqrt(rowSums(U^2))))
X <- N %*% U
ind_comm <- kmeans(X,K)$cluster
plot(graph.adjacency(A,mode='Undirected'),vertex.label.cex=0.75,
                     vertex.size=5,vertex.label.dist=1,
                     edge.arrow.size=0.1)

#f)










