#problem4

# Fixing the random number generator seed:
set.seed(1)

# Simulation parameters:
n      <- 50
R      <- 1000
n_p    <- 100
p_list <- seq(5, 45, 5)

################################################################################
# Functions:                                                                   #
################################################################################
basic_ols <- function(Y, X){
  beta_hat <- solve(t(X) %*% X) %*% (t(X) %*% Y)
  return(beta_hat)
}

lin_pred <- function(beta_hat, X_n){
  Y_hat <- X_n %*% beta_hat
  return(Y_hat)
}

mse <- function(Y_hat, Y_n){
  mse <- mean( (Y_hat - Y_n)^2 )
  return(mse)
}
##############################################################


#storage
mse_storage  <- matrix(0, R, length(p_list))
var_storage <- matrix(0,R,length(p_list))
bias_storage <- matrix(0,R,length(p_list))

#personal arrange
col_names <- c(5,10,15,20,25,30,35,40,45)
colnames(mse_storage) <- col_names
colnames(var_storage) <- col_names
colnames(bias_storage) <- col_names


#loop
for(i in 1:length(p_list)){
  
  p <- p_list[i]
  
  for(r in 1:R){
    
    #data preparing
    beta <- runif(p + 1, 0, 5)
    eps <- rnorm(n)
    x <- cbind( rep(1,n),matrix( rnorm(n*p),n,p) )
    y <- x %*% beta + eps
    
    #new observation for prediction(test)
    eps_n <- rnorm(n_p)
    x_n   <- cbind( rep(1, n_p),matrix(rnorm(n_p*p), n_p,p) )
    y_n   <- x_n%*%beta +eps_n
    
    #beta estimation
    beta_hat <- as.numeric(basic_ols(y,x))
    
    #prediction for new observation
    y_hat <- lin_pred(beta_hat,x_n)
    
    #calculate and store MSE:
    mse_storage[r,i] <- mse(y_hat,y_n)
    
    #calculate and store var
    var_storage[r,i] <- mean((x_n*beta - x_n*beta_hat)^2)
    
    #calculate and store bias
    bias_storage[r,i] <- (mean(x_n*beta - x_n*beta_hat))^2
    
  }
}

#plotting mean of each p of three quantities
plot(colMeans(mse_storage)~col_names,type = 'l',xlab = 'number of p',ylab = 'mean of mse')
plot(colMeans(var_storage)~col_names,type = 'l',xlab = 'number of p',ylab = 'mean of variance')
plot(colMeans(bias_storage)~col_names,type = 'l',xlab = 'number of p',ylab = 'mean of bias')
