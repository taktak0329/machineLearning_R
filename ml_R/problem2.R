#problem2
set.seed(1)
data_bank <- read.csv('UniversalBank.csv')

#the bank is interested in being able to predict 
#whether a customer will accept a personal loan offer
################################################################################
# data_bank
################################################################################
#  dim = 5000,14
#  cols = "ID" "Age" "Experience" "Income" "ZIP.Code"
#         "Family" "CCAvg" "Education" "Mortgage"
#         "Personal.Loan" "Securities.Account" "CD.Account"
#         "Online" "CreditCard" 
##################################################################################
# Variable             Description
# ID                   Customer ID
# Age                  Customer's age in completed years
# Experience           Number of years of professional experience
# Income               Annual income of the customer ($000)
# ZIP Code             Home Address ZIP code.
# Family               Family size of the customer
# CCAvg                Avg. spending on credit cards per month ($000)
# Education            Education Level. 1: Undergrad; 2: Graduate; 3: Advanced/Professional
# Mortgage             Value of house mortgage if any. ($000)
# Personal Loan        Did this customer accept the personal loan offered in the last campaign?
# Securities Account   Does the customer have a securities account with the bank?
# CD Account           Does the customer have a certificate of deposit (CD) account with the bank?
# Online               Does the customer use internet banking facilities?
# CreditCard           Does the customer use a credit card issued by UniversalBank?
############################################################################################################


#data cleaning
data_bank <- data_bank[,-1]

#missing value check
sum(is.na(data_bank)) #the result is 0

#outliers check
summary(data_bank)
data_bank$Experience 
#we have minus on 'Experience' so we gonna fix it
for(i in 1:length(data_bank$Experience)){
  if(data_bank$Experience[i]<0){
    data_bank$Experience[i] <- 0
  }
}
min(data_bank$Experience)# minimum is 0

#standardising
library(caret)
st_mod <- preProcess(data_bank[,1:8],meathod=c("ceter","scale"))
data_bank_st <- cbind(predict(st_mod,data_bank[,1:8]),data_bank[,9:13])

#colmuns order change
data_bank_st <- dplyr::select(data_bank_st,Personal.Loan,everything())

##############################################################################
#data partitioning
n <- nrow(data_bank_st)
train_ind <- sample(1:n,n*0.6)
valid_ind <- sample(setdiff(1:n,train_ind),n*0.3)

train_data <- data_bank_st[train_ind,]
valid_data <- data_bank_st[valid_ind,]
test_data <- data_bank_st[-train_ind,]

train_x <- subset(train_data,select=-Personal.Loan)
train_y <- train_data[["Personal.Loan"]]
valid_x <- subset(valid_data,select=-Personal.Loan)
valid_y <- valid_data[["Personal.Loan"]]
test_x <- subset(test_data,select=-Personal.Loan)
test_y <- test_data[["Personal.Loan"]]

st_mod <- preProcess(train_x,method=c("center","scale"))
train_x <- predict(st_mod,train_x)
valid_x <- predict(st_mod,valid_x)
test_x <- predict(st_mod,test_x)

n_v <- length(valid_y)
n_p <- length(test_y)

#############################################################################
#結果代入しとくやつ作成
results <- data.frame('sensitivity','specificity')
############################################################################
#a) which model or models do you think are appropriate 
#   for this task? train a model for the bank and 
#   justify your choice of model
#カスタマーがパーソナルローンを受けるかどうか調べたい。営業のため。

#二値分類を行う。
#KNN,NBC,LogisticRegression, 

#KNN
library(FNN)
knn <- knn(train_x,test_x,train_y,k=10)#KNN fitting
sum(as.vector(knn)!= test_y)/n_p#testing error
table(knn,test_y)#confusion matrix
k_cand <- 1:20 #test k = 1 to 20
err_v <- numeric(length(k_cand))#storage for error rate
for(a in 1:length(k_cand)){
  #Loop for all k_cand
  k = k_cand[a]
  knn_valid <- knn(train_x,valid_x,train_y,k=k)
  err_v[a] <- sum(as.vector(knn_valid)!=valid_y)/n_v
}
k_opt <- k_cand[which(err_v==min(err_v))[1]]#the best model
knn_cl <- knn(train_x,test_x,train_y,k=k_opt)
sum(as.vector(knn_cl)!=test_y)/n_p
b <- table(knn_cl,test_y)
#sensitivity
round(b[1,1]/sum(b[,1]),3)
#specificity
round(b[2,2]/sum(b[,2]),3)

results <- data.frame('sensitybity' = round(b[1,1]/sum(b[,1]),3),
                      'specificity' = round(b[2,2]/sum(b[,2]),3))
rownames(results) <- 'KNN'

#NBC
library(e1071)
nbc <- naiveBayes(Personal.Loan ~.,data = train_data)#training
pred_class <- predict(nbc,newdata = test_data)#prediction of class
pred_prob <- predict(nbc,newdata=test_data,
                     type='raw')#prediction of probability
#evaluation
n_p <- nrow(test_data)
test_y <- test_data[,1]
sum(as.vector(pred_class) != as.vector(test_y))/n_p
c <- table(pred_class,test_y)
#sensitivity
round(c[1,1]/sum(c[,1]),3)
#specificity
round(c[2,2]/sum(c[,2]),3)
re <- data.frame('sensitybity'= round(c[1,1]/sum(c[,1]),3),
                 'specificity'= round(c[2,2]/sum(c[,2]),3))
rownames(re) <- 'NBC'
results = rbind(results,re)

#logistic regression
log_reg <- glm(train_data$Personal.Loan~.,data = train_data,
               family = "binomial")
options(scipen=999)
summary(log_reg)
pred_prob <- predict(log_reg,newdata= test_data,
                     type='response')
pred_prob <- as.vector(pred_prob)
pred_cl <- (pred_prob > 0.5)*1
#evaluation
n_p <- nrow(test_data)
test_y <- test_data[,1]
sum(pred_cl != test_y)/n_p
d <- table(pred_cl,test_y)
#sensitivity
round(d[1,1]/sum(d[,1]),3)#0.987
#specificity
round(d[2,2]/sum(d[,2]),3)#0.617

re <- data.frame('sensitybity'= round(d[1,1]/sum(d[,1]),3),
                 'specificity'= round(d[2,2]/sum(d[,2]),3))
rownames(re) <- 'logistic regression'
results <- rbind(results,re)




#b) which classification performance measure do you think
#   is going to be the most interesting for the bank?
# 1. sensitivity measure→正解を正解と判別する
# 2. specificity measure→間違いを間違いと判別する
# 3. error rate →単に正解も間違いも区別せずに、間違ったもののレートを考える
# 4. naive rule →　なにこれ
