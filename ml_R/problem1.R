#Problem1
set.seed(1)
data_college <- read.csv('College.csv')

###############################################
#
# data_college
##############################################
#  dim = 777,19
#  cols = "X" "Private" "Apps" "Accept" 
#         "Enroll" "Top10perc" "Top25perc" 
#         "F.Undergrad" "P.Undergrad" "Outstate"
#         "Room.Board" "Books" "Personal" "PhD"
#         "Terminal" "S.F.Ratio" "perc.alumni" 
#         "Expend" "Grad.Rate"  
#####################################################################################
# This dataset contains statistics for a large number of US Colleges 
# from the 1995 issue of US News and World Report.
# It contains 777 observations on the following 18 variables.
#   
# Variable         Description
# Private          If private or public university
# Apps             Number of received applications
# Accept	         Number of accepted applications
# Enroll	         Number of new students
# Top10perc	 Percent of new students from top 10% of high school class
# Top25perc	 Percent of new students from top 25% of high school class
# F.Undergrad      Number of fulltime undergraduates
# P.Undergrad      Number of parttime undergraduates
# Outstate	 Tuition for students from out-of-state
# Room.Board	 Cost of accommodation and food
# Books            Estimated book costs
# Personal	 Estimated personal spending
# PhD		 Percentage of faculty that have Ph.D.
# Terminal         Percentage of faculty with terminal degree
# S.F.Ratio	 Student to faculty rate
# perc.alumni      Percentage of alumni who donate
# Expend           Instructional expenditure per student
# Grad.Rate	 Graduation rate

###############################################


# a) train a model to predict Apps, using the 17 predictors in the dataset.
#    justify your choice of model

#→とりあえず多変量の回帰分析して、残差が最小なことを証明する
#回帰分析。前処理してパーティションする。
#privateがカテゴリカルなので、ダミー変数利用する

#大学名が邪魔なので消す
data_college <- data_college[,-1]

#privateのdummy variable 
library(fastDummies)
data_college <- dummy_cols(data_college,
                           remove_selected_columns = T,
                           remove_first_dummy = T) #ダミーを一個だけにするかどうか


#outliers check　外れ値確認
summary(data_college)#特になし

#missing value check 欠損値確認
sum(is.na(data_college)) #the result is 0

#we don't do normalization 標準化はしない


#data partition
n <- nrow(data_college)
train_ind <- sample(1:n,n*0.6)
train_data <- data_college[train_ind,]
test_data <- data_college[-train_ind,]


#ナイーブベイズ
library(forecast)
n_p <- length(test_data$Apps)
pred_nb <- rep(mean(train_data$Apps),n_p)
accuracy(pred_nb,test_data$Apps)

results <- data.frame(accuracy(pred_nb,test_data$Apps))
rownames(results) <- 'nb'


#regression　変数全部使ってみる
reg <- lm(Apps ~ .,data = data_college,subset = train_ind)
summary(reg)
tr_res <- data.frame(train_data$Apps,
                     reg$fitted.values,
                     reg$residuals)
names(tr_res) <- c('Actual','Fitted','Residual')
head(tr_res)
#prediction
pred <- predict(reg ,newdata = test_data)
pred_err <- test_data$Apps - pred
res <- data.frame(test_data$Apps,pred,pred_err)
names(res) <- c('Actual','Predicted','Prediction Error')
head(res)
#validation
library(forecast)
re <- accuracy(pred,test_data$Apps)#テストデータの予測値と正解ラベル
#                 ME     RMSE      MAE       MPE     MAPE
#Test set -115.1038 1060.416 650.6231 0.1534152 38.28258
result_ols <- data.frame(re)
rownames(result_ols) <- 'OLS_normal'
results <- rbind(results,result_ols)

accuracy(reg$fitted.values,train_data$Apps)#訓練データ中の関数の出力値と正解ラベル
#                              ME     RMSE      MAE      MPE     MAPE
#Test set -0.00000000000001667639 1041.606 634.7493 4.726202 42.77424
boxplot(pred_err,range=0)
hist(pred_err)
#過学習し過ぎ


#variable17個の中からstepwiseしてみる
reg_step <- step(reg)
summary(reg_step)
tr_res_step <- data.frame(train_data$Apps,
                     reg_step$fitted.values,
                     reg_step$residuals)
names(tr_res_step) <- c('Actual','Fitted','Residual')
head(tr_res_step)
#prediction
pred_step <- predict(reg_step ,newdata = test_data)
pred_err_step <- test_data$Apps - pred_step
res_step <- data.frame(test_data$Apps,pred_step,pred_err_step)
names(res_step) <- c('Actual','Predicted','Prediction Error')
head(res_step)
#validation
re <- accuracy(pred_step,test_data$Apps)#テストデータ
#               ME     RMSE      MAE        MPE     MAPE
#Test set -106.325 1049.594 643.0215 -0.9281163 37.85352
result_ols <- data.frame(re)
rownames(result_ols) <- 'OLS_stepwise'
results <- rbind(results,result_ols)

accuracy(reg_step$fitted.values,train_data$Apps)#トレーニングデータ
#                             ME     RMSE      MAE      MPE   MAPE
#Test set 0.00000000000001545979 1045.253 637.4689 2.501279 41.748
boxplot(pred_err_step,range=0)
hist(pred_err_step)


# b)Again train a model to predict Apps, but this time use 
#   the 17 variables in addition to their interactions, if we have 
#   three variables x1,x2,x3, their interactions are x1*x2,x2*x3 and x2*x3.
#   To use all the variables in addition to their interactions in lm(),
#   specify the formula Apps ~.^2

# to do this for glmnet, use the model.matrix() function to create the x matrix()
# give the formula Apps ~.^2 as a first argument to the function and then set
# the data argument to your training data,remember to also create this matrix for the
# testing data, as you will need to pass that to the newx argument in the predict() 
# function. justify your choice of model. compare the results with part a


#interactions の変数を作る。それに対してまたstepwiseで検証していく

#すべての組み合わせに対するinterceptを作成
#intercept154個で回帰分析
reg_2 <- lm(Apps~.^2,data = data_college,subset = train_ind)
summary(reg_2)
tr_res_2 <- data.frame(train_data$Apps,
                     reg_2$fitted.values,
                     reg_2$residuals)
names(tr_res_2) <- c('Actual','Fitted','Residual')
head(tr_res_2)
#prediction
pred_2 <- predict(reg_2 ,newdata = test_data)
pred_2_err <- test_data$Apps - pred_2
res_2 <- data.frame(test_data$Apps,pred_2,pred_2_err)
names(res_2) <- c('Actual','Predicted','Prediction Error')
head(res_2)
#validation
re <- accuracy(reg_2$fitted.values,test_data$Apps)
result_ols <- data.frame(re)
rownames(result_ols) <- 'OLS^2_normal'
results <- rbind(results,result_ols)
#                             ME     RMSE      MAE        MPE     MAPE
#Test set 0.00000000000002885305 522.9216 369.0149 -0.8994801 26.15219


#step-wise method
reg_2_step <- step(reg_2)
tr_res_2_step <- data.frame(train_data$Apps,
                       reg_2_step$fitted.values,
                       reg_2_step$residuals)
names(tr_res_2_step) <- c('Actual','Fitted','Residual')
head(tr_res_2_step)
#prediction
pred_2_step <- predict(reg_2_step ,newdata = test_data)
pred_2_err_step <- test_data$Apps - pred_2_step
res_2_step <- data.frame(test_data$Apps,pred_2_step,pred_2_err_step)
names(res_2_step) <- c('Actual','Predicted','Prediction Error')
head(res_2_step)
#validation
re <- accuracy(pred_2_step,test_data$Apps)
result_ols <- data.frame(re)
rownames(result_ols) <- 'OLS^2_stepwise'
results <- rbind(results,result_ols)
#               ME    RMSE      MAE       MPE     MAPE
#Test set 28.90709 1163.11 698.3142 0.3215443 39.64447
accuracy(reg_2_step$fitted.values,train_data$Apps)
#                             ME     RMSE     MAE        MPE     MAPE
#Test set 0.00000000000000104411 540.3023 384.221 -0.9598742 26.66907

plot(res_2_step$Predicted,type='l',col='red')
par(new=T)
plot(res_2_step$Actual,type='l',col='blue')


#Ridge
library(glmnet)
x_ridge <- as.matrix(subset(train_data,select=-Apps))
y_ridge <- train_data$Apps 
reg_ridge <- glmnet(x_ridge,y_ridge,alpha=0)
reg_ridge$lambda
coef(reg_ridge)
#cross validation
lambda_grid <- 10^seq(10,-2,length=100)
cv_out <- cv.glmnet(x_ridge,y_ridge,alpha=0,
                    nfolds=5,lambda = lambda_grid)
plot(cv_out)
lambda_opt <- cv_out$lambda.min
reg_ridge <- glmnet(x_ridge,y_ridge,alpha=0,lambda=lambda_opt)
coef(reg_ridge)
#prediction and evaluation
pred_ridge <- as.vector(predict(reg_ridge,
                                newx = as.matrix(subset(test_data,select = -Apps))))
re <- accuracy(pred_ridge,test_data$Apps)
result_ols <- data.frame(re)
rownames(result_ols) <- 'Ridge'
results <- rbind(results,result_ols)
#        ME     RMSE      MAE       MPE     MAPE
#Test set -115.3179 1060.058 650.4095 0.1669512 38.27713

#LASSO
x_lasso <- as.matrix(subset(train_data,select=-Apps))
y_lasso <- train_data$Apps 
reg_lasso <- glmnet(x_lasso,y_lasso,alpha=1)
reg_lasso$lambda
dim(coef(reg_lasso))
coef(reg_lasso)

#cross validation
cv_out <- cv.glmnet(x_lasso,y_lasso,alpha = 1,nforlds=5)
plot(cv_out)
lambda_opt <- cv_out$lambda.min
reg_lasso <- glmnet(x_lasso,y_lasso,alpha=1,lambda=lambda_opt)
coef(reg_lasso)

#prediction and evaluation
pred_lasso <- as.vector(predict(reg_lasso,
                                newx=as.matrix(subset(test_data,select=-Apps))))
re <- accuracy(pred_lasso,test_data$Apps)
result_ols <- data.frame(re)
rownames(result_ols) <- 'LASSO'
results <- rbind(results,result_ols)
#                ME     RMSE      MAE        MPE     MAPE
#Test set -118.6305 1045.166 634.2429 0.09896771 37.00853


