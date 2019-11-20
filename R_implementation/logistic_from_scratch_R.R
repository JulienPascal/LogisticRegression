# Logistic regression in R
# Source: https://rpubs.com/junworks/Understanding-Logistic-Regression-from-Scratch

library(ggplot2)
library(dplyr)
#sigmoid function, inverse of logit
sigmoid <- function(z){1/(1+exp(-z))}

#cost function
cost <- function(theta, X, y){
  m <- length(y) # number of training examples

  h <- sigmoid(X%*%theta)
  J <- (t(-y)%*%log(h)-t(1-y)%*%log(1-h))/m
  J
}

#gradient function
grad <- function(theta, X, y){
  m <- length(y)

  h <- sigmoid(X%*%theta)
  grad <- (t(X)%*%(h - y))/m
  grad
}

logisticReg <- function(X, y){
  #remove NA rows
  temp <- na.omit(cbind(y, X))
  #add bias term and convert to matrix
  X <- mutate(temp[, -1], bias =1)
  X <- as.matrix(X[,c(ncol(X), 1:(ncol(X)-1))])
  y <- as.matrix(temp[, 1])
  #initialize theta
  theta <- matrix(rep(0, ncol(X)), nrow = ncol(X))
  #use the optim function to perform gradient descent
  costOpti <- optim(matrix(rep(0, 4), nrow = 4), cost, grad, X=X, y=y)
  #return coefficients
  return(costOpti$par)
}

#load the dataset
  shot <- read.csv('/home/julien/Documents/REPOSITORIES/LogisticRegression/data/shot_logs.csv', header = T, stringsAsFactors = F)
shot.df <- select(shot, FGM, SHOT_CLOCK, SHOT_DIST, CLOSE_DEF_DIST)
head(shot.df)

shot.X <- shot.df[, -1]
shot.y <- shot.df[, 1]

mod <- logisticReg(shot.X, shot.y)
mod

mod1 <- glm(as.factor(FGM) ~ SHOT_CLOCK + SHOT_DIST + CLOSE_DEF_DIST, 
            family=binomial(link = "logit"), data=shot.df)

summary(mod1)$coefficients