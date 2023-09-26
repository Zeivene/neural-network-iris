# *********************************************************************
# ACTIVATION FUNCTIONS ************************************************
# *********************************************************************

# sigmoid

sigmoid <- function(x){
  
  1/(1+exp(-x))
  
}

id <- function(x){x}

# make a "library" of activation functions

softmax <- function(z){
  
  exp(z) / sum(exp(z))
  
}

Did <- function(x){1}

Dsig <- function(x){exp(-x) / ((1+exp(-x))^2)}

Dtanh <- function(x){1-(tanh(x)^2)}

D.softmax <- function(z){
  
  n <- length(z)
  S <- softmax(z)
  DS <- matrix(NA, nrow = n, ncol = n)
  
  for(i in 1:n){
    for(j in 1:n){
      if(i == j){
        D.i.j <- S[i]*(1-S[j])
      }else{
        D.i.j <- -S[i]*S[j]
      }
      DS[i,j] <- D.i.j
    }}
  return(DS)
}

AF  <- list(id, sigmoid, tanh, softmax)
DAF <- list(Did,Dsig,Dtanh, D.softmax)

# *********************************************************************
# LOSS FUNCTIONS ******************************************************
# *********************************************************************

mse <- function(A3,K){
  
  mean((A3 - K)^2)
  
}

cross.entropy <- function(A, K){
  
  return(sum( K * log(1/A) ))
  
}

D.cross.entropy <- function(A, K){
  
  return(-K/A)
  
}

# *********************************************************************
# ONE HOT CONVERT *****************************************************
# *********************************************************************

one.hot <- function(types, type){
  
  number.of.types <- length(types)  
  
  classes <- vector("list", number.of.types)
  
  for(n in 1:number.of.types){
    
    class <- rep(0,number.of.types)
    class[n] <- 1
    classes[[n]] <- class
  }
  
  position.in.types <- which(types == type)
  class.for.type    <- classes[[position.in.types]]
  
  return(class.for.type)
}

# *********************************************************************
# FORWARD PASS ********************************************************
# *********************************************************************

fwd.pass <- function(input, parameters, AF.chosen){
  
  W1 = parameters$W[[1]]
  W2 = parameters$W[[2]]
  
  B1 = parameters$B[[1]]
  B2 = parameters$B[[2]]
  
  AF1 = AF[[AF.chosen[1]]]
  AF2 = AF[[AF.chosen[2]]]
  AF3 = AF[[AF.chosen[3]]]
  
  Z1 = input
  A1 = AF1(Z1)
  Z2 = t(W1) %*% A1 + B1
  A2 = AF2(Z2)
  Z3 = t(W2) %*% A2 + B2
  A3 = AF3(Z3)
  
  return(list("Z" = list(Z1,Z2,Z3), "A" = list(A1,A2,A3)))
  
}

# *********************************************************************
# INITIALIZE PARAMETERS ***********************************************
# *********************************************************************

initialize.parameters <- function(N,a){
  
  W1 <- matrix(runif(N[1]*N[2],-a,a),nrow = N[1])
  W2 <- matrix(runif(N[2]*N[3],-a,a),nrow = N[2])
  B1 <- matrix(runif(N[2],-a,a),nrow = N[2])
  B2 <- matrix(runif(N[3],-a,a),nrow = N[3])
  
  return(list("W" = list(W1,W2), "B" = list(B1,B2)))
  
}

# *********************************************************************
# BACK PROPAGATION ****************************************************
# *********************************************************************

back.propagation <- function(forwardPass,
                             parameters, 
                             AF.chosen, 
                             LF.chosen, 
                             knownResult){
  
  # extract the neuron values
  A3 <- forwardPass$A[[3]]
  A2 <- forwardPass$A[[2]]
  A1 <- forwardPass$A[[1]]
  Z3 <- forwardPass$Z[[3]]
  Z2 <- forwardPass$Z[[2]]
  
  # extract the parameters matrices from "parameters"
  W2 <- parameters$W[[2]]
  W1 <- parameters$W[[1]]
  B2 <- parameters$B[[2]]
  B1 <- parameters$B[[1]]
  # get the required activation function derivatives
  DAF3 <- DAF[[ AF.chosen[3] ]]
  DAF2 <- DAF[[ AF.chosen[2] ]]
  
  # rate of change of Loss as a function of A3 values
  if(LF.chosen == 1){
    DL <- A3 - knownResult
    # rate of change of Loss as a function of Z3 values
    DZ3 <- DL*DAF3(Z3)
    
  }
  
  else if(LF.chosen == 2){
    
    DL   <- D.cross.entropy(A3, knownResult)
    DZ3  <- DAF3(Z3) %*% DL    
    
  }
  # rate of change of Loss as a function of Z2 values
  DZ2 <- (W2 %*% DZ3) * DAF2(Z2)
  # rate of change of Loss as a function of Weights (2nd)
  DW2 <- A2 %*% t(DZ3)
  # rate of change of Loss as a function of Weights (1st)
  DW1 <- A1 %*% t(DZ2)
  # rate of change of Loss as a function of Biases
  DB2 <- DZ3
  DB1 <- DZ2
  
  gradients <- list("DW"=list(DW1, DW2), "DB"=list(DB1, DB2))
  
  return(gradients)
  
}

# *********************************************************************
# UPDATE PARAMETERS ***************************************************
# *********************************************************************

update.parameters <-function(gradients, parameters, rate){
  
  weights <- parameters$W
  biases  <- parameters$B
  DW      <- gradients$DW
  DB      <- gradients$DB
  
  S <- length(weights)
  
  new.weights <- vector("list", S)
  new.biases  <- vector("list", S)
  
  for(s in 1:S){
    
    w                 <- weights[[s]]
    new.weights.s     <- w - (rate * DW[[s]])
    b                 <- biases[[s]]
    new.biases.s      <- b - (rate * DB[[s]])
    
    new.weights[[s]]  <- new.weights.s
    new.biases[[s]]   <- new.biases.s
  }
  
  updated.parameters <- list("W" = new.weights, "B" = new.biases)
  
  return(updated.parameters)
  
}

# *********************************************************************
# DATA PREPARATION ****************************************************
# *********************************************************************

library(rlist)

trainNet <- function(data, N, AF.chosen, epochs){


trainingProportion <- 0.8
numberOfRows       <- nrow(data)
data               <- data[sample(1:numberOfRows),]
trainingRows       <- floor(trainingProportion*numberOfRows)
trainingData       <- data[1:trainingRows,]
testingData        <- data[((trainingRows+1):numberOfRows), ]

# Neural Net Setup
M             <- ncol(data) - 1 
classCol      <- M + 1
numberOfTypes <- length(unique(data[,classCol]))
types         <- sort(unique(data[,classCol]))
step          <- 0.01

# Setup Loss Plot
plot(NA,xlim = c(0,epochs), ylim = c(0,1))

parameters    <- initialize.parameters(N,2)

for(e in 1:epochs){
  
  losses <- c()
  
  for(row in 1:nrow(trainingData)){
    
    input      <- matrix(as.numeric(trainingData[row,1:M]), nrow = M)
    known      <- matrix(one.hot(types, trainingData[row, classCol]), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    grads      <- back.propagation(fPass, parameters, AF.chosen,2,known)
    parameters <- update.parameters(grads,parameters, step)
    
    losses <- c(losses,mse(fPass$A[[3]], known))
    
  }
  
  epochLoss <- mean(losses)
  
  points(e,epochLoss)
  
}

return(list("parameters" = parameters, 
            "trainingData" = trainingData, 
            "testingData" = testingData,  
            "structureVector" = N,
            "AFchosen" = AF.chosen))

}

#trainedNet1 <- trainNet(data = iris, N = c(4,8,3), AF.chosen = c(1,2,4), epochs = 100)
#list.save(trainedNet1, file = 'irisNN1.RData')
#trainedNet1 <- list.load("irisNN1.RData")

# trainedNet2 <- trainNet(data = iris, N = c(4,8,3), AF.chosen = c(1,3,4), epochs = 100)
# list.save(trainedNet2, file = 'irisNN2.RData')
# trainedNet2 <- list.load("irisNN2.RData")

# trainedNet3 <- trainNet(data = iris[,c(2,3,5)], N = c(2,8,3), AF.chosen = c(1,3,4), epochs = 100)
# list.save(trainedNet3, file = 'irisNN3.RData')
# trainedNet3 <- list.load("irisNN3.RData")

# trainedNet4 <- trainNet(data = iris[,c(2,3,4,5)], N = c(3,8,3), AF.chosen = c(1,3,4), epochs = 100)
# list.save(trainedNet4, file = 'irisNN4.RData')
# trainedNet4 <- list.load("irisNN4.RData")

# *********************************************************************
# PERFORMANCE *********************************************************
# *********************************************************************

testNet <- function(trainedNet, dataChoice){
  
  parameters <- trainedNet$parameters
  AF.chosen  <- trainedNet$AFchosen
  
  if(dataChoice == 1){
    data       <- trainedNet$testingData
  } else if (dataChoice == 2){
    data       <- trainedNet$trainingData
  } else if (dataChoice == 3){
    data <- rbind.data.frame(trainedNet$testingData, trainedNet$trainingData)
  }
  
  M             <- ncol(data) - 1 
  classCol      <- M + 1
  numberOfTypes <- length(unique(data[,classCol]))
  types         <- sort(unique(data[,classCol]))
  
  accuracy  <- 0
  
  for(row in 1:nrow(data)){
    
    
    input      <- matrix(as.numeric(data[row,1:M]), nrow = M)
    known      <- matrix(one.hot(types, data[row, classCol]), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    prediction <- round(fPass$A[[3]],0)
    
    if(all(known == prediction)){
      
      accuracy <- accuracy + 1
      
    }
  }
  
  return(accuracy/nrow(data)*100)
  
}


basicTest <- function(data, trainedNet){
  
  parameters <- trainedNet$parameters
  AF.chosen  <- trainedNet$AFchosen

  M             <- ncol(data) - 1 
  classCol      <- M + 1
  numberOfTypes <- length(unique(data[,classCol]))
  types         <- sort(unique(data[,classCol]))
  
  accuracy  <- 0
  
  for(row in 1:nrow(data)){
    
    input      <- matrix(as.numeric(data[row,1:M]), nrow = M)
    known      <- matrix(one.hot(types, data[row, classCol]), nrow = numberOfTypes) 
    fPass      <- fwd.pass(input, parameters, AF.chosen)
    prediction <- round(fPass$A[[3]],0)
    
    if(all(known == prediction)){
      
      accuracy <- accuracy + 1
      
    }
  }
  
  return(accuracy/nrow(data)*100)
  
}
  

  
  











