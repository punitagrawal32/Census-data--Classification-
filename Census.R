#### Problem Description 
# We have been given the census data for 30K individulas. We are required to predict
# if the income of an individual will exceed 50K or not. We have been given many general
# educational and employment related attributes, which we will use to predict the income 
# of a person. 
# General- age, fnlwgt, maital,  relationship, sex, capital_gain, capital_loss, country
# Employment- type_employer, occupation
# Educational- education, education_num




# Model: A
# 1. All variables origianl data
# Train:: AUC: 90.62%  Accuracy:  84.91% Test::AUC: 90.70%  Accuracy: 84.85%
# 2. All variables standardized data
# Train:: AUC: 90.71%  Accuracy:  84.91% Test::AUC: 90.53%  Accuracy: 84.85%
# 3. Model 1 STEP AIC 
# Train:: AUC:  90.53% Accuracy: 84.80% Test::AUC: 90.60% Accuracy: 84.77% 
# 4. Model 2 STEP AIC 
# Train:: AUC:  90.53% Accuracy: 84.80% Test::AUC: 90.71% Accuracy: 84.77% 

# Model: B 
# 1. Dropping Capital Gain column
# Train:: AUC: 88.91% Accuracy: 83.55% Test::AUC: 88.73% Accuracy: 83.1%
# 2. Dropping Capital Loss column
# Train:: AUC:  88.91% Accuracy: 84.66%  Test::AUC: 88.60%  Accuracy: 84.67%
# 3. Dropping Capital Gain and Capital Loss
# Train:: AUC:  88.76% Accuracy:  83.47% Test::AUC: 88.50%  Accuracy: 83.06%
# 4. Dropping education_num column
# Train:: AUC:  90.62% Accuracy: 84.91%  Test::AUC: 90.71%  Accuracy: 84.85% 
# 5. Dropping fnlwgt
# Train:: AUC:  90.60% Accuracy:  84.86% Test::AUC: 90.68% Accuracy: 84.73%
# 6. StepAIC on model 4
# Train:: AUC:  90.53%  Accuracy: 84.80%  Test::AUC: 90.71%  Accuracy: 84.77%
# 7. StepAIC on model 5
# Train:: AUC:  90.52% Accuracy:  84.73% Test::AUC: 90.69% Accuracy: 84.69%
# 8. Model B:1 on standardized data
# Train:: AUC: 88.91% Accuracy: 83.55% Test::AUC: 88.76% Accuracy: 83.11%
# 9. Model B:2 on unstandardized data
# Train:: AUC:  90.38% Accuracy: 84.66%  Test::AUC: 90.35%  Accuracy: 84.67%

# Model: C
# Factor level reduction
# Train:: AUC:  90.49%  Accuracy: 84.73%  Test:: AUC: 90.70%  Accuracy: 84.71%

# Model: D
# Ridge Regression
# Train:: AUC:  90.51% Accuracy: 84.77 Test::AUC: 90.70%  Accuracy: 84.74
# Lasso Regression
# Train:: AUC:  90.50%  Accuracy:  84.79% Test:: AUC:  90.69% Accuracy: 84.75%

# Model: E
# Naive Bayes algorithm: 
# Train:: Accuracy: 81.32% Test:: Accuracy: 81.75%



## 1. Setting the working directory and clearing the R environment

rm(list=ls(all=T))
setwd("C:/Users/Punit/Desktop")

## 2. Loading the required libraries 

library(glmnet)
library(mice)
library(dplyr)
library(ROCR)
library(DMwR)
library(car)
library(MASS)
library(vegan)
library(dummies)
library(infotheo)
library(caTools)
library(caret)
library(e1071)
library(corrplot)

## 3. Reading the data in R 

census= read.csv("Census_Dataset.csv", header= T, sep= ",",na.strings="?")
View(census)

## 4. Data Exploration 

dim(census)
# 32561 x 15 
# 488415 values 

str(census)
# Getting the structure of the data 

summary(census)
# Getting the summary of the data

sum(is.na(census))
# 4262 missing values

colSums(is.na(census))
# the missing values seem to be in the employer type, occupation and country variables 

# Variable explanation 

# 1. age: Age of the observation, Varies from 17 to 90 with mean of 38.58 
plot(census$age)
age_exp= discretize(census$age, "equalwidth")
age_exp
table(age_exp)
barplot(table(age_exp), xlab= "Age Bins", ylab= "Frequency", main= "Distribution of age in the dataset")
plot(census$age, census$income, main=" Age vs Income class")

# 2. type_employer: Type of employment of the person. Working for the Central Govt, State 
# Govt, private sector, self employed, without pay, etc
table(census$type_employer)
barplot(table(census$type_employer), xlab="Employment", ylab="Frequency", main= "Distribution of employer type")
plot(census$type_employer, census$income, main=" Employer type vs Income class")
# Through the table and the plot, we know that most of the people work for private firms 


# 3. fnlwgt (Final sampling weight): This attribute is calculated using 3 different sets 
# of controls,and finally taking a weihgted average of these controls 
fnlwgt_exp= discretize(census$fnlwgt, "equalwidth")
fnlwgt_exp
barplot(table(fnlwgt_exp), xlab= "fnwgt bins", ylab= "Frequency", main= "Fnlwgt values across various bins")
table(fnlwgt_exp)
plot(census$fnlwgt, census$income, main=" fnlwgt vs Income class")

# We observe that very few values are above the median value, and majority of the points 
# fade off after a certain level 

# 4. education: The education level of candidates. Factor with 16 levels like 10th, 11th,
# 12th, undergraduate level bachelors, doctorates, masters, preschool 
table(census$education)
barplot(table(census$education), xlab="Education", ylab="Frequency", main="Frequency of different education levels")
# Most of the observations are HS grads, have a bachelors degree or have done some schooling 
plot(census$education, census$income, main=" Education vs Income class")

# 5. education_num: Although this is read as an integer variable, this looks more like an 
# ordinal variable with the numbers denoting different education level
table(census$education_num)
census$education_num= as.factor(census$education_num)
barplot(table(census$education_num), xlab="Education level (Ordinal)", yab="Frequency", main=" Frequency across education levels" )

# 6. marital: Factor variable with 7 levels describing if a person is married or not, and
# if married, if it is civilian or an armed force. 
# Separated and divored apparrently not the same. Separated implies that the couple is
# still legally married, but do not live together
table(census$marital)
barplot(table(census$marital), xlab="Marital", ylab="Frequency", main="Marital freqencies")
plot(census$marital, census$income, main="Marital vs Income class")

# 7.occupation: Categorical variable with 14 levels 
table(census$occupation)
barplot(table(census$occupation), xlab="Occupation", ylab="Frequency", main=" Frequency across variable occupation")
plot(census$occupation, census$income, main= "Occupation vs Income class ")

# 8. Relationship: Categorical variable with 6 levels. Quite straightforward 
table(census$relationship)
barplot(table(census$relationship), xlab="Relationship", ylab="Frequency", main="Frequency across relationships")
plot(census$relationship, census$income, main="Relationship vs Income class")

# 9. race: Categorical variable with 5 levels. Nothing much to look at here  
table(census$race)
barplot(table(census$race), xlab="Race", ylab="Frequency", main="Frequency across race")
plot(census$race, census$income, main="Rave vs Income Class")

# 10. sex: Gender. 2 levels 
table(census$sex)
barplot(table(census$sex), xlab=" Gender", ylab="Frequency", main="Frequency across gender")
plot(census$sex, census$income, main="Gender vs Income class")

# 11. Capital Gain: This variable denotes the income from investments 
str(census)
summary(census)
table(census$capital_gain>0)
hist(census$capital_gain, main=" Capital Gain", xlab="Value", ylab=" Frequency", col="black")
plot(census$capital_gain, census$income)
# Most of the values here are 0. We will most likely drop these columns in our analysis ahead

# 12. Capital Loss: This variable denotes the losses from investments
str(census)
summary(census)
table(census$capital_gain>0)
hist(census$capital_loss, main=" Capital Loss", xlab="Value", ylab=" Frequency", col="black")
plot(census$capital_loss, census$income)
# Most of the values here are 0. We will most likely drop these columns in our analysis ahead

# 13. hr_per_week: The house rent being paid per week
summary(census)
hist(census$hr_per_week)
plot(census$hr_per_week, census$income)

#14. country: The country the observed person is from. 41 levels here
table(census$country)
barplot(table(census$country), xlab="Country", ylab=" frequency", main="Frequency acrosss countries")
plot(census$country, census$income)

# 15. Income: Target variable with 2 levels. This is what we are looking to predict. 
table(census$income)[[1]]/nrow(census)
# 76% is the BASELINE MODEL accuracy upon which we have to improve
barplot(table(census$income), main= "Income levels")


## 5.Data Cleaning 

md.pattern(census)

# This tells us that missing values in type_employer and occupation co-occur most of the 
# time. 

set.seed(123)
cenkNN= na.omit(census)
cenkNN
sum(is.na(cenkNN))

## 0 NA values here now 

summary(cenkNN)
str(cenkNN)

cenkNN$education_num= as.factor(cenkNN$education_num)
table(cenkNN$country)
which(cenkNN$country=="Holand-Netherlands")
cenkNN=cenkNN[-19610,]
num= cenkNN[,sapply(cenkNN, is.integer)]
num
num_std= decostand(num, method = "standardize")
cat= cenkNN[,sapply(cenkNN, is.factor)]
dim(cat)
cenkNN_std= data.frame(num_std,cat)
cenkNN_std
dim(cenkNN_std)
View(cenkNN_std)


## 6. Model: A
# 1. All variables origianl data      
# 2. All variables standardized data
# 3. Model 1 STEP AIC 
# 4. Model 2 STEP AIC 


set.seed(123)
rows1=sample.split(cenkNN$income,SplitRatio = 0.7)
rows1
train1= cenkNN[rows1==T,]
test1= cenkNN[rows1==F,]
test1
dim(train1)

model1= glm(income~., data= train1, family="binomial")
summary(model1)
# AIC of 13861
par(mfrow=c(2,2))
plot(model1)

pred1_1= predict(model1,type="response")
prob1_1= prediction(pred1_1,train1$income)
auc= performance(prob1_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.62%
pred1= predict(model1, newdata=test1, type= "response" )
prob1= prediction(pred1, test1$income)
perf1= performance(prob1,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf1, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob1, measure="auc")
auc@y.values[[1]]
# AUC is 90.71%


val= ifelse(pred1_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.91


val= ifelse(pred1>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.85


set.seed(123)
rows2=sample.split(cenkNN_std$income,SplitRatio = 0.7)
train2= cenkNN_std[rows2==T,]
test2= cenkNN_std[rows2==F,]

model2= glm(income~., data= train2, family="binomial")
summary(model2)
# AIC of 13861
par(mfrow=c(2,2))
plot(model2)

pred2_1= predict(model2,type="response")
prob2_1= prediction(pred2_1,train2$income)
auc= performance(prob2_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.62%
pred2= predict(model2, newdata=test2, type= "response" )
prob2= prediction(pred2, test2$income)
perf2= performance(prob2,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf2, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob2, measure="auc")
auc@y.values[[1]]
# AUC is 90.71%


val= ifelse(pred2_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train2$income)

# Train accuracy of 84.91


val= ifelse(pred2>0.5, ">50K", "<=50K")
confusionMatrix(val, test2$income)

# Test accuracy of 84.85

model3= stepAIC(model1)
summary(model3)
# AIC of 14508
par(mfrow=c(2,2))
plot(model3)

pred3_1= predict(model3,type="response")
prob3_1= prediction(pred3_1,train1$income)
auc= performance(prob3_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.53%
pred3= predict(model3, newdata=test1, type= "response" )
prob3= prediction(pred3, test1$income)
perf3= performance(prob3,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf3, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob3, measure="auc")
auc@y.values[[1]]
# AUC is 90.71%

val= ifelse(pred3_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.8


val= ifelse(pred3>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.54


model4= stepAIC(model2)
summary(model4)
# AIC of 14508
par(mfrow=c(2,2))
plot(model4)

pred4_1= predict(model4,type="response")
prob4_1= prediction(pred4_1,train2$income)
auc= performance(prob4_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.53%
pred4= predict(model4, newdata=test2, type= "response" )
prob4= prediction(pred4, test2$income)
perf4= performance(prob4,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf4, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob4, measure="auc")
auc@y.values[[1]]
# AUC is 90.71%


val= ifelse(pred4_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train2$income)

# Train accuracy of 84.8


val= ifelse(pred4>0.5, ">50K", "<=50K")
confusionMatrix(val, test2$income)

# Test accuracy of 84.77


## 7. Model: B 
# 1. Dropping Capital Gain column
# 2. Dropping Capital Loss column
# 3. Dropping Capital Gain and Capital Loss
# 4. Dropping education_num column
# 5. Dropping fnlwgt
# 6. StepAIC on model 4
# 7. StepAIC on model 5
# 8. Model B:1 on standardized data
# 9. Model B:2 on unstandardized data

model1= glm(income~.-capital_gain, data= train1, family="binomial")
summary(model1)
# AIC of 15687
par(mfrow=c(2,2))
plot(model1)

pred1_1= predict(model1,type="response")
prob1_1= prediction(pred1_1,train1$income)
auc= performance(prob1_1,measure="auc")
auc@y.values[[1]]
# AUC is 88.91%
pred1= predict(model1, newdata=test1, type= "response" )
prob1= prediction(pred1, test1$income)
perf1= performance(prob1,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf1, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob1, measure="auc")
auc@y.values[[1]]
# AUC is 88.76


val= ifelse(pred1_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 83.55


val= ifelse(pred1>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 83.11



model2= glm(income~.-capital_loss, data= train1, family="binomial")
summary(model2)
# AIC of 14761
par(mfrow=c(2,2))
plot(model2)

pred2_1= predict(model2,type="response")
prob2_1= prediction(pred1_1,train1$income)
auc= performance(prob2_1,measure="auc")
auc@y.values[[1]]
# AUC is 88.91%
pred2= predict(model2, newdata=test1, type= "response" )
prob2= prediction(pred2, test1$income)
perf2= performance(prob2,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf2, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob2, measure="auc")
auc@y.values[[1]]
# AUC is 90.35%


val= ifelse(pred2_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.66


val= ifelse(pred2>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.67


model3= glm(income~.-capital_gain-capital_loss, data= train1, family="binomial")
summary(model3)
# AIC of 15854
par(mfrow=c(2,2))
plot(model3)

pred3_1= predict(model3,type="response")
prob3_1= prediction(pred3_1,train1$income)
auc= performance(prob3_1,measure="auc")
auc@y.values[[1]]
# AUC is 88.76%
pred3= predict(model3, newdata=test1, type= "response" )
prob3= prediction(pred3, test1$income)
perf3= performance(prob3,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf3, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob3, measure="auc")
auc@y.values[[1]]
# AUC is 88.50%


val= ifelse(pred3_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 83.47


val= ifelse(pred3>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 83.06


model4= glm(income~.-education_num, data= train1, family="binomial")
summary(model4)
# AIC of 14514
par(mfrow=c(2,2))
plot(model4)

pred4_1= predict(model4,type="response")
prob4_1= prediction(pred4_1,train1$income)
auc= performance(prob4_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.62%
pred4= predict(model4, newdata=test1, type= "response" )
prob4= prediction(pred4, test1$income)
perf4= performance(prob4,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf4, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob4, measure="auc")
auc@y.values[[1]]
# AUC is 90.71%


val= ifelse(pred4_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.91


val= ifelse(pred4>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.85


model5= glm(income~.-fnlwgt, data= train1, family="binomial")
summary(model5)
# AIC of 14526
par(mfrow=c(2,2))
plot(model5)

pred5_1= predict(model5,type="response")
prob5_1= prediction(pred5_1,train1$income)
auc= performance(prob5_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.60%
pred5= predict(model5, newdata=test1, type= "response" )
prob5= prediction(pred5, test1$income)
perf5= performance(prob5,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf5, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob5, measure="auc")
auc@y.values[[1]]
# AUC is 90.68%


val= ifelse(pred5_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.86


val= ifelse(pred5>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.73


model6= stepAIC(model4)
summary(model6)
# AIC of 14508
par(mfrow=c(2,2))
plot(model6)

pred6_1= predict(model6,type="response")
prob6_1= prediction(pred6_1,train1$income)
auc= performance(prob6_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.53%
pred6= predict(model6, newdata=test1, type= "response" )
prob6= prediction(pred6, test1$income)
perf6= performance(prob6,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf6, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob6, measure="auc")
auc@y.values[[1]]
# AUC is 90.71%


val= ifelse(pred6_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.8


val= ifelse(pred6>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.77


model7= stepAIC(model5)
summary(model7)
# AIC of 14517
par(mfrow=c(2,2))
plot(model7)

pred7_1= predict(model7,type="response")
prob7_1= prediction(pred7_1,train1$income)
auc= performance(prob7_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.52%
pred7= predict(model7, newdata=test1, type= "response" )
prob7= prediction(pred7, test1$income)
perf7= performance(prob7,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf7, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob7, measure="auc")
auc@y.values[[1]]
# AUC is 90.69%


val= ifelse(pred7_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train1$income)

# Train accuracy of 84.73


val= ifelse(pred7>0.5, ">50K", "<=50K")
confusionMatrix(val, test1$income)

# Test accuracy of 84.69


model8= glm(income~.-capital_gain, data= train2, family="binomial")
summary(model8)
# AIC of 15687
par(mfrow=c(2,2))
plot(model8)

pred8_1= predict(model8,type="response")
prob8_1= prediction(pred8_1,train2$income)
auc= performance(prob8_1,measure="auc")
auc@y.values[[1]]
# AUC is 88.91%
pred8= predict(model8, newdata=test2, type= "response" )
prob8= prediction(pred8, test2$income)
perf8= performance(prob8,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf8, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob8, measure="auc")
auc@y.values[[1]]
# AUC is 88.76%


val= ifelse(pred8_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train2$income)

# Train accuracy of 83.55


val= ifelse(pred8>0.5, ">50K", "<=50K")
confusionMatrix(val, test2$income)

# Test accuracy of 83.11


model9= glm(income~.-capital_loss, data= train2, family="binomial")
summary(model9)
# AIC of 14761
par(mfrow=c(2,2))
plot(model9)

pred9_1= predict(model9,type="response")
prob9_1= prediction(pred9_1,train2$income)
auc= performance(prob9_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.38%
pred9= predict(model9, newdata=test2, type= "response" )
prob9= prediction(pred9, test2$income)
perf9= performance(prob9,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf9, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob9, measure="auc")
auc@y.values[[1]]
# AUC is 90.35%


val= ifelse(pred9_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train2$income)

# Train accuracy of 84.66


val= ifelse(pred9>0.5, ">50K", "<=50K")
confusionMatrix(val, test2$income)

# Test accuracy of 84.67


## We observe that the performance of the standardized data and the un-standardardized 
## data is the same. Hence, from now on we will use only the unstandardized data.


## 8. Model: C
# Reducing the number of levels in all columns. Using that to make predictions. 
# Also, using StepAIC on the new model


train3=train2
test3=test2

train3$race= ifelse(!train3$race=="White","Not-White","White")
test3$race= ifelse(!test3$race=="White","Not-White","White")
train3$race= as.factor(train3$race)
test3$race= as.factor(test3$race)
table(train3$race)

train3$country= ifelse(train3$country=="United-States","US","Not-US")
test3$country= ifelse(test3$country=="United-States","US","Not-US")
train3$country=as.factor(train3$country)
test3$country=as.factor(test3$country)
table(train3$country)

table(train3$education, train3$income)
train3$education= ifelse(train3$education=="10th"|train3$education=="11th"|train3$education=="12th"|train3$education=="1st-4th"|train3$education=="5th-6th"|train3$education=="7th-8th"|train3$education=="9th","<=12",
                         ifelse(train3$education=="Assoc-acdm"|train3$education=="Assoc-voc","Assoc",
                                ifelse(train3$education=="Bachelors","Bachelors",
                                       ifelse(train3$education=="Doctorate"|train3$education=="Prof-school","PhD",
                                              ifelse(train3$education=="HS-grad","HS-grad",
                                                     ifelse(train3$education=="Masters","Masters",
                                                            ifelse(train3$education=="Preschool","Preschool","Some-college")))))))
test3$education= ifelse(test3$education=="10th"|test3$education=="11th"|test3$education=="12th"|test3$education=="1st-4th"|test3$education=="5th-6th"|test3$education=="7th-8th"|test3$education=="9th","<=12",
                        ifelse(test3$education=="Assoc-acdm"|test3$education=="Assoc-voc","Assoc",
                               ifelse(test3$education=="Bachelors","Bachelors",
                                      ifelse(test3$education=="Doctorate"|test3$education=="Prof-school","PhD",
                                             ifelse(test3$education=="HS-grad","HS-grad",
                                                    ifelse(test3$education=="Masters","Masters",
                                                           ifelse(test3$education=="Preschool","Preschool","Some-college")))))))

table(train2$education)
table(train3$education)
table(test3$education)
str(train3)
train3$education=as.factor(train3$education)
test3$education=as.factor(test3$education)

table(train3$type_employer)
train3$type_employer= revalue(train3$type_employer,c("Federal-gov"="Govt","Local-gov"="Govt","State-gov"="Govt"))
test3$type_employer= revalue(test3$type_employer,c("Federal-gov"="Govt","Local-gov"="Govt","State-gov"="Govt"))

model= glm(income~.-education_num-fnlwgt, data= train3, family="binomial")
summary(model)
# AIC of 14534
par(mfrow=c(2,2))
plot(model)

pred_1= predict(model,type="response")
prob_1= prediction(pred_1,train3$income)
auc= performance(prob_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.49%
pred= predict(model, newdata=test3, type= "response" )
prob= prediction(pred, test3$income)
perf= performance(prob,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob, measure="auc")
auc@y.values[[1]]
# AUC is 90.7


val= ifelse(pred_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train3$income)

# Train accuracy of 84.73


val= ifelse(pred>0.5, ">50K", "<=50K")
confusionMatrix(val, test3$income)

# Test accuracy of 84.71


model2= stepAIC(model)
summary(model2)
# AIC of 14534
par(mfrow=c(2,2))
plot(model2)

pred2_1= predict(model2,type="response")
prob2_1= prediction(pred2_1,train1$income)
auc= performance(prob2_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.49%
pred2= predict(model2, newdata=test3, type= "response" )
prob2= prediction(pred2, test3$income)
perf2= performance(prob2,"tpr","fpr")
par(mfrow=c(1,1))
plot(perf2, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(prob2, measure="auc")
auc@y.values[[1]]
# AUC is 90.7


val= ifelse(pred2_1>0.5, ">50K", "<=50K")
confusionMatrix(val, train3$income)

# Train accuracy of 84.73


val= ifelse(pred2>0.5, ">50K", "<=50K")
confusionMatrix(val, test3$income)

# Test accuracy of 84.71



## 9. Model: D
# Ridge and Lasso Regularization 

train4=train3
test4=test3

str(train)
dummies <- dummyVars(income~., data = train4)
x.train=predict(dummies, newdata = train4)
y.train=train4$income
x.test = predict(dummies, newdata = test4)
y.test = test4$income
str(x.train)
x.train
View(x.train)
summary(x.train)
fit.lasso <- glmnet(x.train, y.train, family="binomial", alpha=1)
fit.lasso
fit.lasso.cv <- cv.glmnet(x.train, y.train, type.measure="auc", alpha=1, 
                          family="binomial",nfolds=10,parallel=TRUE)
par(mfrow=c(1,1))
plot(fit.lasso, xvar="lambda")
plot(fit.lasso.cv)
fit.lasso.cv$lambda.min
coef(fit.lasso.cv,s = fit.lasso.cv$lambda.min)
pred.lasso.cv.train <- predict(fit.lasso.cv,x.train,s = fit.lasso.cv$lambda.min,type="response")
pred.lasso.cv.test <- predict(fit.lasso.cv,x.test,s = fit.lasso.cv$lambda.min,type="response")
prob_1= prediction(pred.lasso.cv.train,y.train)
auc= performance(prob_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.51%
predval= prediction(pred.lasso.cv.test, test4$income)
perf= performance(predval,"tpr","fpr")
plot(perf, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(predval, measure="auc")
auc@y.values[[1]]

# AUC is 90.7%


val= ifelse(pred.lasso.cv.train>0.5, ">50K", "<=50K")
confusionMatrix(val, train4$income)

# Train accuracy of 84.77


val= ifelse(pred.lasso.cv.test>0.5, ">50K", "<=50K")
confusionMatrix(val, test4$income)

# Test accuracy of 84.74


fit.ridge <- glmnet(x.train, y.train, family="binomial", alpha=0)
fit.ridge
fit.ridge.cv <- cv.glmnet(x.train, y.train, type.measure="auc", alpha=1, 
                          family="binomial",nfolds=10,parallel=TRUE)
par(mfrow=c(1,1))
plot(fit.ridge, xvar="lambda")
plot(fit.ridge.cv)
fit.ridge.cv$lambda.min
coef(fit.ridge.cv,s = fit.ridge.cv$lambda.min)
pred.ridge.cv.train <- predict(fit.ridge.cv,x.train,s = fit.ridge.cv$lambda.min,type="response")
pred.ridge.cv.test <- predict(fit.ridge.cv,x.test,s = fit.ridge.cv$lambda.min,type="response")
prob_1= prediction(pred.ridge.cv.train,y.train)
auc= performance(prob_1,measure="auc")
auc@y.values[[1]]
# AUC is 90.5
predval= prediction(pred.ridge.cv.test, test4$income)
perf= performance(predval,"tpr","fpr")
plot(perf, col=rainbow(10),colorize=T,print.cutoffs.at=seq(0,1,0.05))
auc=performance(predval, measure="auc")
auc@y.values[[1]]
# AUC is 90.69%


val= ifelse(pred.ridge.cv.train>0.5, ">50K", "<=50K")
confusionMatrix(val, train4$income)

# Train accuracy of 84.79


val= ifelse(pred.ridge.cv.test>0.5, ">50K", "<=50K")
confusionMatrix(val, test4$income)

# Test accuracy of 84.75

#### NAIVE BAYES

## We are going to use the census data. 
## The calculation of probability requires us to convert variable to categories. 
## For this, we will bin the numerical variables, and try to make predictions on the 
## data thus formed. 

## Data exploration and data understanding has already been carried out. 


## 1. Data conversion to categorical

sum(is.na(cenkNN))
set.seed(123)
cenkNN= knnImputation(census,k=3)
cenkNN2= cenkNN
View(cenkNN)
str(cenkNN)
summary(cenkNN)

cenkNN$age= ifelse(cenkNN$age>16 & cenkNN$age<29,1,
                   ifelse(cenkNN$age>28 & cenkNN$age<38,2,
                          ifelse(cenkNN$age>37 & cenkNN$age<49,3,4)))
# Binning the age variable accourding to the quartiles

cenkNN$age= as.factor(cenkNN$age)
table(cenkNN$age)
cenkNN$fnlwgt= ifelse(cenkNN$fnlwgt>12284 & cenkNN$fnlwgt<117833,1,
                   ifelse(cenkNN$fnlwgt>117832 & cenkNN$fnlwgt<178364,2,
                          ifelse(cenkNN$fnlwgt>178363 & cenkNN$fnlwgt<237056,3,4)))

table(cenkNN$fnlwgt)
cenkNN$fnlwgt= as.factor(cenkNN$fnlwgt)

cenkNN$education_num= as.factor(cenkNN$education_num)

cenkNN$capital_gain= ifelse(cenkNN$capital_gain>0,1,0)
cenkNN$capital_loss= ifelse(cenkNN$capital_loss>0,1,0)
cenkNN$capital_gain= as.factor(cenkNN$capital_gain)
cenkNN$capital_loss= as.factor(cenkNN$capital_loss)

str(cenkNN)

cenkNN$hr_per_week= ifelse(cenkNN$hr_per_week>0 & cenkNN$hr_per_week<41,1,
                      ifelse(cenkNN$hr_per_week>40 & cenkNN$hr_per_week<46,2,3))
table(cenkNN$hr_per_week)
cenkNN$hr_per_week= as.factor(cenkNN$hr_per_week)

str(cenkNN)

## 2. Train- Test Split 

set.seed(123)
rows1= sample.split(cenkNN$income, 0.7)
train1= cenkNN[rows1==T,]
test1= cenkNN[rows1==F,]
dim(train1)
dim(test1)


## 3. Naive Bayes model building 

table(train1$income)
table(test1$income)
model= naiveBayes(income~., data=train1)
model
summary(model)
pred0= predict(model,newdata=train1)
confusionMatrix(pred0, train1$income,positive = ">50K")

# 81.32% train accuracy

pred= predict(model, newdata=test1)
confusionMatrix(pred, test1$income,positive = ">50K")

# 81.75% test accuracy


