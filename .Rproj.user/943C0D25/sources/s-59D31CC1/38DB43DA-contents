---
title: "Employee Attrition Report"
author: "Nkonzo Sithole"
date: "7/26/2021"
output:
  pdf_document: default
html_document:
  data_print: paged
word_document: default
---
  
1. **Introduction**
  #Hiring and retaining top talent is an extremely challenging task that requires capital, time and skills
  #Employee attrition generally has negatively impact to many companies. It is important for HR to identifying employees who are most likely to leave company and prepare or avoid for such loss.
  
  #For example, studies found that staff churn is correlated with both demographic information as well as behavioral activities, satisfaction, etc. 
  #Machine learning models or techniques can give better prediction on employee attrition, as by nature they mathematically model the correlation between factors and attrition outcome and maximize 
  #the probability of predicting the correct group of people with a properly trained machine learning model.
  #In the data-driven employee attrition prediction model, normally two types of data are taken into consideration. 
  
  #---Demographic and organizational information of an employee such as *age*, *gender*, *title*, etc. This type does not change.
  #---Second type of data is the dynamically involving information about an employee. In this study, (https://towardsdatascience.com/employee-retention-using-machine-learning-e7193e84bec4), they are looking at the cause of such leaving.
  # I will use the data from https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset


#**installing**
```{r}
if(!require(dplyr)) install.packages("dplyr") 
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(RColorBrewer)) install.packages("RColorBrewer")
if(!require(plotrix)) install.packages("plotrix")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(caret)) install.packages("caret")
if(!require(corrplot)) install.packages("corrplot")
if(!require(corrgram)) install.packages("corrgram")
if(!require((grid)) install.packages("grid")
if(!require(gridExtra)) install.packages("gridExtra")
```
2. **Data exploration**

The experiments will be conducted on a data set of employees. The data set is publicly available and can be found at https://www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset.
```{r}
data <- read.csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
head(data)
```
##summary
```{r}
summary(data)
```
##Fix Age column
```{r}
colnames(data)[1] <- "Age"
```
## Check missing data
```{r}
apply(is.na(data), 2, sum)
```

## dataset is made up of the following rows and columns
```{r}
str(data)
cat("Data Set has ",dim(data)[1], " Rows and ", dim(data)[2], " Columns" )
```
## no missing and duplicate values
```{r}
sum(is.na(duplicated(data)))
```
3. **Data Visualization**

```{r}
data %>%
  group_by(Attrition) %>%
  tally() %>%
  ggplot(aes(x = Attrition, y = n,fill=Attrition)) +
  geom_bar(stat = "identity") +
  theme_minimal()+
  labs(x="Attrition", y="Count of Attriation")+
  ggtitle("Attrition")+
  geom_text(aes(label = n), vjust = -0.5, position = position_dodge(0.9))
```

b) Checking employees status(attrition) per job title
```{r}
ggplot(data, aes(JobRole, fill=Attrition)) +
  geom_bar(aes(y=(..count..)/sum(..count..)), position="dodge") +
  xlab("Job Role") +
  ylab("Percentage")
```

c) Income, jobRole, previous percentage salary hike and service years may affect decision for employees to leave. 

```{r}
ggplot(filter(data, (PercentSalaryHike >= 11) & (YearsAtCompany >= 2) & (YearsAtCompany <= 5) & (JobLevel < 3)),
       aes(x=factor(JobRole), y=MonthlyIncome, color=factor(Attrition))) +
  geom_boxplot() +
  xlab("Department") +
  ylab("Monthly income") +
  scale_fill_discrete(guide=guide_legend(title="Attrition")) +
  theme_bw() +
  theme(text=element_text(size=13), legend.position="top")
```

d) Employees grid graph in relation with Years of service, Growth, Manager, Income and salary increase.

EmployeesYearOfService <- ggplot(data,aes(YearsAtCompany,fill = Attrition))+geom_bar()
EmployeesGrowth <- ggplot(data,aes(YearsSinceLastPromotion,fill = Attrition))+geom_bar()
EmployeesManager <- ggplot(data,aes(YearsWithCurrManager,fill = Attrition))+geom_bar()
EmployeeSalIncrease <- ggplot(data,aes(PercentSalaryHike,Attrition))+geom_point(size=4,alpha = 0.01)
EmployeesIncome <- ggplot(data,aes(MonthlyIncome,fill=Attrition))+geom_density()
grid.arrange(EmployeesYearOfService,EmployeesGrowth,EmployeesManager,EmployeeSalIncrease,EmployeesIncome,ncol=2,top = "Grid graphs")

```
## data correlation

## remove near zero variables

```{r}
near_Zero_variables <- names(data[, nearZeroVar(data)]) %>% print()
data <- data %>% select(-one_of(near_Zero_variables))
```
corrgram(data, lower.panel = panel.shade, upper.panel = panel.pie, text.panel = panel.txt, main = "Corrgram of all numeric  variables")

From this,  I will use algorithms like rf or XGBoost to build a model that can predict in fact which employees are most likely to leave in the future 

4. **Data Preparation and Partitioning** 

## convert certain integer variable to factor variable.

```{r}
factor_variables <- c("Education", "EnvironmentSatisfaction", "JobInvolvement", "JobLevel", "JobSatisfaction", "NumCompaniesWorked", "PerformanceRating", "RelationshipSatisfaction", "StockOptionLevel")
data[, factor_variables] <- lapply((data[, factor_variables]), as.factor)
data <- data %>% mutate_if(is.character, as.factor)
str(data)
```

Before modeling, first I use `set.seed(1)` and partition my data into train and test sets, which will be used to model and produce predictions. 
Then towards the end of this report, I will show the final model performance on the validation set. 

```{r}
set.seed(1)
train_index <- createDataPartition(data$Attrition , times =1, p = 0.7, list = FALSE)
train <- data[train_index,] 
test <- data[-train_index,]
```
5. **Modeling, Tuning & Evaluation**

##training control to tune 
control <- trainControl(method="repeatedcv", number=3, repeats=1)
```{r}
#random forest model 
random_forest_model <- train(dplyr::select(data, -Attrition), 
                             data$Attrition,
                             data=train, 
                             method="rf", 
                             preProcess="scale", 
                             trControl=control)

prediction_rfm <- predict(random_forest_model, newdata=select(test, -Attrition))
confusionMatrix(prediction_rfm,reference=test$Attrition,positive="Yes")

imp <- varImp(random_forest_model, scale=FALSE)
```
6. **Conclusion**
We can see that Salary has big impact in employees attriction
plot(imp)



