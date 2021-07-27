---
  title: "MovieLens Report"
author: "Nkonzo Sithole"
date: "2/17/2021"
output:
  pdf_document: default
html_document:
  df_print: paged
word_document: default
---
  
  #Introduction
  
  #Environment preparation
  ```{r}
knitr::opts_chunk$set(echo = TRUE, fig.align = 'center', cache=FALSE, cache.lazy = FALSE)
```
##Install and load libraries

#**installing**
```{r}
if(!require(tidyverse)) install.packages("tidyverse") 
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(tidyr)) install.packages("tidyr")
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(stringr)) install.packages("stringr")
if(!require(forcats)) install.packages("forcats")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(lubridate)) install.packages("lubridate")
if(!require(caret)) install.packages("caret")
```
##**Loading**
```{r echo=FALSE}
library(dplyr)
library(tidyverse)
library(kableExtra)
library(tidyr)
library(stringr)
library(forcats)
library(ggplot2)
library(lubridate)
library(caret)

```
#*Data* *preperation*
##Create test set, validation set from movieLens that has 10M records in the dataset
```{r}

dl <- tempfile()
set.seed(1)
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#Create data frames

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")
head(movielens)
#Total records
nrow(movielens$genres)
#distinct genres
n_distinct(movielens$genres)

```
# Split movielens data, Validation set will be 10% of MovieLens data

```{r}
set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
test <- movielens[-test_index,]
train <- movielens[test_index,]
# sami_join will ensure that userId and movieId are in validation dataset
validation <- train %>% 
  semi_join(test, by = "movieId") %>%
  semi_join(test, by = "userId")
# Add rows removed from validation set back into test set
removed <- anti_join(train, validation)
test <- rbind(test, removed)
```

## Data Summary and Explortory Data Analysis 

```{r}
str(test)
summary(test)
#from the summary of data we see that the rating is between 1 to 5, with mean of the rating of 3.512 and the mode is 4.0.

test %>% group_by(rating) %>% summarize(count = n()) %>% top_n(5) %>%
  arrange(desc(count))  

test %>% 
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))


test %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "light blue",bins = 30 , binwidth = 0.2)+
  scale_x_log10()+
  ggtitle(" number of Rating per Movie")+
  theme_gray()

# Some movies are not rated and some are rated about 1000 times.
```

#*Predictions**

```{r}
##Create RMSE function that will be used in our models, where y_hat vector is for predicted ratings
RMSE <- function(true_ratings, y_hat){
  sqrt(mean((true_ratings - y_hat)^2))
}
```

##**First** **Model**

```{r}
#Creating first model for the predicted ratings driven by avarage ratings only
y_hat <- mean(test$rating)

rmse_m1 <- RMSE(test$rating,y_hat)
cat("RMSE from Model 3: ", rmse_m1)
```
##**Second** **Model**

```{r}
#Second model driven by difference between ratings and avarage ratings b_i
mu <- mean(test$rating)
movie_avgs <- test %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

#b_i histogram graph looks normal 
movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

predicted_ratings <- mu + test %>% 
  left_join(movie_avgs, by='movieId') %>%
  pull(b_i)

rmse_m2 <- RMSE(predicted_ratings, test$rating)
cat("RMSE from Model 2: ", rmse_m2)
```

#**Third** **Model**
```{r}
#Visualising b_i based on data grouped by userID
train %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  filter(n()>=100) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

#finding b_u, the difference from on rating with mean and b_i
user_avgs <- train %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

#finding predicted_ratings, the sum of b_u , mean and b_i
predicted_ratings <- test %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)

rmse_m3 <- RMSE(predicted_ratings, test$rating)

## RMSE of the validation set

valid_pred_rating <- validation %>%
  left_join(movie_avgs , by = "movieId" ) %>% 
  left_join(user_avgs , by = "userId") %>%
  mutate(pred = mu + b_i + b_u ) %>%
  pull(pred)

rmse_m3_final <- RMSE(validation$rating, valid_pred_rating)
cat("RMSE from Model 3: ", rmse_m3_final)

```

**All models results**
  
  ```{r}
cat("RMSE from Model 1: ", rmse_m1)
cat("RMSE from Model 2: ", rmse_m2)
cat("RMSE from Model 3: ", rmse_m3_final)
```

**Conclusion**
  
  ```{r}
#We can see that the RMSE improved when ratings per user are considered, Although we must note the possibility of overfitting. Regularisation with many fold can be considered.
```
