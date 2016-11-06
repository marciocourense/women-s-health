# Import library nnet. 
# Please keep in mind, when you implement your model in Azure ML, you need to also import this library in your Execute R Script module.
# Otherwise, you will encounter error when you use the model to predict you target

library(nnet)
install.packages("dplyr")
library(dplyr)
install.packages("Amelia")
library(Amelia)
library(ggplot2)
# Specify the URL of data. 
# Also specify the rda file that you want to use to save the model
dataURL <- 'http://az754797.vo.msecnd.net/competition/whra/data/WomenHealth_Training.csv'

# Read data to R workspace. The string field religion is read as factors
colclasses <- rep("integer",50)
colclasses[36] <- "character"
dataset1 <- read.table(dataURL, header=TRUE, sep=",", strip.white=TRUE, stringsAsFactors = F, colClasses = colclasses)
summary(dataset1)

# Combine columns geo, segment, and subgroup into a single column. 
#the line below was already provided by Bill Gates foundation, this will be the target collumn
combined_label <- 100*dataset1$geo + 10*dataset1$segment + dataset1$subgroup
data.set <- cbind(dataset1, combined_label)
data.set$combined_label <- as.factor(data.set$combined_label)

# Skip the columns patientID, segment, subgroup, and INTNR from the feature set
ncols <- ncol(data.set)
feature_index <- c(2:18, 20:(ncols-3))

#######################
#### Analysis #########
#######################
#check for missing values
missmap(data.set, col = c("yellow","black"))

#CLEANING MISSING DATA
#First: set up easy reliable rules to deal with blank values

#!!!“ever been pregnant” ---- ATTENTION a women can be pregnant but not gave labor (e.g.: abortion/fetus died)
#alternative method
library("lattice")
densityplot(~age,data=data.set, groups = na.omit(data.set$LaborDeliv),plopoints= F, lwd = 3,col=c("blue","red"))
mosaicplot(data.set$LaborDeliv ~ data.set$religion)
mosaicplot(data.set$geo ~ data.set$LaborDeliv)

#The sentence below decribes that the data was not retrieve by professional staticians
#"The data for this competition was collected from around 9000 young (15 to 30 years old) woman subjects when they visited clinics in 9 underdeveloped regions, with around 1000 subjects in each region. Each subject was asked by clinical practitioners some questions and her answers were recorded, together with her demographic information".
#source: "https://gallery.cortanaintelligence.com/Competition/Womens-Health-Risk-Assessment-1"
#therefore some basic validation is necessary

#check for the missing values on Debut
densityplot(data.set$Debut[is.na(data.set$Debut)] ~ data.set$age,plopoints= F, lwd = 3,col=c("blue","red"))



#finally cleaning the LAborDeliv data
data.set$LaborDeliv[is.na(data.set$LaborDeliv) & data.set$EVER_BEEN_PREGNANT == 0] <- 0
data.set$babydoc[is.na(data.set$babydoc) & data.set$EVER_BEEN_PREGNANT == 0] <- 0

#girls who were pregnat, but never had sex
#it might be common to answer "no" due to fear of persecution, more input is needed


#fill NA on literacy, based on education
mosaicplot(data.set$literacy ~ data.set$educ)


#one outlier never had sex, but used condom
data.set$EVER_HAD_SEX[is.na(data.set$EVER_HAD_SEX) & data.set$usecondom == 1] <- "1"

#When reading the description of Modcon they are only relevant if the subject is sexually active
data.set$EVER_HAD_SEX[is.na(data.set$EVER_HAD_SEX) & data.set$ModCon == 1] <- "1"


#clean missing data: multpart
data.set$multpart[is.na(data.set$EVER_HAD_SEX == 0)] <- "0"


#clean missing data: babydoc
#this variable is not necessarely linked to the biological motehr (e.g.: war orfans)

###### Despite massive cleaning some NA values will still prevail
# Clean missing data by replacing missing values with 0 (with "0" for string variable religion)
data.set[is.na(data.set)] <- 0
data.set[data.set$religion=="", "religion"] <- "0"
data.set$religion <- factor(data.set$religion)
data.set$combined_label <- relevel(data.set$combined_label, ref = '111')

###################################################

# Split the data into train (75%) and validation data (25%)
nrows <- nrow(data.set)
sample_size <- floor(0.75 * nrows)
set.seed(98052) # set the seed to make your partition reproductible
train_ind <- sample(seq_len(nrows), size = sample_size)

train <- data.set[train_ind, ]
validation <- data.set[-train_ind, ]

###################################
#### Application of the model #####
###################################

# First select the model
#Possibilities considered : decision trees, random forest, XGBoost, deep learning 
#this step was done in Azure ML, due to the fact that is possible find the best hyperparameters
#and do multiple models in paralalel and compre the results

# Predict the validation data and calculate the accuracy
predicted_labels <- predict(model, validation)
accuracy <- round(sum(predicted_labels==validation$combined_label)/nrow(validation) * 100)
print(paste("The accuracy on validation data is ", accuracy, "%", sep=""))

