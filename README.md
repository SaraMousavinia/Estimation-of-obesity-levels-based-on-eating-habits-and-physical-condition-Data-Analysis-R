# Estimation-of-obesity-levels-based-on-eating-habits-and-physical-condition-Data-Analysis-R
1 Motivation

Obesity is a chronic disease that is common and serious and linked to an increased risk of chronic diseases such as type 2 diabetes, heart disease, and some forms of cancer. It is, however, strongly related to a variety of risk factors that are challenging to identify. In this regard, analyzing and studying to identify these risk factors has the potential to provide greater insight into their relationships with health outcomes. The purpose of this project was to apply different predictive models based on health factors to predict the level of obesity.

2	Source of the data

The dataset of “Estimation of obesity levels based on eating habits and physical condition Data Set” include data for the estimation of obesity levels in individuals from the countries of Mexico, Peru and Colombia. It contains 17 attributes, 2111 records, and a categorical class variable.<https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition>[1]

3 Details of the dataset

The dataset contains 6 predictors related to eating habits, 5 predictors related to responders’ characteristics, and 5 predictors related to physical conditions. The output is a categorical variable in 7 levels: Insufficient Weight, Normal Weight,Overweight Level I,Overweight Level II,Obesity TypeI,Obesity TypeII and Obesity TypeIII

I analyzed the dataset with several regression and classification models but classification models outperformed.

These are the models I applied:
Multinomial Regression | LDA | KNN | Random Forest | Boosted Tree | SVM Linear | SVM Poly

Which I screipted by CARET package in R

[1] Palechor, F. M., & de la Hoz Manotas, A. (2019). Dataset for estimation of obesity levels based on eating habits and physical condition in individuals from Colombia, Peru and Mexico. Data in Brief, 104344
