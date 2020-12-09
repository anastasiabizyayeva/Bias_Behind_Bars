# Bias Behind Bars

This project is an exploration of the Tom Cardoso's Globe and Mail article titled ['Bias Behind Bars'](https://www.theglobeandmail.com/canada/article-investigation-racial-bias-in-canadian-prison-risk-assessments/), published on October 24th, 2020. Cardoso found that the subjective assessments being used to determine reintegration potential scores for offenders were outdated, and largely created based on information about white offenders. As a result, there are calls for the CSC to revisit these assessments and rebuild them to account for the differing experiences of the variety of demographics served by the CSC.

In this exploration, I sought to look through the CSC data and put together a classification model that could determine the reintegration potential of offenders, and see whether there was racial bias in the system. I reached the following conclusions:

* The final model had an accuracy score of 0.826 and an ROC AUC of 0.941.
* I created two test sets - one for white offenders and the other for non-white offenders. The accuracy of the model was 5.1% greater for white offenders than non-white offenders.

Ultimately, what I found is that the majority of offender information pertained to white inmates, and thus predictions of likelihood to reintegrate into the community were more accurate for that demographic, but not suitable for use on non-white offenders. As a result, the calculation of reintegration potential scores for non-white offenders needs to be reformed. 

## Code and Resources Used

**Python Version:** 3.9

**Packages:** pandas, numpy, sklearn, matplotlib, seaborn, missingno, xgboost

**Helpful Articles:**
* ['Annotating Seaborn'](https://stackoverflow.com/questions/43214978/seaborn-barplot-displaying-values)
* ['Handing NaNs in Categorical Data'](https://medium.com/analytics-vidhya/best-way-to-impute-categorical-data-using-groupby-mean-mode-2dc5f5d4e12d)
* ['Strategies for Working with Continuous Numeric Data'](https://towardsdatascience.com/understanding-feature-engineering-part-1-continuous-numeric-data-da4e47099a7b)
* ['Ordinal Encoding'](https://towardsdatascience.com/all-about-categorical-variable-encoding-305f3361fd02)
* ['Confusion Matrix'](https://medium.com/analytics-vidhya/multi-class-ml-models-evaluation-103c9fdadb41)
* ['Randomized & Grid Search for Random Forest'](https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74)


## Dataset

The data consisted of one record for every offender managed by Correctional Services of Canada (CSC). The  data were extracted from the Offender Management System (OMS) and reflect the status and attributes  of offenders as of 2012-2013 fiscal year end and 2013-2014 fiscal year end. It contained roughly 750k records for roughly 50k offenders. 

The data had the following columns:
* **Number** – Unique record identifier 
* **Race** – This corresponds to the race/ethno-cultural background of the offender. This is voluntary  information that is self reported by the offender at the time the offender is admitted to CSC. 
* **Race Grouping** – This summarizes the Race into two categories. “Aboriginal” corresponds to offenders  with a Race of “First Nations”, “Métis”, or “Inuit”. “Non Aboriginal” corresponds to all others. 
* **Gender** – This corresponds to the gender of the offender at the time the data were extracted. 
* **Age** – This is the age of the offender, in years, at the time the data were * extracted. 
* **In Custody/Community** – This identifies if the offender is in custody in a federal institution or supervised in  the community on conditional release. Offenders away from a federal institution on a temporary absence  are considered to be “In Custody”. 
* **Supervision Type** – This identifies the type of supervision for the supervised offenders. DP = day parole,  FP = full parole, SR = statutory release, LTSO = long term supervision period, and RES = residency  conditions. 
* **Jurisdiction** – This identifies if the offender is serving a federal sentence (2 years or more) or a provincial  sentence (less than two years). 
* **Sentence Type** – This corresponds to the type of sentence imposed by the courts. Determinate  sentences have a set expiry date. Indeterminate sentences never expire. 
* **Aggregate Sentence Length** – This is the length of the sentence imposed by the courts expressed in  days.
* **Institutional Security Level** – This corresponds to the security level of the institution where the offender’s  case is being managed. 
* **Province** – This corresponds to the province where the offender’s case file is being managed. 
* **Location Type** – This corresponds to the type of facility where the offender’s case file is being managed. 
* **Offender Security Level** – This corresponds to the results of the last Offender Security Level decision  records for the offender at the time the data were extracted. 
* **Dynamic/Need** – This corresponds to the offender’s need for intervention, as identified in the last dynamic factors evaluation.
* **Static/Risk** – This corresponds to the risk level of the offender, as identified in the last static factors evaluation. 
* **Reintegration Potential** – This corresponds to the results of the assessment of the offender’s ability to  reintegrate into the community without reoffending, as identified in the last Correctional Plan or Correctional Plan Progress Report. 
* **Motivation** – This corresponds to the degree of the offender’s commitment to his or her correctional plan,  as identified in the last Correctional Plan or Correctional Plan Progress Report. 
* **Major Offence Group** – This is the type of offence considered the most serious on the offender’s current  sentence. 
* **Religion** - This corresponds to the religion of the offender. This voluntary information that is self reported  by the offender at the time the offender is admitted to CSC.

## Data Cleaning

During the data cleaning stage the following adjustments were made: 

*	Dropped all records that didn't have a race 
* Dropped records with a sentencel length less than 0 days
* Dropped the 'Judge' column since it was redacted 
* Corrected a spelling error in the columns 
* Converted the 'fiscal year' column to a more readable format 
* Bucketed the races into four groups: white, black, indigenous, and other
* Converted sentence length in days to sentence length in years 

## EDA

|![alt text](https://github.com/anastasiabizyayeva/Bias_Behind_Bars/blob/master/images/offenders_gender.JPG "Gender of Offenders")|![alt text](https://github.com/anastasiabizyayeva/Bias_Behind_Bars/blob/master/images/supervision_type.JPG "Supervision Type")|![alt text](https://github.com/anastasiabizyayeva/Bias_Behind_Bars/blob/master/images/sentence_length.JPG "Sentence Length")|
| ------------- |:-------------:| -----:|

## Feature Engineering 

During the feature engineering stage we worked through the following steps:

* Examined and visualized missing data.
* Determined where it's appropriate to drop records, and the right way to impute missing values for remaining records.
* Binned some of our numeric features.
* Condensed our offence descriptions.
* OneHot & ordinal encoded our variables.
* Reshaped our dataframe so that each record is an inmate in a particular year.

|![alt text](https://github.com/anastasiabizyayeva/Bias_Behind_Bars/blob/master/images/missingno.JPG "Missingno")|![alt text](https://github.com/anastasiabizyayeva/Bias_Behind_Bars/blob/master/images/corr_map.JPG "Heatmap")|![alt text]|
| ------------- |:-------------:| 

## Model Building

During the model-building phase, our goals were two-fold: first ,we wanted to put together a model that had a good accuracy score for predicting reintegration potential. Second, we wanted to explore whether there was a difference in the accuracy scores for white vs. non-white offender test sets. 

|![alt text](https://github.com/anastasiabizyayeva/Bias_Behind_Bars/blob/master/images/rf_cm.JPG "Random Forest")|
| ------------- |

In order to achieve this, we took the following steps:

* Carved out the white offenders for most of our modelling and split this dataset into training, validation, and test sets.
* The remaining data, composed of non-white offenders, was to be a secondary test set - our expectation was that if there's no racial bias in the dataset, our models should have perform identically on our white and non-white test sets.
* Fit our data to the following models:
  * **Multinomial Logistic Regression - OvR:** Baseline model.
  * **K-Nearest Neighbors:** Helpful since it's a nonparametric model. Since we had a lot of data and don't want to worry too much about choosing just the right features this was a good addition to the exploration.
  * **Random Forest:** Usually more accurate than decision trees and doesn't tend to overfit.
  * **XGBoost:** Comparatively faster than other ensemble classifiers.
* Tested the accuracy score and ROC AUC for all classifiers to and determined Random Forest was the most accurate. 
* Tuned hyperparameters with GridSearchCV for most successful model.
* Retrain most successful model on a combination of training and validation data, and then predict on both of our test sets (white and non-white offenders).

## Conclusion 

There was a dramatic difference in the two accuracy scores, with our model achieving an accuracy score 5.1% higher on the white offender test set than on the non-white offender test set. This suggests that there is indeed racial bias in determining the reintegration potential of offenders. 
