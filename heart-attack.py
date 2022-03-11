# The data of interest can be found at the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) website. 

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.gam.api import BSplines
from statsmodels.gam.generalized_additive_model import LogitGam
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, make_scorer, accuracy_score
from sklearn.ensemble import AdaBoostClassifier as AdaBoost, GradientBoostingClassifier as GradBoost, RandomForestClassifier as Forest
from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB



#import and clean dataset
df = pd.read_csv("data/heart.csv")
df.info()


print("Number of null values \n {}".format(df.isnull().sum().sum()))
print("\nNumber of na values \n {}".format(df.isna().sum().sum()))


# In[4]:


data = df
data['sex'].replace(1, 'male', inplace=True)
data['sex'].replace(0, 'female', inplace=True)

data['cp'].replace(0, 'typical angina', inplace=True)
data['cp'].replace(1, 'atypical angina', inplace=True)
data['cp'].replace(2, 'non-anginal pain', inplace=True)
data['cp'].replace(3, 'asymptomatic', inplace=True)

data['restecg'].replace(0, 'normal', inplace=True)
data['restecg'].replace(1, 'ST-T wave abnormality', inplace=True)
data['restecg'].replace(2, 'possible hypertrophy', inplace=True)

data['ca'].replace(0, '0', inplace=True)
data['ca'].replace(1, '1', inplace=True)
data['ca'].replace(2, '2', inplace=True)
data['ca'].replace(3, '3', inplace=True)
data['ca'].replace(4, '4', inplace=True)

data['thal'].replace(1, 'normal' , inplace=True)
data['thal'].replace(2, 'fixed defect', inplace=True)
data['thal'].replace(3, 'reversable defect', inplace=True)
data['thal'].replace(0, '0', inplace=True)


data['fbs'].replace(0, 'False', inplace=True)
data['fbs'].replace(1, 'True', inplace=True)

data['slope'].replace(0, 'upsloping', inplace=True)
data['slope'].replace(1, 'flat', inplace=True)
data['slope'].replace(2, 'downsloping', inplace=True)

data['exang'].replace(0, 'no', inplace=True)
data['exang'].replace(1, 'yes', inplace=True)


# In[5]:


conts = {
    'chol' : 'Cholesterol',
    'age' : 'Age',
    'trestbps' : 'Resting_Blood_Pressure',
    'oldpeak' : 'ST_depression',
    'thalach' : 'Maximum_Heart_Rate_Achieved'
}
data.rename(columns=conts, inplace=True)
notCats = conts.values()

catTitles = {
    'sex': 'Sex', 
    'cp': 'Chest_Pain_Type', 
    'restecg': 'Resting_Electrocardiographic_Results', 
    'thal': 'Thalassemia', 
    'fbs': 'fasting_blood_sugar_greater_120', 
    'slope': 'slope_of_the_peak_exercise_ST_segment', 
    'exang': 'exercise_induced_angina',
    'ca': 'Number_of_major_vessels_colored_by_flourosopy',
}
data.rename(columns=catTitles, inplace=True)
    
cats = catTitles.values()
data.info()


# Bar charts and histograms are ideal for identifying missing values and outliers.

# In[6]:


#set ideal size of bar plots
plt.figure(figsize=(20,20))
plt.rc('font', size=10)

#create bar plots of distributions of features
#display them in 5 row by 3 column grid
for i in range(len(data.columns)):
    col = df.columns[i]
    plt.subplot(5, 3, i+1)
    plt.hist(data[col])
    plt.title('{}'.format(col))
    
plt.show()



#Thalassemia = 0 represents less than 5 percent of records. Complete Case Analysis allows us to remove this
print("Proportion of records with Thalassemia = 0: ", sum(data['Thalassemia'] == '0') / data.shape[0])


# In[7]:


data.drop(data['Thalassemia'][data['Thalassemia'] == '0'].index, inplace=True)


# The level, possible hypertrophy, in the Resting_Electrocardiographic_Results column is somewhat rare in the dataset. Therefore, it should be combined with ST-T wave abnormality to form a new level, abnormal. This should improve model training times and model performance. 

# In[8]:


#combine into one level for convenience and clarity
data['Resting_Electrocardiographic_Results'].replace('ST-T wave abnormality', 'abnormal', inplace=True)
data['Resting_Electrocardiographic_Results'].replace('possible hypertrophy', 'abnormal', inplace=True)


# Split the dataset into a features set and a target set.

# In[9]:


#Split the dataset into a features set and a target set.
X = data[[col for col in data.columns if col != "target"]]
y = data['target']

X.describe()


# In[10]:


#normalize/standardize all numerical feature values. This helps with interpretability and model performance.
for cont in notCats:
    X[cont] = (X[cont] - X[cont].mean()) / X[cont].std() 


# In[11]:


#Split the data into a training and test set, where the test set contains 20% of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 23)
print(len(X_train), len(X_test))


# In[34]:


#plot a pretty coorelation matrix
seaborn.set(rc = {'figure.figsize':(15,8)})
seaborn.set(font_scale=1.4)
seaborn.heatmap(
    X_train.corr(),  
    cmap = 'coolwarm',
    linewidths=.1, 
    square=True, 
    yticklabels=True, 
    annot=True,
    center = 0
)


# In[35]:


#run statistical test to see if if there is an association
model = smf.ols(formula="Maximum_Heart_Rate_Achieved ~ Age + ST_depression", data = X).fit()
print(model.summary())


# Now let us identify some possible continuous-valued features that may influence heart attack risk.

# In[12]:


plt.figure(figsize=(20,20))
plt.rc('font', size=15)

i = 1
#plot scatter plots of numerical features vs target
#expect clusters of similar colored points at opposite ends of the specturm
#plot in 3 by 2 grid
for cont in notCats:
    plt.subplot(3, 2, i)
    plt.scatter(X_train[cont], y_train, c=y_train, cmap='coolwarm')
    plt.title("{} vs Heart Attack Risk".format(cont))
    plt.ylabel("Target")
    i += 1




plt.figure(figsize=(20,20))
plt.rc('font', size=12)
train = pd.concat([X_train, y_train], axis=1)

i = 1
#Plot bar charts for each categorical feature
#X-axis : the level
#Y-axis : the proportion of observed individuals classified as being at high risk of a heart attack
#plot in 4 by 2 grid
for col in cats:
    plt.subplot(4, 2, i)
    grp = train.groupby(col).sum().reset_index()
    counts = train.groupby(col).count().reset_index()['target']
        
    plt.bar(grp[col], grp['target'] / counts )
    plt.title('{}'.format(col))
    i+=1
    
plt.show()




#Tune hyperparameters and identiy best model using 3-fold cross validation
grid_forest = GridSearchCV(Forest(), 
             param_grid = {
              'n_estimators' : [2 ** x for x in [*range(1,6)]],
              'ccp_alpha' : [2 ** x for x in [*range(-5,5)]]
             },
            scoring = make_scorer(f1_score),
            n_jobs = 4
            ).fit(pd.get_dummies(X_train), y_train)


# In[43]:


ada_grid = GridSearchCV(AdaBoost(), 
             param_grid = {
                'base_estimator' : [Tree(max_depth = 1), Tree(max_depth = 2), Tree(max_depth = 4)],
                'n_estimators' : [2** x for x in [*range(1,6)]],
                'learning_rate' : [2** x for x in [*range(-4,2)]]
             },
            scoring = make_scorer(f1_score),
            n_jobs = 4
            ).fit(pd.get_dummies(X_train), y_train)


# In[56]:


boost_grid = GridSearchCV(GradBoost(), 
             param_grid = {
              'n_estimators' : [2 ** x for x in [*range(3,8)]],
              'learning_rate' : [2 ** x for x in [*range(-8,-4)]],
              'max_depth' : [1, 2, 4, 8, 16]
             },
            scoring = make_scorer(f1_score),
            n_jobs = 4
            ).fit(pd.get_dummies(X_train), y_train)



# In[59]:


svm_grid = GridSearchCV(SVC(), 
             param_grid = {
              'C' : [2**i for i in range(-5,5)],
              'gamma' : [2**i for i in range(-5,5)], 
              'kernel' : ['poly','rbf', 'linear', 'sigmoid']
             },
            scoring = make_scorer(f1_score),
            n_jobs = 4
            ).fit(pd.get_dummies(X_train), y_train)


# In[60]:


knn_grid = GridSearchCV(KNN(), 
             param_grid = {
                 'n_neighbors' : [*range(1,15)],
                 'weights' : ['uniform', 'distance'],
                 'p' : [1,2,3]
             },
            scoring = make_scorer(f1_score),
            n_jobs = 4
            ).fit(pd.get_dummies(X_train), y_train)



# In[61]:


#remove highly coorelated feature from Bayesian training set columns
cols = [col for col in X_train.columns if col != 'Maximum_Heart_Rate_Achieved']
seaborn.set(rc = {'figure.figsize':(15,8)})
seaborn.set(font_scale=1.4)
seaborn.heatmap(
    X_train[cols].corr(),  
    cmap = 'coolwarm',
    linewidths=.1, 
    square=True, 
    yticklabels=True, 
    annot=True,
    center = 0
)


# In[62]:


naive_pred = GaussianNB().fit(pd.get_dummies(X_train[cols]), y_train).predict(pd.get_dummies(X_test[cols]))


# In[63]:

#calculate variance inflation factors to diagnose multicolinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = X_train[notCats]
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['features'] = variables.columns
vif



# In[123]:

#train logistic model with regularization
train = pd.concat([X_train, y_train], axis=1)
pred_vars = " + ".join(col for col in X_train.columns)
formula = "target ~ {}".format(pred_vars)

alpha = np.array([0]*21) #penalty term
alpha[[-1,-2,-3,-4,-5]] = 0.1
LogR_model = smf.logit(formula, data=train).fit_regularized(alpha=alpha, maxitter=1000)
pred_lr = LogR_model.predict(X_test)




#This function returns evaluation measures, given an instance of GridSearchCV
#Printed are the best model parameters, best f1 and accuracy score and feature importances (if applicable)
def printScore(grid, name):
    print("\n{} best training score : {}".format(name, grid.best_score_))
    print("{} best params: {}".format(name, grid.best_params_))

    pred = grid.best_estimator_.predict(pd.get_dummies(X_test))
    print("{} F1 Score on testing data: {}".format(name, f1_score(y_test, pred)))
    print("{} Accuracy Score on testing data: {}".format(name, accuracy_score(y_test, pred)))
    if hasattr(grid.best_estimator_,'feature_importances_'):
        importances = grid.best_estimator_.feature_importances_
        indices = np.argsort(importances)[::-1]
        features = pd.get_dummies(X).columns

        plt.figure(figsize=(10,10))
        plt.rc('font', size=15)
        plt.title('Feature Importances')
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.show()


# In[73]:


printScore(grid_forest, "Random Forest")


# In[74]:


printScore(ada_grid, "AdaBoost")



# In[75]:


printScore(boost_grid, "Gradient Boost")



# In[78]:


printScore(svm_grid, "SVM")
printScore(knn_grid, "KNN")



# In[116]:


print("Bayes testing F1 Score ", f1_score(y_test, naive_pred))
print("Bayes testing Accuracy", accuracy_score(y_test, naive_pred))
print("\nF1 score for logistic regression {}".format(f1_score(y_test, pred_lr >= 0.5)))
print("Accuracy score {}".format(accuracy_score(y_test, LogR_model.predict(X_test) >= 0.5)))


# In[69]:

#create ROC plot
fpr, tpr, th = roc_curve(y_test, pred_lr)
plt.plot(fpr, tpr, 'k-')
plt.plot(np.arange(0, 1.1, 0.1), np.arange(0, 1.1, 0.1), 'r--')
plt.title("ROC Plot")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.text(0.1,0, "Area Under Curve (AUC) {}".format(roc_auc_score(y_test, pred_lr)))
plt.show()



# In[124]:


LogR_model.summary()


# In[101]:

#fit logistic model with interaction
formula_inter = formula + "+ Cholesterol:fasting_blood_sugar_greater_120"
LogR_inter = smf.logit(formula_inter, data=train).fit(maxitter=1000)
pred_inter = LogR_inter.predict(X_test)
print("\nF1 score for logistic regression with interaction {}".format(f1_score(y_test, pred_inter >= 0.5)))
