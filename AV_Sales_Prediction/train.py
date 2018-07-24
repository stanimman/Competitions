coding: utf-8

# In[301]:

import pandas as pd
import numpy as np
train = pd.read_csv('/Users/SKanagaraj/Documents/DataMart_AV/Train_Data_mart.csv',
)
test = pd.read_csv('/Users/SKanagaraj/Documents/DataMart_AV/Test_Data_mart.csv',
)
all_data = pd.concat((train.loc[:,'Item_Identifier':'Outlet_Type'], test.loc[:,'Item_Identifier':'Outlet_Type']))


# In[302]:

all_data.columns


# In[303]:

#Split Apply Conbine : A killer Function , Need Hands on grip on this
#http://pbpython.com/pandas_transform.html
all_data["Imputed_Weight"] = all_data.groupby('Item_Identifier')["Item_Weight"].transform('mean')


# In[304]:

#loc is use to select subset of row based on the label/list of label/Boolen condition
#df.loc[,]
#https://www.shanelynn.ie/select-pandas-dataframe-rows-and-columns-using-iloc-loc-and-ix/
all_data.loc[all_data['Outlet_Size'].isnull(),'Outlet_Size'] = 'Small'


# In[305]:

# Creating Meaningfull Feature
all_data['Operation_Duration'] = 2013 - all_data['Outlet_Establishment_Year']


# In[306]:

all_data['MRP_Visibility'] = all_data['Item_Visibility'] * all_data['Item_MRP']


# In[307]:

#all_data.loc[all_data['Item_Fat_Content'] == 'LF' ] = 'Low Fat'
#all_data.loc[all_data['Item_Fat_Content'] == 'low fat' ] = 'Low Fat'
#all_data.loc[all_data['Item_Fat_Content'] == 'reg'] = 'Regular'


# In[308]:

# Dealing With Ordinal Catagorical Values
#Size_Mapping = {'Small':1,'Medium':2,'High':3}
Tier_Mapping ={'Tier 3':1,'Tier 2':2,'Tier 1':3}
#Fat_Mapping = {'Low Fat':1, 'Regular':2, 'low fat':1, 'LF':1, 'reg':2}
Fat_Mapping = {'Low Fat':'Low Fat', 'Regular':'Regular', 'low fat':'Low Fat', 'LF':'Low Fat', 'reg':'Regular'}


# In[309]:

#all_data['Mapped_size'] = all_data['Outlet_Size'].map(Size_Mapping)
all_data['Mapped_Tier'] = all_data['Outlet_Location_Type'].map(Tier_Mapping)
all_data['Mapped_Fat'] = all_data['Item_Fat_Content'].map(Fat_Mapping)


# In[310]:

# Dealing With Cardinal Catagorical Values 
all_data = pd.get_dummies(data=all_data, columns=['Item_Type', 'Outlet_Type','Outlet_Size','Mapped_Fat'])


# In[311]:

#Remove Unwanted Catagorical Variable from the Dataframe
all_data = all_data.drop(['Item_Weight','Item_Identifier','Outlet_Identifier','Outlet_Location_Type','Item_Fat_Content'], axis=1)


# In[312]:

type(all_data)


# In[313]:

all_data.shape


# In[314]:

# Cleaning and readying data for scikit learn operation
def clean_dataset(df):
assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
df.dropna(inplace=True)
indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
return df[indices_to_keep].astype(np.float64)


# In[315]:

all_data = clean_dataset(all_data)


# In[316]:

all_data_df = all_data


# In[317]:

type(all_data)


# In[318]:

# Normalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Min max Scalar
mms = MinMaxScaler()
# Mean SD Scalar
stdsc = StandardScaler()
all_data = mms.fit_transform(all_data)


# In[319]:

#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]


# In[320]:

y = train['Item_Outlet_Sales']/train['Item_MRP']


# In[321]:

#y = np.log(train.Item_Outlet_Sales)


# In[322]:

print X_train.shape
print X_test.shape
print y.shape


# In[323]:

#https://github.com/kalpishs/House-Prices-Advanced-Regression-Techniques
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
get_ipython().magic(u'matplotlib')
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
import seaborn as sns
sns.set()

def rmse_cv(model):
rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
return(rmse)

model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[324]:

# Lasso Model
model_lasso = LassoCV().fit(X_train, y)
rmse_cv(model_lasso).mean()


# In[325]:

preds = pd.DataFrame({"preds":model_lasso.predict(X_train), "true":y})
preds["residuals"] = preds["true"] - preds["preds"]
preds = model_lasso.predict(X_test)


# In[326]:

#Ridge Solution RidgeCV is Ridge regression with built in cross Validation , no need to find alpha seperatly
model_ridge = RidgeCV().fit(X_train, y)
rmse_cv(model_ridge).mean()
preds = model_ridge.predict(X_test)


# In[327]:

preds = test['Item_MRP']*preds


# In[328]:

#preds = np.exp(preds)


# In[329]:

solution = pd.DataFrame({"Item_Identifier":test.Item_Identifier,"Outlet_Identifier":test.Outlet_Identifier,"Item_Outlet_Sales":preds})
solution.to_csv("/Users/SKanagaraj/Documents/DataMart_AV/ridge_sol.csv",index= False, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])


# In[330]:

model_ridge.score(X_train,y)


# In[331]:

#Random Forest


# In[332]:

from sklearn.ensemble import RandomForestRegressor
Random_Forest_model = RandomForestRegressor(n_estimators=1000)
Random_Forest_model.fit(X_train, y)
ypred = Random_Forest_model.predict(X_test)


# In[333]:

ypred = test['Item_MRP']*ypred


# In[334]:

#ypred = np.exp(ypred)


# In[335]:

solution = pd.DataFrame({"Item_Identifier":test.Item_Identifier,"Outlet_Identifier":test.Outlet_Identifier,"Item_Outlet_Sales":ypred})
solution.to_csv("/Users/SKanagaraj/Documents/DataMart_AV/RF.csv",index= False, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])


# In[336]:

importances = Random_Forest_model.feature_importances_
indices = np.argsort(importances)[::-1]
std = np.std([tree.feature_importances_ for tree in Random_Forest_model.estimators_],
axis=0)


# In[337]:

# Print the feature ranking
print("Feature ranking:")

for f in range(X_train.shape[1]):
print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X_train.shape[1]), importances[indices],
color="r", yerr=std[indices], align="center")
plt.xticks(range(X_train.shape[1]), indices)
plt.xlim([-1, X_train.shape[1]])
plt.show()


# In[338]:

all_data_df.iloc[:,[23,1,3,0,5,4,2,26]]


# In[339]:

all_data_df.columns


# In[340]:

#Gradient Boosting
#https://www.kaggle.com/bklim1/gradient-boosting-regression-0-33/notebook
#Nice Notebook on data exploration and model building


# In[341]:

from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from pylab import rcParams
GBmodel = GradientBoostingRegressor()
param_dist = {"learning_rate": np.linspace(0.05, 0.15,5),
"max_depth": range(3, 5),
"min_samples_leaf": range(3, 5)}

rand = RandomizedSearchCV(GBmodel, param_dist, cv=7,n_iter=10, random_state=5)
rand.fit(X_train,y)
rand.grid_scores_

print(rand.best_score_)
print(rand.best_params_)


# In[342]:

GBmodel = GradientBoostingRegressor(min_samples_leaf= 4, learning_rate= 0.05, max_depth= 3)


# In[343]:

GBmodel.fit(X_train,y)


# In[344]:

GBpred = GBmodel.predict(X_test)


# In[345]:

GBpred = test['Item_MRP']*GBpred


# In[346]:

#GBpred = np.exp(GBpred)


# In[347]:

solution = pd.DataFrame({"Item_Identifier":test.Item_Identifier,"Outlet_Identifier":test.Outlet_Identifier,"Item_Outlet_Sales":GBpred})
solution.to_csv("/Users/SKanagaraj/Documents/DataMart_AV/GB.csv",index= False, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])


# In[348]:

# #############################################################################
# Plot feature importance
feature_importance = GBmodel.feature_importances_
# make importances relative to max importance
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.subplot(1, 2, 2)
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, all_data_df.columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()


# In[ ]:




# In[ ]:




# In[349]:

GBpred


# In[350]:

ypred


# In[296]:

preds


# In[ ]:




# In[353]:

Ensemblepred = 0.35*preds+0.65*GBpred


# In[354]:

solution = pd.DataFrame({"Item_Identifier":test.Item_Identifier,"Outlet_Identifier":test.Outlet_Identifier,"Item_Outlet_Sales":Ensemblepred})
solution.to_csv("/Users/SKanagaraj/Documents/DataMart_AV/Ensemble.csv",index= False, columns=['Item_Identifier', 'Outlet_Identifier', 'Item_Outlet_Sales'])
