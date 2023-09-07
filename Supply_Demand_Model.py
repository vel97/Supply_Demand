import numpy as np 
import pandas as pd
from sklearn import preprocessing
import seaborn as sns
from plotnine import ggplot, aes, geom_histogram, geom_boxplot
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn .model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Demand Data
demand = pd.read_csv("C:\\Users\\SriramvelM\\Downloads\\archive (7)\\demand.csv")
demand.head()
demand.dtypes
demand.shape

#Checking for null values in data
demand.isnull().sum()

#Imputing missing values
demand['CSUSHPISA'].fillna(demand['CSUSHPISA'].mean(), inplace=True)
demand['INTDSRUSM193N'].fillna(demand['INTDSRUSM193N'].mean(), inplace=True)

#Changing datatype for date
demand['DATE'] = pd.to_datetime(demand['DATE'])
demand.dtypes

#Supply Data
supply = pd.read_csv("C:\\Users\\SriramvelM\\Downloads\\archive (7)\\supply.csv")
supply.head()
supply.dtypes
supply.shape

#Checking for null values in data
supply.isnull().sum()

#Removing insignificant row
supply = supply[:-1]

##################Changing datatypes##################################
supply['DATE'] = pd.to_datetime(supply['DATE'])

supply['CSUSHPISA'].value_counts()

#Imputing with previous value
supply['CSUSHPISA'] = supply['CSUSHPISA'].replace(['.'], '297.8966667')

#Converting to numeric
supply['CSUSHPISA'] = pd.to_numeric(supply['CSUSHPISA'])
supply['MSACSR'] = pd.to_numeric(supply['MSACSR'])
supply['PERMIT'] = pd.to_numeric(supply['PERMIT'])
supply['TLRESCONS'] = pd.to_numeric(supply['TLRESCONS'])
supply['EVACANTUSQ176N'] = pd.to_numeric(supply['EVACANTUSQ176N'])
supply.dtypes

#Merging the datasets
data = pd.merge(demand,supply, on='DATE',suffixes=('_demand', '_supply'))
data.head()
data.shape

#Dropping common column
data = data.drop('CSUSHPISA_demand', axis=1)

#Renaming column
data.columns
data.rename(columns={'CSUSHPISA_supply': 'CSUSHPISA'}, inplace=True)

#Check for null
data.isnull().sum()

#Dopping duplicates if exists
data = data.drop_duplicates()

#Datetome decompoition
data['Year'] = data['DATE'].dt.year
data['Month'] = data['DATE'].dt.month
data['Day'] = data['DATE'].dt.day
data.dtypes

#Datatype changes for day,year and month
data['Year'] = data['Year'].astype(str)
data['Month'] = data['Month'].astype(str)
data['Day'] = data['Day'].astype(str)

#Check for outliers
cont = data.select_dtypes(include='number')
cat = data.select_dtypes(include='object')
data.describe()

#Identifying and removing outliers
for i in cont:
    iqr = data[i].quantile(0.75) - data[i].quantile(0.25)
    Upper_T = data[i].quantile(0.75) + (1.5 * iqr)
    Lower_T = data[i].quantile(0.25) - (1.5 * iqr)
    data[i] =  data[i].clip(Upper_T, Lower_T)

data.describe()

# label_encoder object knows how to understand word labels.
# For tree algorithms label encoding is sufficient for model to understand our data.
#No need of one-hot encoding.
label_encoder = preprocessing.LabelEncoder()

# data['y'] = label_encoder.fit_transform(cust['y'])
for i in cat:
    data[i] = label_encoder.fit_transform(data[i])

#Changing datatype again after label encoding
data['Year'] = data['Year'].astype(str)
data['Month'] = data['Month'].astype(str)
data['Day'] = data['Day'].astype(str)

#Correlation identification
#Heatmap using corr function
sns.heatmap(cont.corr(), annot=True, vmin=-1, vmax=1)
plt.show()

#Histogram plot
for i in cont:
    ggplot(data) + aes(x=i) + geom_histogram()
    plt.show()

#Scatter plot to check replationship between continuous independent variables
for i in cont:
    sns.scatterplot(x=i, y="CSUSHPISA", data=data)
    plt.show()

#Regression plot
for i in cont:
    sns.regplot(x=i, y="CSUSHPISA", data=data)
    plt.show()

#Boxplot to check the spread of the data
for j in cont:
    for i in cat:
        ggplot(data) + aes(x=i, y=j) + geom_boxplot()

#Dropping date and month since it is insignificant after extraction
data = data.drop(['DATE', 'Month'], axis = 1)

#Splitting the dataset as train and test
x = data.drop('CSUSHPISA', axis='columns')
y = data['CSUSHPISA']

#Train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state=0)

np.shape(x_train)
np.shape(x_test)
np.shape(y_train)
np.shape(y_test)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

#Scaling the variables
x_train_cont = x_train.select_dtypes(include='number').columns
t = [('n', StandardScaler(), x_train_cont)]
selective = ColumnTransformer(transformers=t)
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR()}

#Creating pipeline and looping it for all models
from sklearn.pipeline import Pipeline
result = {}
for i,j in models.items():
    pipeline = Pipeline([('s',selective),('m',j)])
    scores = cross_val_score(pipeline, x_train, y_train, cv=5, scoring='neg_mean_squared_error', error_score='raise')
    mse_scores = -scores
    avg_mse = mse_scores.mean()
    result[i] = avg_mse

#Identifying the best model
best_model = min(result, key=result.get)
best_model_in_the_group = models[best_model]

#Fitting and predicting the best model
best_model_in_the_group.fit(x_train, y_train)
pred = best_model_in_the_group.predict(x_test)

#Mean Square Error for best model
mse = mean_squared_error(y_test, pred)
for model, mse_score in result.items():
    print(f"{model}: MSE={mse_score}")

print("Best Model is",best_model, "and its MSE on Testing Set:",mse)


#R squared 
r2 = r2_score(y_test, pred)
print("R-squared value:", r2)

#Coefficients
coefficients = best_model_in_the_group.coef_

for feature, coefficient in zip(x_train.columns, coefficients):
    print(f"{feature}: {coefficient}")
