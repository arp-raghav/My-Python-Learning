import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame
from sklearn.decomposition import PCA


data=pd.read_csv("C:\\Users\\arpit\\Documents\\Big_data_A2\\stock_data.csv")
data.sort_values(by=['Name'])
#Task 1(Load Data)
print(data)
#Task 2(Identify, Sort unique names in alphabetical order, count them and list to do first and last 5)
uniqueNames=data['Name'].unique()
no_of_uniques = uniqueNames.size
print(no_of_uniques)
#print first and last 5
#Task 3
groupedData = data.groupby(by=['Name'])
print(groupedData)
temp_data=data.groupby(by=['Name']).apply(lambda x: pd.Series({'isValid' : (min(x['date'])<'2014-
01-01')&(max(x['date'])>'2017-12-31')})).reset_index()
valid_stocks=temp_data[temp_data['isValid']==True]
invalid_stocks=temp_data[temp_data['isValid']==False]
valid_stock_names=np.array(valid_stocks['Name'].to_numpy())
invalid_stock_names= np.array(invalid_stocks['Name'].to_numpy())
print(invalid_stocks)
print(len(valid_stocks))
#Task 4
filtered_data=data[data['Name'].isin(valid_stock_names)].filter(items=['Name','date'])
dateGroup=filtered_data.groupby(by=['date']).count().reset_index()
filtered_dates=dateGroup[dateGroup['Name']==valid_stock_names.shape[0]]
filtered_dates=filtered_dates[(filtered_dates['date']>'2014-01-01')&(filtered_dates['date']<'2017-12-
31')]['date'].to_numpy()
print(len(filtered_dates))
#print first and last 5 dates
print(filtered_dates[5:])
print(filtered_dates[:-5])
#Task 5
updated_dataframe={'date':filtered_dates}
filtered_df=data[data['date'].isin(filtered_dates)&data['Name'].isin(valid_stock_names)].filter(items
=['Name','date','close'])#.sort_values(by='date')
print(filtered_df)
filtered_data_array=filtered_df.to_numpy()
print(filtered_data_array)
for row in filtered_data_array:
 if row[0] in updated_dataframe:
 updated_dataframe[row[0]]=np.append(updated_dataframe[row[0]],row[2])
 else:
 updated_dataframe[row[0]]=np.array(row[2])
new_df=pd.DataFrame.from_dict(updated_dataframe)
print(new_df)
#Task 6
##print(new_df.shape)
closing_values=new_df.filter(items=valid_stock_names)
closing_values_arr=closing_values.to_numpy()
current_close_values=closing_values_arr[1:]
previous_close_values=closing_values_arr[0:closing_values_arr.shape[0]-1]
close_value_ratio=np.array(np.multiply((current_close_values),(1/previous_close_values)))
##print(close_value_ratio.shape)
new_updated_df=pd.DataFrame(data=close_value_ratio,columns=valid_stock_names,index=filtered
_dates[1:])
print('closing values')
print(new_updated_df)
#Task 7
x= new_updated_df
pca = PCA()
principalcomponents = pca.fit_transform(x)
principalDF =pd.DataFrame(data = principalcomponents)
print('principal componenets')
print(principalDF)
#Task 7 ,8, 9 & 10
x_values=range(1,21)
y_values=pca.explained_variance_ratio_[0:20]
plt.plot(x_values,y_values)
plt.xlabel('Component number')
plt.ylabel('explained variance')
plt.title('explained variance vs component number')
print('plotting graph')
plt.show()
print(y_values)
#cum_variance=np.cumsum(pca.explained_variance_ratio_)
#print(cum_variance)
def pca_analysis(dataframe:DataFrame):
 pca=PCA()
 pca.fit(dataframe)
 x_values=range(1,21)
 y_values=pca.explained_variance_ratio_[0:20]
 plt.plot(x_values,y_values)
 plt.xlabel('Component number')
 plt.ylabel('explained variance')
 plt.title('explained variance vs component number')
 plt.show()
 cum_variance=np.cumsum(pca.explained_variance_ratio_)
 plt.plot(range(1,cum_variance.shape[0]+1),cum_variance)
 plt.xlabel('Component number')
 plt.ylabel('cumulative variance')
 plt.title('cumulative variance vs component number')
 plt.show()
pca_analysis(new_updated_df)
normalised_df=pd.DataFrame((new_updated_df-new_updated_df.mean())/new_updated_df.std())
pca_analysis(normalised_df)
normalised_df 
