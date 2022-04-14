import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sbn

dataFrame = pd.read_excel("merc.xlsx")

#sbn.distplot(dataFrame["price"])
#sbn.countplot(dataFrame["year"])
#sbn.scatterplot(x="mileage",y="price",data=dataFrame)

print(dataFrame.corr()["price"].sort_values()) #Corelation

#print(dataFrame.sort_values("price",ascending=False).head(20)) 

#len(dataFrame) * 0.01 # 1% of the dataset. 131 data. 

cleanedDf = dataFrame.sort_values("price",ascending=False).iloc[131:] #The highest price cars are gone. 
#sbn.distplot(cleanedDf["price"])
cleanedDf = cleanedDf[cleanedDf.year != 1970]
#print(cleanedDf.groupby("year").mean()["price"])
cleanedDf.drop("transmission",axis=1,inplace=True)


