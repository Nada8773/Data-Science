import numpy as np
import pandas as pd

columns_names=['user_id','item_id','rate','time']
Movie_data=pd.read_csv('Movie_udata.csv',sep='\t',names=columns_names)

Movie_title=pd.read_csv('Movie_Id_Titles.csv')

Movie_data=pd.merge(Movie_data,Movie_title,on='item_id')

Movie_item=pd.read_csv('Movie_uitem.csv')

print(Movie_item.head())