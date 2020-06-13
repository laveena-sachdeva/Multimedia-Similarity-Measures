# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 19:56:43 2018

@author: lavee
"""

import pandas as pd
import csv
from numpy.linalg import norm
import numpy as np
import sys
import datetime
from operator import itemgetter
import params as par
import locationpreprocess as lp
import time

#To make the dataframe of uniform length, this function is used to find the maximum row length across the inout file
def get_max_len(filepath):
    with open(filepath, 'r',encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        num = []
        for i, row in enumerate(reader):
            num.append(len(''.join(row).split()))
        m = max(num)
        return m

def read_file(filepath):
	return (pd.read_csv(filepath,sep = "\s+", header = None, names= list(range(get_max_len(filepath))),engine='python',encoding = "utf-8"))

#Get user input for processing USER, IMAGE or LOCATION textual descriptors	
print("This process is running for :"+str(datetime.datetime.now()))
entity=input("Input 1: Users, 2: Images, 3:Location\n")


if(entity=="1"):
	file_name=par.user_file_name
elif(entity=="2"):
	file_name=par.image_file_name
elif(entity=="3"):
	all_file_names=lp.location_preprocess()
	file_name=par.parsed_location_file_name
else:
	print("Invalid input")
	sys.exit()
	

start_time=time.time()
#Capture command line input	
entity_id = sys.argv[1]
model = sys.argv[2]
k = sys.argv[3]

print("Reading files .. ")

#Call the read file function and get the file in a Pandas Dataframe
text_desc = read_file(par.file_location+file_name)

#Create a list (flat_list) of all entities with each occuring as many number of times as the term it contains
#This will be used when we create a separate row for each term of each entity (user,location or image)
text_desc[0] = (',' + text_desc[0].astype(str))*(((text_desc.count(axis=1)-1)/4).astype(int))

list_keys = text_desc[0].tolist()
list_keys = [i.split(',' ) for i in list_keys]

flat_list = [item for sublist in list_keys for item in sublist]
flat_list = list(filter(lambda a: a != '', flat_list))

#Delete the entity IDs so that the DataFrame can be reshaped correctly
del text_desc[0]

#This will delete any column which is all null(for example if the file contains an extra delimiter at the end of each line)
text_desc = text_desc.dropna(axis=1,how='all')

#Reshape the dataframe to create a separate row for each term
reshaped_text_desc = pd.DataFrame(text_desc.values.reshape((-1, 4)), columns=['Term', 'TF','DF','TFIDF'])

#Delete any row which is all null (this will happen after reshaping because each entity has different row length)
text_desc_vert = reshaped_text_desc.dropna(axis=0,how='all')

#Associate the flat_list created above with each term, tf, df and tfidf values
text_desc_vert.insert(loc=0,column='Entity_Id',value=flat_list)


#Convert this dataframe to a dictionary with each entity as a key, its value being a dictionary which has Terms as the keys, and their TF,DF or TFIDF as values
if(model=='TF'):
	desc_dict = text_desc_vert.groupby('Entity_Id').apply(lambda x: dict(zip(x.Term, x.TF))).to_dict()
elif(model=='DF'):
	desc_dict = text_desc_vert.groupby('Entity_Id').apply(lambda x: dict(zip(x.Term, x.DF))).to_dict()
elif(model=='TF-IDF'):
	desc_dict = text_desc_vert.groupby('Entity_Id').apply(lambda x: dict(zip(x.Term, x.TFIDF))).to_dict()


#Get all the terms of the input entity
input_val=desc_dict.get(entity_id)

#Weights of each terms used to calculate the norm later
val_input_val=list(input_val.values())

list_distances=[]
term_products=[]


#Iterate through all other entities in the descriptor file and calculate distance from each of them
for key,value in desc_dict.items():
	sum=0
	for input_key,input_value in input_val.items():
		term_product = input_val[input_key]*desc_dict[key].get(input_key,0)
		sum = sum + term_product
		term_products.append([key,input_key,term_product])	
		
	
	list_weights=list(desc_dict[key].values())
	sum = sum/(norm(val_input_val)*norm(list_weights))
	list_distances.append([key,sum])
	

#Create a dataframe from the list with distance from all other entities
similarity_df=pd.DataFrame(list_distances)

#Sort the dataframe in ascending order, to get the most similar at top (least distance)
sorted_df = similarity_df.sort_values(by=[1,0],ascending=[False,True])

#Pick the top k similar entities
output_df = sorted_df.head(int(k))

#Rename columns to logical names
output_df=output_df.rename(columns={0:'nearest_entity',1:'similarity'})

#Contribution of each term
term_products_df = pd.DataFrame(term_products)

#Fetch only top k similar entities
top_k = term_products_df.loc[term_products_df[0].isin(output_df['nearest_entity'])]

#Sort and keep just keep top 3 contributors for each entity
top_k=top_k.sort_values([0,2],ascending=[True,False])
top_k=top_k.groupby(0).head(3)


#Create a comma separated list 
top_k=top_k.groupby(0)[1].agg([(1, ', '.join)])
top_k=top_k.reset_index()

#Rename columns to logical names
top_k=top_k.rename(columns={0:'nearest_entity',1:'Top 3 Terms'})

#Create a final dataframe by displaying everything together
final_df=pd.merge(output_df,top_k,on='nearest_entity',how='inner')

if(entity=="3"):
	final_df_location=pd.DataFrame()
	for idx,row in final_df.iterrows():
		row['nearest_entity']=all_file_names[str(int(row['nearest_entity']))]
		final_df_location=final_df_location.append(row)
	final_df_location=final_df_location[['nearest_entity','similarity','Top 3 Terms']]
	print(final_df_location)
	print("--- %s seconds ---" % (time.time() - start_time))
	sys.exit()
		
print(final_df)
print("--- %s seconds ---" % (time.time() - start_time))
