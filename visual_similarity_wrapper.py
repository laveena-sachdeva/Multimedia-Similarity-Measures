
import visual_similarity_all_models as task5
import pandas as pd
import params as par
import datetime
from bs4 import BeautifulSoup as Soup
import numpy as np
import sys
import time


#Create a list of list containing location ids and location names by  processing the xml file
def location_preprocess():
	xml_file_name=par.xml_file_name
	xml_file_location=par.xml_file_loc

	visual_desc_location=par.visual_desc_location

	datasource = open(xml_file_location + xml_file_name).read()
	soup=Soup(datasource,'lxml')
	
	location=[]
	location_id=[]
	location_name=[]

	for number in soup.findAll('number'):
		location_id.append(number.string)
	for title in soup.findAll('title'):
		location_name.append(title.string)
		

	location.append(location_id)
	location.append(location_name)
	
	
	return(location)

#create a dictionary with location ids as key, and location names as values
def set_file_names(location,model):
	location_id=location[0]
	location_name=location[1]
	
	dict = {}
	
	for idx,key in enumerate(location_id):
			dict[key] = location_name[idx] + str(" ") + model + str(".csv")
			
	return(dict)	

#scale the dataframe in a range of 0 to 1 by dividing with the maximum value	
def normalize(df):
    result = df.copy()
    max_value = df['avg_distance'].max()
    min_value = df['avg_distance'].min()
    result['avg_distance'] = (df['avg_distance'] / max_value)
    return result	

#Finding teh index of input query	
def rank(df):
	for idx,row in df.iterrows():
		if(row['input']==row['candidate']):
			return idx
		

#Start here..		
start_time=time.time()		
	
	
location=location_preprocess()		
location_id=location[0]
location_name = location[1]

#capture user input
input_id = sys.argv[1]
input_k = sys.argv[2]

target_location = par.file_location

#list of models in system
models = ['CM','CM3x3','CN','CN3x3','CSD','GLRLM','GLRLM3x3','HOG','LBP','LBP3x3']
file_names={}
for idx, model in enumerate(models):
	file_names[model] = set_file_names(location,model)

number_of_entities = len(location_name)	

print("1..")

dict_position={}
list_dataframe=[]

#Find distances for CM model
df_CM=pd.DataFrame(task5.calculate_distance(input_id,file_names['CM'][input_id],'CM',file_names['CM']))
df_CM.insert(loc=0,column='Model',value='CM')
df_CM.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_CM=normalize(df_CM)

dict_position['CM']=rank(df_CM)
list_dataframe.append(df_CM)

#print(df_CM)
#Find distances for CM3x3 model
print("2.. ")
df_CM3=pd.DataFrame(task5.calculate_distance(input_id,file_names['CM3x3'][input_id],'CM3',file_names['CM3x3']))
df_CM3.insert(loc=0,column='Model',value='CM3x3')
df_CM3.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_CM3=normalize(df_CM3)

dict_position['CM3x3']=rank(df_CM3)
list_dataframe.append(df_CM3)
#print(df_CM3)

#Find distances for CN model
print("3..")
df_CN=pd.DataFrame(task5.calculate_distance(input_id,file_names['CN'][input_id],'CN',file_names['CN']))
df_CN.insert(loc=0,column='Model',value='CN')
df_CN.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_CN=normalize(df_CN)

dict_position['CN']=rank(df_CN)
list_dataframe.append(df_CN)
#print(df_CN)

#Find distances for CN3x3 model
print("4..")
df_CN3=pd.DataFrame(task5.calculate_distance(input_id,file_names['CN3x3'][input_id],'CN3x3',file_names['CN3x3']))
df_CN3.insert(loc=0,column='Model',value='CN3x3')
df_CN3.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_CN3=normalize(df_CN3)

dict_position['CN3x3']=rank(df_CN3)
list_dataframe.append(df_CN3)
#print(df_CN3)

#Find distances for CSD model
print("5..")		
df_CSD=pd.DataFrame(task5.calculate_distance(input_id,file_names['CSD'][input_id],'CSD',file_names['CSD']))
df_CSD.insert(loc=0,column='Model',value='CSD')
df_CSD.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_CSD=normalize(df_CSD)

dict_position['CSD']=rank(df_CSD)
list_dataframe.append(df_CSD)
#print(df_CSD)

#Find distances for GLRLM model
print("6..")
df_GLRLM=pd.DataFrame(task5.calculate_distance(input_id,file_names['GLRLM'][input_id],'GLRLM',file_names['GLRLM']))
df_GLRLM.insert(loc=0,column='Model',value='GLRLM')
df_GLRLM.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_GLRLM=normalize(df_GLRLM)

dict_position['GLRLM']=rank(df_GLRLM)
list_dataframe.append(df_GLRLM)
#print(df_GLRLM)

#Find distances for GLRLM3x3 model
print("7..")
df_GLRLM3=pd.DataFrame(task5.calculate_distance(input_id,file_names['GLRLM3x3'][input_id],'GLRLM3x3',file_names['GLRLM3x3']))
df_GLRLM3.insert(loc=0,column='Model',value='GLRLM3x3')
df_GLRLM3.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_GLRLM3=normalize(df_GLRLM3)

dict_position['GLRLM3x3']=rank(df_GLRLM3)
list_dataframe.append(df_GLRLM3)
#print(df_GLRLM3)

#Find distances for HOG model
print("8..")
df_HOG=pd.DataFrame(task5.calculate_distance(input_id,file_names['HOG'][input_id],'HOG',file_names['HOG']))
df_HOG.insert(loc=0,column='Model',value='HOG')
df_HOG.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_HOG=normalize(df_HOG)

dict_position['HOG']=rank(df_HOG)
list_dataframe.append(df_HOG)
#print(df_HOG)

#Find distances for LBP model
print("9..")
df_LBP=pd.DataFrame(task5.calculate_distance(input_id,file_names['LBP'][input_id],'LBP',file_names['LBP']))
df_LBP.insert(loc=0,column='Model',value='LBP')
df_LBP.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_LBP=normalize(df_LBP)

dict_position['LBP']=rank(df_LBP)
list_dataframe.append(df_LBP)
#print(df_LBP)

#Find distances for LBP3x3 model
print("10..")
df_LBP3=pd.DataFrame(task5.calculate_distance(input_id,file_names['LBP3x3'][input_id],'LBP3x3',file_names['LBP3x3']))
df_LBP3.insert(loc=0,column='Model',value='LBP3x3')
df_LBP3.rename(columns={0: "input", 1:"candidate", 2:"avg_distance"},inplace=True)
df_LBP3=normalize(df_LBP3)

dict_position['LBP3x3']=rank(df_LBP3)
list_dataframe.append(df_LBP3)
#print(df_LBP3)

#dict_ranked={key: rank/10 for rank, key in enumerate(sorted(dict_position, key=dict_position.get, reverse=True), 1)}
dict_ranked={key: 1  for rank, key in enumerate(sorted(dict_position, key=dict_position.get, reverse=True), 1)}

#find the mean distance of each location from all model
weighted_distances=[]
for idx,item in enumerate(location_id):
	w_distance=0
	contributions=[]
	for idx2, model in enumerate(models):
		df=list_dataframe[idx2]
		w = (dict_ranked[model]*(df.iloc[df.index[df['candidate'].astype('int')==int(item)].tolist()[0]]['avg_distance']))
		w_distance = w_distance + w
		contributions.append([model,w])
	weighted_distances.append([item,w_distance/10,contributions])

#sort the values and pick the top k locations	
weighted_distances_df = pd.DataFrame(weighted_distances,columns=['location','weighted_distance','contribution'])
weighted_distances_df = weighted_distances_df.sort_values(by='weighted_distance')
weighted_distances_df=weighted_distances_df.head(int(input_k))


df_final=pd.DataFrame()
#replace location id with location name
for idx,row in weighted_distances_df.iterrows():
	row['location']=file_names['CM'][str(int(row['location']))].split(' ')[0]
	df_final=df_final.append(row)

df_final = df_final[['location','weighted_distance','contribution']]
#print data and write it to a csv file as well
print(df_final)
df_final.to_csv(target_location+'cse515_visual_similarity_all_models.csv',sep=',')
print("--- %s seconds ---" % (time.time() - start_time))

		