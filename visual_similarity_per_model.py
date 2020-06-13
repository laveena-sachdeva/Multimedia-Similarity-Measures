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
import params as par
from bs4 import BeautifulSoup as Soup
from sklearn.metrics.pairwise import manhattan_distances
from numpy import genfromtxt
import sklearn
import itertools
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

#Read file into a numpy 2D array
def read_file(filepath):
	return(genfromtxt(filepath, delimiter=',',encoding = "utf-8"))

#Manhattan distance calculation	
def write_distances_manhattan(visual_desc,file2):
	distance_list=[]
	id_list=[]
	distance=pd.DataFrame()
	#print(id_list)
	for input in visual_desc:
		for first in file2:
			dist = manhattan_distances(input[2:].reshape(1,-1),first[2:].reshape(1,-1))[0][0]
			distance_list.append(dist)
			id_list.append([input[0],input[1],first[0],first[1]])
	#print(id_list)
	#distance=distance.append(pd.DataFrame({'Distance':[dist]}))		
	distance=distance.append(pd.DataFrame(id_list, columns=['Input Location','Input Id','Second Location','Second Id']))
	distance.insert(loc=0,column='Distance',value=distance_list)		
	distance_sorted = distance.sort_values('Distance',ascending=True)	
	return(distance_sorted)

	
#Finding distances between all image pairs of two locations	
def write_distances(visual_desc,file2):
	input_images=visual_desc[:,1]
	images_candidate=file2[:,1]
	
	input_location=visual_desc[0,0]
	candidate_location=file2[0,0]
	
	distance=pd.DataFrame()
	distance_pairwise = sklearn.metrics.pairwise.euclidean_distances(visual_desc[:,2:],file2[:,2:])
	
	sz=np.size(distance_pairwise)
	
	product = list(itertools.product(input_images,images_candidate))
	a=np.array(product)
	sz_a=np.size(a)
		
	distance_pairwise=distance_pairwise.reshape(sz,1)
	
	distance = np.concatenate((distance_pairwise,a), axis=1)
	
	distance=np.insert(distance, 1, values=input_location, axis=1)
	distance=np.insert(distance, 3, values=candidate_location, axis=1)
	distance_sorted = distance[distance[:,0].argsort()]
	return(pd.DataFrame(distance_sorted,columns=['Distance','Input Location','Input Id','Second Location','Second Id']))
	
#Distance calculation for 3x3 models by taking a mean of 9 subimages	
def write_distances_3x3(visual_desc,file2):
	input_images=visual_desc[:,1]
	images_candidate=file2[:,1]
	input_location=visual_desc[0,0]
	candidate_location=file2[0,0]
	distance=pd.DataFrame()
	
	num_of_columns = visual_desc.shape[1]
	range = int((num_of_columns-2)/9)
	
	low = 2
	high = low + range
	list_3x3=[]
	while(high<=num_of_columns):
		distance_pairwise = sklearn.metrics.pairwise.euclidean_distances(visual_desc[:,low:high],file2[:,low:high])
		sz=np.size(distance_pairwise)
		distance_pairwise=distance_pairwise.reshape(sz,1)
		list_3x3.append(distance_pairwise)
		low=high
		high=high+range
	distance_pairwise=np.mean(list_3x3,axis=0)
	
	product = list(itertools.product(input_images,images_candidate))
	a=np.array(product)
	sz_a=np.size(a)
	
	distance_pairwise=distance_pairwise.reshape(sz,1)
	distance = np.concatenate((distance_pairwise,a), axis=1)
	
	
	distance=np.insert(distance, 1, values=input_location, axis=1)
	distance=np.insert(distance, 3, values=candidate_location, axis=1)
	distance_sorted = distance[distance[:,0].argsort()]
	return(pd.DataFrame(distance_sorted,columns=['Distance','Input Location','Input Id','Second Location','Second Id']))
	
#create a dictionary to create a mapping between location id and location name
def set_file_names(location,model):
	location_id=location[0]
	location_name=location[1]
	
	dict = {}
	
	for idx,key in enumerate(location_id):
			dict[key] = location_name[idx] + str(" ") + model + str(".csv")
			
	return(dict)
		
#Start here		
start_time=time.time()

print("Processing....")

#Capture command line inputs
input_id = sys.argv[1]
input_model = sys.argv[2]
input_k = sys.argv[3]

location=location_preprocess()		
location_id = location[0]
location_name = location[1]


file_names = set_file_names(location,input_model)

target_file_location = par.file_location
file_location = par.visual_desc_location
input_file_name = file_names[input_id]
visual_desc = read_file(file_location + input_file_name)

#del file_names[str(input_id)]
#print(file_names)


number_of_entities = len(file_names)

file_data=[]

visual_desc = np.insert(visual_desc, 0, values=input_id, axis=1)

#Read all other location files for the given model
idx = 0
for key in file_names:
		file_data.append(read_file(file_location+file_names[key]))
		file_data[idx]=np.insert(file_data[idx],0, values=key,axis=1)
		idx = idx+1
		
		
#find distances of given location with other location
t=[]
for i in range(number_of_entities):
	if(input_model.find("3x3")!=-1):
		t.append(write_distances_3x3(visual_desc,file_data[i]))
	else:
		t.append(write_distances(visual_desc,file_data[i]))

df_distance=pd.DataFrame(columns=['input','candidate','avg_distance','input_pairs'])

#Find the median of distances between all image pairs, and find the top 3 contributors
for i in range(number_of_entities):
	unpickled_df = t[i]
	avg_distance=unpickled_df["Distance"].median()
	input=unpickled_df["Input Location"].iloc[0]
	candidate=unpickled_df["Second Location"].iloc[0]
	input_pairs=[]
	for i in range(3):
		input_pairs.append([int(unpickled_df["Input Id"].iloc[i]),int(unpickled_df["Second Id"].iloc[i])])
		
	df_distance=df_distance.append([{'input':input,'candidate':candidate,'avg_distance':avg_distance,'input_pairs':input_pairs}])

#Sort the distances based on avg distance, lesser the distance more the similarity
sorted_df = df_distance.sort_values('avg_distance',ascending=True)	


output_df = sorted_df.head(int(input_k))
df_final=pd.DataFrame()


#Replace the location id with location name
for idx,row in output_df.iterrows():
	row['candidate']=file_names[str(int(row['candidate']))].split(' ')[0]
	row['input']=input_file_name.split(' ')[0]
	df_final=df_final.append(row)
print(df_final)
print("--- %s seconds ---" % (time.time() - start_time))
#Write to a csv file
df_final.to_csv(target_file_location+"cse515_visual_similarity_"+input_model+".csv")
sys.exit()



