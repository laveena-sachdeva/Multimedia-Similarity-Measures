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
import dask.dataframe as dd
import threading
import params as par
import datetime
from operator import itemgetter
import statistics as s
from sklearn.metrics.pairwise import manhattan_distances
from numpy import genfromtxt
import sklearn
import itertools


#read file into a 2D numpy array
def read_file(filepath):
	return(genfromtxt(filepath, delimiter=',',encoding = "utf-8"))
		
#Find distances between each location pair
def write_distances(visual_desc,file2):
	input_images=visual_desc[:,1]
	images_candidate=file2[:,1]
	input_location=visual_desc[0,0]
	candidate_location=file2[0,0]
	#print(images_candidate)
	#print(input_images)
	distance=pd.DataFrame()
	distance_pairwise = sklearn.metrics.pairwise.euclidean_distances(visual_desc[:,2:],file2[:,2:])
	
	sz=np.size(distance_pairwise)
	
	product = list(itertools.product(input_images,images_candidate))
	a=np.array(product)
	sz_a=np.size(a)
	#print(candidate_location)
	
	
	distance_pairwise=distance_pairwise.reshape(sz,1)
	
	distance = np.concatenate((distance_pairwise,a), axis=1)
	
	
	distance=np.insert(distance, 1, values=input_location, axis=1)
	distance=np.insert(distance, 3, values=candidate_location, axis=1)
	distance_sorted = distance[distance[:,0].argsort()]
	return(distance_sorted)
	
#Find distance between each location ppair for 3x3 models	
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
	return(distance_sorted)
	
#create all pairs of images for given input location and call functions to find distances
def calculate_distance(input_id,input_file_name,input_model_name,file_dict):		
	#print("Entering calculate_distance  "+str(datetime.datetime.now()))
	visual_desc_location=par.visual_desc_location
	visual_desc = read_file(visual_desc_location+input_file_name)
	
	file_list=list(file_dict.values())
	
	number_of_entities = len(file_list)
	
	files_data=[]
	
	visual_desc = np.insert(visual_desc, 0, values=input_id, axis=1)

#Read data of all files			
	idx = 0
	for key in file_dict:
		files_data.append(read_file(visual_desc_location+file_dict[key]))
		files_data[idx]=np.insert(files_data[idx],0, values=key,axis=1)
		idx = idx+1
#find distances
	t=[]
	for i in range(number_of_entities):
		if(input_model_name.find("3x3")!=-1):
			t.append(write_distances_3x3(visual_desc,files_data[i]))
		else:
			t.append(write_distances(visual_desc,files_data[i]))
	df_distance=[]
	
#Find the median distance	
	for i in range(number_of_entities):
		unpickled_df = np.array(t[i])
		avg_distance=s.median(unpickled_df[:,0])
		input=unpickled_df[0,1]
		candidate=unpickled_df[0,3]
		df_distance.append([input,candidate,avg_distance])
	
	
	
	sorted_df = sorted(df_distance,key=itemgetter(2),reverse=False)	
	
	
	return(sorted_df)

