import params as par
from bs4 import BeautifulSoup as Soup
import re

def location_preprocess():
	xml_file_name=par.xml_file_name
	xml_file_location=par.xml_file_loc

	text_desc_name=par.location_file_name
	text_desc_location=par.file_location
	parsed_location_file_name = par.parsed_location_file_name

	datasource = open(xml_file_location + xml_file_name).read()
	soup=Soup(datasource,'lxml')
		
	location_id=[]
	location_name=[]

	for number in soup.findAll('number'):
		location_id.append(number.string)
	for title in soup.findAll('title'):
		location_name.append(title.string.replace('_',' '))
	
	location={}
	for idx,item in enumerate(location_id):
		location[item]=location_name[idx]
		
		
		
	#open your csv and read as a text string
	with open(text_desc_location + text_desc_name, 'r',encoding='utf-8') as f:
		location_file = f.read()

	new_location_file=location_file
	# substitute
	for idx,item in enumerate(location_name):
		new_location_file = re.sub(item, location_id[idx], new_location_file)




	new_csv_path = text_desc_location + parsed_location_file_name # or whatever path and name you want
	with open(new_csv_path, 'w',encoding='utf-8') as f:
		f.write(new_location_file)
		
	return(location)