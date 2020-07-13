# Multimedia Similarity Measures

Given a dataset about locations and the associated users and images, this implementation tries to find the most similar entities using various given descriptors.
The program directly processes csv/txt files. No database setup is required.

## GENERAL USAGE NOTES

- The project currently only supports a specifc file structure and dataset in a particular format.
- The dataset used for all the testing can be found [here](http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/ ).
- The files used from the above dataset are all present in the devset folder:
   - devset_topics.xml
   - descvis.zip
   - desctxt.zip
- Refer to the README file at [this link](http://skuld.cs.umass.edu/traces/mmsys/2015/paper-5/Div150Cred_readme.txt) to understand the dataset.

## GETTING STARTED: INSTALLATION ON WINDOWS10

### Installing Python 3.7.0 for Windows10
1. Download Python3.7.0 by clicking on [this link](https://www.python.org/ftp/python/3.7.0/python-3.7.0-webinstall.exe).
2. Open the executable.
3. Follow the instructions on the screen to install.

### Libaries required to  be installed (run the corresponding commands on cmd after installing Python 3.7.0)

- pandas : pip install pandas  
- csv: Part of python's standard library  
- numpy: pip install numpy  
- sys: Part of python's standard library  
- datetime: pip install datetime  
- operator: Part of python's standard library  
- bs4: pip install bs4  
- statistics: pip install statistics  
- lxml: pip install lxml  
- time: Part of python's standard library  
- re: Part of python's standard library  
- sklearn: pip install sklearn  
- itertools: Part of python's standard library  


## HOW TO USE

### Python programs:
Below are the 3 Python programs needed to accomplish the tasks of finding similarity:
1. textual_similarity.py - This can be used to match a given input entity (user/image/location) with other corresponding entities and return top k similar entities, along with the features that have the maximum contribution in similarity.
Expected input arguments: <entity id> <model - TF/DF/IDF> <k>

2. visual_similarity_per_model.py - This can be used to match a given input location with other locations based on the user desired visual descriptors and return top k similar locations, along with the image-pairs that have the maximum contribution in similarity.
Expected input arguments: <location id> <model - CM, CM3x3, CN, CN3x3,CSD,GLRLM, GLRLM3x3,HOG,LBP, LBP3x3> <k>

3. visual_similarity_wrapper.py - This internally uses visual_similarity_all_models.py for various functionalities. This can be used to match a given input location with other locations based on differemt weights of all the visual descriptors (CM, CM3x3, CN, CN3x3,CSD,GLRLM, GLRLM3x3,HOG,LBP, LBP3x3) and return top k similar locations, along with the individual contribution of the models.
Expected input arguments: <location id> <k>


Files required, along with the above listed Python programs:
1. visual_similarity_all_models.py
2. params.py
3. locationpreprocess.py


Get all the code files and data files in the directory where you will run the code and set the parameters in params.py accordingly. 

### Initializing params.py (The default file is given below)  

- file_location: Set this to the base directory path where all your datafiles would be present. The user, images and location textual descriptors data can further be in differnet directories.  
- user_file_name: Set this to the file name for the Textual Descriptors for users. Also include the file path after the 'file_location' in user_file_name.  
- image_file_name: Set this to the file name for the Textual Descriptors for images. Also include the file path after the 'file_location' in image_file_name.  
- location_file_name: Set this to the file name for the Textual Descriptors for location. Also include the file path after the 'file_location' in location_file_name. The file for location descriptors is expected to contain just the location description as the identifier, and NOT location description and location name both.  
- parsed_location_file_name: The location process internally uses this file to process the location textual descriptors. It can be any logical filename name.  
- xml_file_loc: Path of the topic xml file containing location meta-data in the current dataset.  
- xml_file_name: Name of the topic xml file containing location meta-data.  
- visual_desc_location: Path where all the visual descriptors and present. 

## TASKS:  

### User Similarity: Implement a program which, given a user ID, a model (TF, DF, TF-IDF), and value “k”, returns the most similar k users based on textual descriptors. For each match, also list the overall matching score as well as the 3 terms that have the highest similarity contribution.  

Interface Specification: textual_similarity.py <input_id> <model> <k>  
Example: textual_similarity.py 56087830@N00 DF 8  
Output is displayed on the console.

### Image Similarity: Implement a program which, given an image ID, a model (TF, DF, TF-IDF), and value “k”, returns the most similar k images based on textual descriptors. For each match, also list the overall matching score as well as the 3 terms that have the highest similarity contribution.  

Interface Specification: textual_similarity.py <input_id> <model> <k>  
Example: textual_similarity.py 288051306 DF 5  
Output is displayed on the console.

### Location Similarity: Implement a program which, given a location ID, a model (TF, DF, TF-IDF), and value “k”, returns the most similar k locations based on textual descriptors. For each match, also list the overall matching score as well as the 3 terms that have the highest similarity contribution.  

Interface Specification: textual_similarity.py <input_id> <model> <k>  
Example: textual_similarity.py 27 TF 5  
Output is displayed on the console.

### Model Based Location Similarity: Implement a program which, given a location ID, amodel (CM, CM3x3, CN, CN3x3,CSD,GLRLM,GLRLM3x3, HOG,LBP, LBP3x3), and value “k”, returns the most similar k locations based on the corresponding visual descriptors of the images as specified in the “img” folder. For each match, also list the overall matching score as well as the 3 image pairs that have the highest similarity contribution.  

Interface Specification:  visual_similarity_per_model.py <location_id> <model> <k>  
Example: visual_similarity_per_model.py 10 CN3x3 7  
Output: The output is written to a CSV file in your working directory. The file is named as visual_similarity_<input_model>.csv.

### Location Similarity using Visual Descriptors: Implement a program which, given a location ID and value “k”, returns the most similar k locations based on the corresponding visual descriptors of the images as specified in the “img” folder. For each match, also list the overall matching score and the individual contributions of the 10 visual models.  

Interface Specification:  visual_similarity_wrapper.py <location_id> <k>  
Example: visual_similarity_wrapper 4 5  
Output: The output is written to a CSV file in your working directory. The file is named as visual_similarity_all_models.csv.



<!-- ## TESTS:

### Given a userid (39052445@N00), model (TF) and k(2):
Open CMD   
Run Program: python <directory>/user_input_opt_v1.py 39052445@N00 TF 2

Output  
Similarity |     ID      | Similar Terms  
0.866  | 9688281@N07 | "bok","tower","garden"  
0.789  | 93858545@N00| "garden","tower","bok"   -->
            
## LIBRARY REFERENCES:         

- pandas: http://pandas.pydata.org/pandas-docs/stable/
- csv: https://docs.python.org/3.7/library/csv.html
- numpy: https://docs.scipy.org/doc/
- sys: https://docs.python.org/3.7/library/sys.html
- datetime: https://docs.python.org/3.7/library/datetime.html
- operator: https://docs.python.org/3/library/operator.html
- bs4: https://beautiful-soup-4.readthedocs.io/en/latest/
- statistics: https://docs.python.org/3/library/statistics.html
- lxml: https://lxml.de/

## AUTHORS:
[Laveena Sachdeva](https://github.com/laveena-sachdeva)
