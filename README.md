# sciml
This contains all the code used in the azure python streaming data experiments that were described in the blog [on microservice performance architecture](http://esciencegroup.com/2015/10/08/performance-analysis-of-a-cloud-microservice-based-ml-classifier/) and [Processing Scholarly Event Streams in the Cloud](http://esciencegroup.com/2015/09/08/processing-scholarly-event-streams-in-the-cloud/).   The code has been rewritten to be free of all azure dependencies with the exception of the use of Azure Tables for the final storage from the table web service.   It is certainly possible to rewrite this to use another database.

##The Data

There are four type

1. The arxiv configuration files.  They take the form of config\_name.json where name can be all4 (the top level), bio (the arxiv q-bio objects), compsci (computer science), math, phy (Physics), finance (finance).   
2. The machine learning model files (as generated by doc\_analysis\_final.py described below)
3. The raw daa from the streams.  There are three of these.   The sciml\_data\_arxiv is the original data set from arxiv.  sciml\_data\_arxiv\_new\_9\_28\_15 is a recent snapshot of arxiv data not used in the training a portion of this was used for the training set.  The sciml\_data\_scimags is the rss data from the various science mags. 
4. The output of the main (top level) classifer.  This was used to push events directly to the message broker for use in the performance analysis.  This takes the form dump\_all\_subtopic\_name where name is one of q-bio, Physics, compsci, math or physics. (note these are not the same as the names on the config files.)
 
The data is stored in two places. 

1. The configuration data, the rss feed input data and model data is stored on a publicly readable oneDrive site.  The url for this is  [http://1drv.ms/1PCOT8l](http://1drv.ms/1PCOT8l) (you may need to cut this and paste it into your browser)
2. The oneDrive files are ok for download from the browser, but not as efficient for program level access.  So the programs here read the files from an public, read-only account "http://esciencegroup.blob.core.windows.net/scimlpublic" The code for reading the files is included in the source codes.  

##doc\_analysis\_final.py

This is the generic scifeed document analyzer.  
It reads in the discipline specific config file and it will load the data and go from there.
Tt will generate all the models used by the classifier.
You invoke it with two arguments:  the topic which is one of
all4, bio, compsci, finance, math, phy, and
an integer which is the max size of the training set. note:
if the set of documents in a subcategory is less than the max size the entire set
of documents in that subcategory, the entire subcategory is selected as the training set. 
A sample invocation is 
 
	ipython doc_analysis_final.py all4 1200 

this one will generate the machine learning modes for the top-level analysis for 1200 randomly selected items from arxiv

##main\_classifier.py

This is a version of the classifier that generates topic lists
it is used to generate the entire set of predicted main topic
classifications used by the performance analysis
invoke it with two paramters like this

	ipython main_classifer.py all4  1200

this will pick 1200 random items from arxiv and run the top level
clasifier.   this can also be run with one of the other subtopics 
note: the number of items selected may be less that 1200 after duplicates
are removed. 

##predictor8big.py
this version reads the model data from local files 
and makes predictions based on thresh value
pick a threshold around between 0.1 and 1.0
and some number of samples.   it will print the 
the samples that are above the threshold of confidence.
For data is uses the new arxiv data (not used in the training) and the science magazine data (also not used in the training).
A reasonable invocation is 

	ipython predictor8big.py all4 300 0.15

or

	ipython predictor8big.py bio 300 0.15

this is pulling a sample of the data and show you the results only if the prediction is above the threshold.   If you see nothing, they
the threshold  is probably too big.

##classifer
This directory hold the Dockerized version of the classifier that uses the rabbimq service and it assumes the scimlservice is running.
Noice in the Dockerfile there are a number of parameters that must be provided before you build the docker image.  They are

1. the address of rabbitmq service
2. the id of the rabbit user
3. the password for the rabbit user
4. the address of scimlservice

##scimlservice
This is the table web service component.    you will see that it need the first three items above (rabbitmq identity) and also the 
Azure account and access key so that it can put things in the table.  

##run\_sciml\_services.ipynb
This file requires that you install the ipython notebook.  it is a set of commands to the classifer service to wait on queues and to send data to those queues. 
