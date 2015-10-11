# this is the subarea classifier.  
#it does the following:
#  
#    1. When launched it opens a connection to the  roles  queue and wait for a topic.
#    2. When it receives the topic message from the  role  queue the classifier service, must initialize all the ML 
#        models for that topic from the saved trained models. (The models have been previously trained as described in 
#        the previous post and the trained models have been  pickled  and saved as blob in Azure blob storage.)
#    3  It then begins to scan the queue for that topic.   It pulls the json document objects from the queue and 
#        applies the classifier. It then packages up a new json object consisting of the main topic, new sub-classification, 
#        title and abstract. For example, if the item came from the  physics  queue and the classifier decides it in the 
#         subclass  General Relativity , then that is the sub-classification that in the object. 
#          (The classifier also has a crude confidence estimator. If it is not very certain the sub-classification 
#         is  General Relativity? .   If it is very uncertain and General Relativity is the best of the bad choices, 
#         then it is  General Relativity??? .) It then sends this object via a web service call to a local web service 
#        that is responsible for putting the object into an Azure table. (more on this step below. )
#    4. The service repeats 3 until it receives a  reset  message in the topic queue.   It then returns to the roles queue and step 1.

import pika
import socket
import requests
import time
#from azure.storage import TableService, Entity, BlobService
from pattern.web import Newsfeed
from pattern.web import cache
from pattern.en import parse, pprint, tag
from pattern.web import download, plaintext
import numpy as np
import nltk.stem
import pickle
import random
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from gensim import corpora, models, similarities
import urllib

#use this version if the config files are local.  You can
#get them from this url http://1drv.ms/1PCOT8l
#this is a one drive directory that is publicly readable
#basepath is the location of the config directory on the local
#machine
def read_config_from_local(subj, basepath):
    docpath =basepath+ "/config/config_"+subj+".json"
    with open(docpath, 'rb') as f:
        doc = f.read() 
    z =json.loads(doc)
    subject = z['subject']
    loadset = z['loadset']
    subtopics = []
    for w in z['supertopics']:
        subtopics.extend([(w[0], w[1])])
    return subject, loadset, subtopics


#use this version if you want to pull from my public (read-only) azure
#public blob store.  (note: no basepath parameter)
def read_config_azure_blob(subj, base):
    #base is the url for the azure blob store
    docpath =base+ "/config_"+subj+".json"
    f = urllib.urlopen(docpath)
    doc = f.read()
    z =json.loads(doc)
    subject = z['subject']
    loadset = z['loadset']
    subtopics = []
    for w in z['supertopics']:
        subtopics.extend([(w[0], w[1])])
    return subject, loadset, subtopics

def makeInvertedTrainingList(super_topics):
    #create list of all training set items
    #  (doc, subtopicname, subtopic-index)
    lis = []
    n = 0
    for top in super_topics:
        items = top[1]
        for i in range(0, len(items)):
            lis.extend([(items[i], top[0], n)])
        n = n+1
    return lis



#this version works for an arbitrary text string
def cosdistString(vectorizer, item, centroids):
    new_title_vec = vectorizer.transform([item])
    scores = []
    for i in range(0, len(centroids)):
        dist = np.dot(new_title_vec.toarray()[0], centroids[i].toarray()[0])
        scores.extend([(dist, i)])
    scores = sorted(scores,reverse=True)
    return scores

#for any new doc this returns list of tuples (distance, itemno, abstract for item)
#distance = cosdist(newdoc, item[itemno]) for all items in the training set.
#list sorted by distance 
#this is used by the "best of 3 algorithm" computed  in "makeguess" below  
def compdist(new_title_vec, indexlist, X, titles):
    similar = []
    for i in indexlist:
        if np.linalg.norm(X[i].toarray()) != 0.0:
            dist = np.dot(new_title_vec.toarray()[0],X[i].toarray()[0])
            similar.append((dist,i, titles[i]))
    similar = sorted(similar,reverse=True)
    return similar

def clean(doc):
    st = ""
    sciencestopwords = set([u'model','according', 'data', u'models', 'function', 'properties', 'approach', 'parameters',
                'systems', 'number', 'order', u'data', 'analysis', u'information', u'journal',
                'results','using','research', 'consumers', 'scientists', 'model', 'models', 'journal',
                'researchers','paper','new','study','time','case', 'simulation', u'simulation', 'equation',
                'based','years','better', 'theory', 'particular','many','due','much','set', 'studies', 'systems',
                'simple', 'example','work','non','experiments', 'large', 'small', 'experiment', u'experiments',
                'provide', 'analysis', 'problem', 'method', 'used', 'methods'])
    for (word, pos) in tag(doc):
        if pos in ["JJ", "NNS", "NN", "NNP"]:
            st = st+word+" "
        else:
            if st!= "":
                st = st[0:-1]+" "
                #print "got one"
    wordl = st.lower().split()
    s = ""
    for word in wordl:
        if word not in sciencestopwords:
            s = s+" "+word
    return s

#this is the best-of-3 algorithm.  it needs the training set to measure distances.
#the X and titles used here should be for the training list

def makeguess(statmt, km, vectorizer, lsi, dictionary, index_lsi, ldamodel, index_lda, X, titles):
    statement = clean(statmt)
    new_title_vec = vectorizer.transform([statement])
    new_title_label = km.predict(new_title_vec)
    similar_indicies = (km.labels_==new_title_label).nonzero()[0]
    similar = compdist(new_title_vec, similar_indicies, X, titles)
    kmeans_items = list(x[1] for x in similar)

    #now for lsi items
    new_title_vec_lsi = dictionary.doc2bow(statement.lower().split())
    new_title_lsi = lsi[new_title_vec_lsi]
    sims = index_lsi[new_title_lsi] # perform a similarity query against the corpus
    simlist = list(enumerate(sims))
    topten = sorted(simlist, key = lambda x: x[1], reverse = True)[0:30]
    lsi_items = list(x[0] for x in topten)

    #now do lda
    new_title_vec_lda = dictionary.doc2bow(statement.lower().split())
    new_title_lda = ldamodel[new_title_vec_lda]
    sims = index_lda[new_title_lda] # perform a similarity query against the corpus
    simlist = list(enumerate(sims))
    topten = sorted(simlist, key = lambda x: x[1], reverse = True)[0:30]
    lda_items = list(x[0] for x in topten)


    dist_lsi = compdist(new_title_vec, lsi_items, X, titles)
    
    dist_km = compdist(new_title_vec, kmeans_items, X, titles)
    
    dist_lda = compdist(new_title_vec, lda_items, X, titles)
    
    s = dist_lda + dist_km + dist_lsi
    #s is a list of tuples (distance, itemno, abstract for item) for the best guesess 
    # based on each of the three algorithms
    
    d1 = sorted(s, reverse=True)
    d = [x for x in d1 if x[0] > 0.00]
    #return the sorted list with duplicates removed
    notdups = []
    for x in d:
        if x not in notdups:
            notdups.extend([x])
    return notdups

#this is the routine that makes the classification.
#it first does the best of three and then the random forest.
#it then computes the centroid test. 
def bigpredict(statement, km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda,
               Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames ):

    bestof3l = makeguess(statement, km, vectorizer, lsi, dictionary, index_lsi,
                        lda, index_lda, Xtrain, traindocs)
    #bestof3l[0] is the tuple (distance, item index in training set, abstract for item) 
    #so bestof3l[0][1] is the index in the training set of the closest item.   
    #notice we never use the abstract of the best, but it is there if you want it.
    if len(bestof3l)> 0:
        best = trainlabel[bestof3l[0][1]]
    else:
        best = "?"
    if len(bestof3l) > 1:
        nextb = trainlabel[bestof3l[1][1]]
    else:
        nextb = "?"
        
    #next comput the random forest prediction
    X_new_counts = c_vectorizer.transform([statement])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predictedrf = clfrf.predict(X_new_tfidf)
    rf = groupnames[predictedrf[0]]

    #finally we compute the centroid metric.   in this case
    #we look at the distance of the best choice.  
    #if  best == rf we use this as an alternative
    #otherwise if our confidence is very high we
    #use this.  
    #but if best != rf and next best != rf
    #our cconfidence is very low. score = ??? 
    #if the distance is low but nextb == rf
    #we have slight lack of confidence. score = ?
    
    z = cosdistString(vectorizer, statement, new_centroids)
    if z[0][0] > 0.18 or best == rf:
        cent = groupnames[z[0][1]]
    else:
        if best != rf and nextb != rf:
            cent = "???"
        else:
            cent = "?"

    return rf, best, nextb, cent

#use this version of init if you are reading the models from a local
#directory.  you can find models in a zip file models.zip in 
#http://1drv.ms/1PCOT8l basepath should be the path to the location
#of the uncompressed models directory

def init_from_local(subj, basepath):
    
    subject, loadset, supertopics =read_config(subj, basepath)
        # for pushing to blob storage set basepath = ""
    cvectpath = basepath+"/models/count_vectorizer-"+subject+".p"
    tfidftranpath = basepath+"/models/tfidf_transformer-"+subject+".p"
    rfpath = basepath+"/models/random_forest-"+subject+".p"
    namspath = basepath+"/models/supertopic_names-"+subject+".p"
    GBpath = basepath+"/models/gradientboosting-"+subject+".p"
    vectpath = basepath+"/models/vectorizer-"+subject+".p"
    lsimodpath = basepath+"/models/lsimod-"+subject+".p"
    lsiindpath = basepath+"/models/lsiind-"+subject+".p"
    ldamodpath = basepath+"/models/ldamod-"+subject+".p"
    ldaindpath = basepath+"/models/ldaind-"+subject+".p"
    kmpath =  basepath+"/models/km-"+subject+".p"
    ncentpath = basepath+"/models/ncent-"+subject+".p"
    Xpath = basepath+"/models/Xtrain-"+subject+".p"
    trainsetpath = basepath+"/models/Tset-"+subject+".p"
    dictpath = basepath+"/models/Dict-"+subject+".p"

    with open(cvectpath, 'rb') as f:
		cvecb = f.read() 
    with open(vectpath, 'rb') as f:
		vecb = f.read() 
    with open(tfidftranpath, 'rb') as f:
		tranb = f.read() 
    with open(namspath, 'rb') as f:
		groupb = f.read() 
    with open(rfpath, 'rb') as f:
		rfbb = f.read() 
    with open(lsimodpath, 'rb') as f:
		lsimodb = f.read() 
    with open(lsiindpath, 'rb') as f:
		lsib = f.read() 
    with open(ldamodpath, 'rb') as f:
		ldamodb = f.read() 
    with open(ldaindpath, 'rb') as f:
		ldab = f.read() 
    with open(kmpath, 'rb') as f:
		kmpathb = f.read() 
    with open(ncentpath, 'rb') as f:
		ncentb = f.read() 
    with open(Xpath, 'rb') as f:
		Xb = f.read() 
    with open(trainsetpath, 'rb') as f:
		trainingb = f.read() 
    with open(dictpath, 'rb') as f:
		dictb = f.read() 
   

    c_vectorizer = pickle.loads(str(cvecb))
    tfidf_transformer = pickle.loads(str(tranb))
    groupnames = pickle.loads(str(groupb))
    clfrf = pickle.loads(str(rfbb))
    vectorizer = pickle.loads(str(vecb))
    new_centroids = pickle.loads(str(ncentb))
    index_lsi = pickle.loads(str(lsib))
    lsi = pickle.loads(str(lsimodb))
    index_lda = pickle.loads(str(ldab))
    lda = pickle.loads(str(ldamodb))
    km = pickle.loads(str(kmpathb))
    Xtrain = pickle.loads(str(Xb))
    trainingSets = pickle.loads(str(trainingb))
    dictionary = pickle.loads(str(dictb))

    traininglist = makeInvertedTrainingList(trainingSets)
    traindocs = [tex[0] for tex in traininglist]
    trainlabel = [tex[1] for tex in traininglist]
    traintarget = [tex[2] for tex in traininglist]

    return  km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames

def init_from_azure_blob(subj, basepath):
    base = "http://esciencegroup.blob.core.windows.net/scimlpublic"
    #ignoring the basepath param passed in
    subject, loadset, supertopics =read_config_azure_blob(subj, base)
        # for pushing to blob storage set basepath = ""
    cvectpath = base+"/count_vectorizer-"+subject+".p"
    tfidftranpath = base+"/tfidf_transformer-"+subject+".p"
    rfpath = base+"/random_forest-"+subject+".p"
    namspath = base+"/supertopic_names-"+subject+".p"
    GBpath = base+"/gradientboosting-"+subject+".p"
    vectpath = base+"/vectorizer-"+subject+".p"
    lsimodpath = base+"/lsimod-"+subject+".p"
    lsiindpath = base+"/lsiind-"+subject+".p"
    ldamodpath = base+"/ldamod-"+subject+".p"
    ldaindpath = base+"/ldaind-"+subject+".p"
    kmpath =  base+"/km-"+subject+".p"
    ncentpath = base+"/ncent-"+subject+".p"
    Xpath = base+"/Xtrain-"+subject+".p"
    trainsetpath = base+"/Tset-"+subject+".p"
    dictpath = base+"/Dict-"+subject+".p"
    

    cvecb = urllib.urlopen(cvectpath).read()
    vecb = urllib.urlopen(vectpath).read() 
    tranb = urllib.urlopen(tfidftranpath).read() 
    groupb = urllib.urlopen(namspath).read()  
    rfbb = urllib.urlopen(rfpath).read()  
    lsimodb = urllib.urlopen(lsimodpath).read()  
    lsib = urllib.urlopen(lsiindpath).read()  
    ldamodb = urllib.urlopen(ldamodpath).read()  
    ldab = urllib.urlopen(ldaindpath).read()  
    kmpathb = urllib.urlopen(kmpath).read()  
    ncentb = urllib.urlopen(ncentpath).read()  
    Xb = urllib.urlopen(Xpath).read()  
    trainingb = urllib.urlopen(trainsetpath).read()  
    dictb = urllib.urlopen(dictpath).read() 
    

    c_vectorizer = pickle.loads(str(cvecb))
    tfidf_transformer = pickle.loads(str(tranb))
    groupnames = pickle.loads(str(groupb))
    clfrf = pickle.loads(str(rfbb))
    vectorizer = pickle.loads(str(vecb))
    new_centroids = pickle.loads(str(ncentb))
    index_lsi = pickle.loads(str(lsib))
    lsi = pickle.loads(str(lsimodb))
    index_lda = pickle.loads(str(ldab))
    lda = pickle.loads(str(ldamodb))
    km = pickle.loads(str(kmpathb))
    Xtrain = pickle.loads(str(Xb))
    trainingSets = pickle.loads(str(trainingb))
    dictionary = pickle.loads(str(dictb))
    
    traininglist = makeInvertedTrainingList(trainingSets)
    traindocs = [tex[0] for tex in traininglist]
    trainlabel = [tex[1] for tex in traininglist]
    traintarget = [tex[2] for tex in traininglist]

    return  km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames

#webservice = "bogus"
def sendrest(st):
    try:
        hostnm = webservice
        #this version assumes there is a local version of the webservice
        addr ="http://"+hostnm+":16666/echo_string"
        print "sending to wbserv at "+addr
        payload = {'string':st}
        #print st
        r = requests.post(addr, data=payload)
        print "got reply:"+r.content
    except:
        print "error in sendrest"


#here we do the final classification.   body is document from the event queue.
# it represents the output of the toplevel classifier we must unpack the json doc. 

def classify(body, km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda,
    Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer,
    new_centroids, trainlabel, groupnames):
    #print body
    try:
        z = json.loads(str(body))
    except:
        print "bad json"
        z = {"title": "bad json", "doc": "forget me"}
    #print z["title"]
    try:
        rf, best, nextb, cent = bigpredict(z["doc"], km, vectorizer, lsi, dictionary, index_lsi,
        lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames )
        if cent == "???" or cent=="?":
            rf = rf+cent
    except:
        rf = "flub"
    return rf





def run(topic, basepath, pikaIP, rabbitid, rabbidpasswd):
    creds = pika.PlainCredentials(rabbitid, rabbidpasswd)

    #sendrest("starting init for topic "+topic)
    print "starting init for topic "+topic
    #note: there are two versions of init.  init_from_local and init_from_azure_blob
    km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames = init_from_azure_blob(topic, basepath)
    print "starting work now"
    # we are reading docs from the "topic" queue.  This was placed in this queue by the top toplevel
    # clssifier based on the rf or best score (or both)
    #The body should take the form:
    # {"truetop": "name of the catergory that this document should belong to", 
    #   "title": "title of doc", 
    #   "doc": "the abstract body",
    #   "cent": "the classification from centroid", 
    #   "rf": "random forest classification", 
    #   "best": "best of three classification"} 
    #below we only use the "doc" field for the next classification, but for the record, we push the entire doc into the table 
    #partition associated with that topic 
    while True:
        try:
            sendrest("listening on topic "+topic)
            connec = pika.BlockingConnection(pika.ConnectionParameters(pikaIP, 5672, '/', creds))
            chan = connec.channel()
            chan.queue_declare(queue=topic)
            while True:
                    method_frame, header_frame, body = chan.basic_get(topic)
                    if method_frame:
                            print body
                            if body=="reset":
                                chan.basic_ack(method_frame.delivery_tag)
                                chan.cancel()
                                connec.close()
                                sendrest("terminating "+topic)
                                return body
                            classres = classify(body, km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames)
                            #print classres
                            sendrest("("+topic+")"+classres+ " :"+body)
                            chan.basic_ack(method_frame.delivery_tag)
                    else:
                            time.sleep(1)
                            #print 'No message returned'
        except:
            print "oops"
            try:
                    chan.cancel()
                    connec.close()
            except:
                    print "double oops"
            time.sleep(5)

    requeued_messages = chan.cancel()
    connec.close()


def gettopic(pikaIP, rabbitid, rabbitpasswd):
    print pikaIP, rabbitid, rabbitpasswd
    creds = pika.PlainCredentials(rabbitid, rabbitpasswd)

    hostnm = socket.gethostname()
    #hostnm = socket.gethostbyaddr(socket.gethostname())[0]
    
    print hostnm
    #this version assumes there is a local version of the webservice
    addr ="http://"+hostnm+":16666/echo_string"
    queueset = set(["math", "bio", "phy", "finance", "compsci"] )
    #sendrest("start listening at "+str(hostnm))
    while True:
        try:
            connec = pika.BlockingConnection(pika.ConnectionParameters(pikaIP, 5672, '/', creds))
            chan = connec.channel()
            chan.queue_declare(queue='roles')
            while True:
                    method_frame, header_frame, body = chan.basic_get('roles')
                    if method_frame:
                        #print body
                        chan.basic_ack(method_frame.delivery_tag)
                        if body in queueset:
                            chan.cancel()
                            connec.close()
                            sendrest("topic for host :"+str(hostnm)+"="+body)
                            return body
                        else:
                           #chan.cancel()
                            #connec.close()
                            sendrest("unkown role command :"+str(hostnm)+"="+body)
                    else:
                        #chan.cancel()
                        #connec.close()
                        time.sleep(1)
                        #print 'No message returned'
        except:
            print "oops"
            try:
                    chan.cancel()
                    connec.close()
            except:
                    print "double oops"
            time.sleep(5)

    requeued_messages = chan.cancel()
    connec.close()

import sys
import getopt

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

def main(argv=None):
    
    if argv is None: 
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
            raise Usage(msg)
       
        rabbitid = "fee"
        rabbitpasswd = "fi"
        pikaIP = 'fum'
        if len(args)< 4:
            print "args needed are: IPaddress of Rabbitmq, rabbitid, rabbitpasswd, ip for webservice"
            return 2
        
        pikaIP = args[0]
        rabbitid = args[1]
        rabbitpasswd = args[2]
        global webservice
        webservice = args[3]
        print pikaIP, rabbitid, rabbitpasswd, webservice
        basepath = "./"
        while True:
            print "fetching topic"
            
            subj = gettopic(pikaIP, rabbitid, rabbitpasswd)
            run(subj, basepath, pikaIP, rabbitid, rabbitpasswd)

        print "all done."
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
