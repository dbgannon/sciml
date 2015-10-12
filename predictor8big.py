#this version reads the model data from local files 
#and makes predictions based on thresh value
#pick a threshold around between 0.1 and 1.0
#and some number of samples.   it will print the 
#the samples that are above the threshold of confidence

import urllib
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



def read_config(subj, basepath):
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


def load_data(path, name):
    #ignoring path here.   see load data local
    base = "http://esciencegroup.blob.core.windows.net/scimlpublic"
    docpath =base+ "/"+name+".p"
    f = urllib.urlopen(docpath)
    z = f.read()
    print len(z)
    lst = pickle.loads(str(z))
    titles = []
    sitenames = []
    abstracts = []
    for i in range(0, len(lst)):
        titles.extend([lst[i][0]])
        sitenames.extend([lst[i][1]])
        abstracts.extend([lst[i][2]])
        
    print "done loading "+name
    return abstracts, sitenames, titles
    	
#for loading data from a local copy
def load_data_local(path, name):
    filename = path+name+".p"
    fileobj = open(filename, "rb")
    z = fileobj.read()
    lst = pickle.loads(str(z))
    titles = []
    sitenames = []
    abstracts = []
    for i in range(0, len(lst)):
        titles.extend([lst[i][0]])
        sitenames.extend([lst[i][1]])
        abstracts.extend([lst[i][2]])
        
    print "done loading "+filename
    return abstracts, sitenames, titles	

	

#returns a list consiting of tuples
#   (subtopic_name, [list of title-numbers in that subtopic])
def make_topic_lists_sets(nounlist, sitenames, disp_title, supertopics):
    topic_lists = []
    for topic in supertopics:
        #print topic[0]
        subtop_list = []
        topicset = set(topic[1])
        print topicset
        for i in range(0,len(nounlist)):
            t = disp_title[i]
            sn = sitenames[i]
            #found = False
            if sn in topicset:
                subtop_list.extend([i])
                
        topic_lists.extend([(topic[0], subtop_list)])
        #print subtop_list
    return topic_lists #returns the list of tuples for the training sets
    

# for each subtopic
#   (subtopic name, training set items, list of the ARXiV sub areas for this supertopic )
def fillTopicTables(nounlist, sitenames, disp_title, supertopics, rate ):
    toplsts = make_topic_lists_sets(nounlist, sitenames, disp_title, supertopics)
    super_topics = []   
    for i in range(0,len(toplsts)):
        topl = toplsts[i]
        area = topl[0]
        items = topl[1]
        z = int(rate*len(items))
        tupletitles = []
        tuplenums = []
        for r in range(0,z):
            w = int(r)
            tupletitles.extend([clean(nounlist[items[w]])])
            tuplenums.extend([items[w]])
        tup = (area, tupletitles, supertopics[i][1])
        super_topics.extend([tup])
    
    for top in super_topics:
        print top[0] + " "+ str(len(top[1]))+ " " + str(top[2])

    return super_topics
	
def makeInvertedTrainingList(super_topics):
    #create list of all training set items
    #  (doc, docno, subtopicname, subtopic-index)
    lis = []
    n = 0
    for top in super_topics:
        items = top[1]
        for i in range(0, len(items)):
            lis.extend([(items[i], top[0], n)])
        n = n+1
    return lis

def findTopic(topdict, groupnames, found):
    for nm in groupnames:
        if topdict[nm].find(found)> 0:
            return nm
    return "none"

#for an item in the titles list compare its nonlist vector to the list of centroid and return 
#the sorted list (closest first)
def cosdist(vectorizer, itemno, centroids, nounlist):
    new_title_vec = vectorizer.transform([nounlist[itemno]])
    #new_title_vec = vectorizer.transform([titles[itemno]])
    scores = []
    for i in range(0, len(centroids)):
        dist = np.dot(new_title_vec.toarray()[0], centroids[i].toarray()[0])
        scores.extend([(dist, i)])
    scores = sorted(scores,reverse=True) 
    return scores

#this version works for an arbitrary text string
def cosdistString(vectorizer, item, centroids):
    new_title_vec = vectorizer.transform([clean(item)])
    scores = []
    for i in range(0, len(centroids)):
        dist = np.dot(new_title_vec.toarray()[0], centroids[i].toarray()[0])
        scores.extend([(dist, i)])
    scores = sorted(scores,reverse=True) 
    return scores

#returns list of tuples (distance, sitename, itemno, abstract for item)
def compdist(new_title_vec, indexlist, X, titles):
    similar = []
    for i in indexlist:
        if np.linalg.norm(X[i].toarray()) != 0.0:
            #dist = np.linalg.norm((new_title_vec-X[i]).toarray())
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
    	
#the X and titles used here should be for the training list
def makeguess(statmt, km, vectorizer, lsi, dictionary, index_lsi, ldamodel, index_lda, X, titles):
    #statement = clean(statmt)
    statement = statmt
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
    #dist_lsi = []
    dist_km = compdist(new_title_vec, kmeans_items, X, titles)
    #dist_km = []
    dist_lda = compdist(new_title_vec, lda_items, X, titles)
    #dist_lda = []
    s = dist_lda + dist_km + dist_lsi
    #print s
    d1 = sorted(s, reverse=True)
    d = [x for x in d1 if x[0] > 0.00]

    notdups = []
    for x in d:
        if x not in notdups:
            notdups.extend([x])
    return notdups

def bigpredict(statement, km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, 
               Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames ):
    
    bestof3l = makeguess(statement, km, vectorizer, lsi, dictionary, index_lsi, 
                        lda, index_lda, Xtrain, traindocs)
    if len(bestof3l)> 0:
        best = trainlabel[bestof3l[0][1]]
    else:
        best = "?"
    if len(bestof3l) > 1:
        nextb = trainlabel[bestof3l[1][1]]
    else:
        nextb = "?"
    
    X_new_counts = c_vectorizer.transform([statement])
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predictedrf = clfrf.predict(X_new_tfidf)
    rf = groupnames[predictedrf[0]]
    
    z = cosdistString(vectorizer, statement, new_centroids)
    cent = groupnames[z[0][1]]
    
    return rf, best, nextb, cent, z[0][0]

def init_models(subj, basepath):
    
    base = "http://esciencegroup.blob.core.windows.net/scimlpublic"
    subject, loadset, supertopics =read_config_azure_blob(subj, base)
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
   
#--------- this is the main -----------------------
def run(subj, numsamples, thresh):
    
    km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames = init_models(subj, "./" )

    
    titles1, sitenames1, disp_title1 = load_data("./","sciml_data_arxiv_new_9_28_15")
    titles2, sitenames2, disp_title2 = load_data("./","sciml_data_scimags")
    titles = titles1+titles2
    sitenames = sitenames1+sitenames2
    disp_title = disp_title1+disp_title2
    print "data loaded"
    
    print "filling test tables"
    num = numsamples
    max = len(titles)
    print "titles len ="+str(max)
    sublist = []
    chosen = set()
    dupcnt = 0
    sub_sitenames = []
    for i in range(0,num):
        w = int(random.random()*max)
        if w not in chosen:
            chosen.add(w)
            sublist.extend([titles[w]])
            sub_sitenames.extend([sitenames[w]])
    
        else:
            dupcnt = dupcnt+1
    print "dups ="+ str(dupcnt)
        
    print "random sublist len == "+str(len(sublist))
    
    testdocs = sublist
    
    
    
    listTable = {gname:[] for gname in groupnames}
    num = 0.
    correct = 0.
    dataIn = []
    ans = []
    for i in range(0, len(testdocs)):
        #statement = clean(testdocs[i])
        statement = testdocs[i]
        rf, best, nextb, cent, centval = bigpredict(statement, km, vectorizer, lsi, dictionary, index_lsi, 
        lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames )
        tup = (rf,best,nextb, cent, i, testdocs[i], sitenames[i])
        if centval > thresh:
            print "----------------------------------------------------------"
            print "rf="+rf[0:5]+" best="+best[0:5]+" cent="+cent[0:5]+ " sitename= "+sub_sitenames[i]
            print "cent val ="+str(centval)
            print " abstract " + statement[0:100]
        
  

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
        print "calling run with subect name =" + str(args)
        subj = args[0]
        if len(args)> 1:
            thresh = float(args[2])
        else:
            thresh = 0.06
        if len(args) > 0:
            numsamples = int(args[1])
        else:
            numsamples = 600
        print "running for subj="+subj+" with confidence thresh="+str(thresh)+ " for "+ str(numsamples)+" random samples"
        run(subj, numsamples, thresh)
        print "all done"
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sysip.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
