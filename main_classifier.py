#This is a version of the classifier that generates topic lists
#it is used to generate the entire set of predicted main topic
#classifications used by the performance analysis
#invoke it with two paramters like this
#ipython main_classifer.py all4  1200
#this will pick 1200 random items from arxiv and run the top level
#clasifier.   this can also be run with one of the other subtopics 
#note: the number of items selected may be less that 1200 after duplicates
#are removed. 

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
    
	
def load_data(name):
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

def load_data2(readtopics):
    titles, sitenames, disp_title = load_data("sciml_data_arxiv")
    bio = []
    sitenames2 = []
    disp_tit2 = []
    ls = [str(x) for x in readtopics]
    siteset = set(ls)
    for i in range(0, len(titles)):
        if sitenames[i]in siteset:
            bio.extend([titles[i]])
            sitenames2.extend([sitenames[i]])
            disp_tit2.extend([disp_title[i]+" ["+sitenames[i]+"]"])
        #else:
            #print sitenames[i], " ", disp_title[i]
    print len(bio)
    titles = bio
    sitenames = sitenames2
    disp_title = disp_tit2
    return titles, sitenames, disp_title
	
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
    


    
def fillTopicTables2(nounlist, sitenames, disp_title, supertopics, rate ):
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
            #note use of clean below.  this may not be necessary
            #as long as clean is applied when testing the doc
            #tupletitles.extend([clean(nounlist[items[w]])])
            #added the title to the text in a () tuple.
            #tupletitles.extend([nounlist[items[w]]])
            tupletitles.extend([(nounlist[items[w]], disp_title[items[w]])])
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
    new_title_vec = vectorizer.transform([item])
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
    if z[0][0] > 0.06:
        cent = groupnames[z[0][1]]
    else:
        cent = "??"
    
    return rf, best, nextb, cent
    
#--------- this is the main -----------------------
def run(subj, numsamples):
    
    base = "http://esciencegroup.blob.core.windows.net/scimlpublic"
    #ignoring the basepath param passed in
    subject, loadset, supertopics =read_config_azure_blob(subj, base)
    #note: there are two versions of init.  init_from_local and init_from_azure_blob
    basepath = "./"
    km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames = init_from_azure_blob(subj, basepath)
    print "starting work now"
    titles, sitenames, disp_title = load_data2(loadset)
    print "data loaded"
    
    
    print "filling test tables"
    num = numsamples
    maxn = len(titles)
    sublist = []
    sub_disp_title = []
    chosen = set()
    dupcnt = 0
    sub_sitenames = []
    if numsamples < maxn:
        for i in range(0,num):
            w = int(random.random()*maxn)
            if w not in chosen:
                chosen.add(w)
                sublist.extend([titles[w]])
                sub_disp_title.extend([disp_title[w]])
                sub_sitenames.extend([sitenames[w]])
            else:
                dupcnt = dupcnt+1
    else:
        for i in range(0, maxn):
            sublist.extend([titles[i]])
            sub_disp_title.extend([disp_title[i]])
            sub_sitenames.extend([sitenames[i]])
            
                
    print "dups ="+ str(dupcnt)
        
    print "random sublist len == "+str(len(sublist))
    
    fulllist = fillTopicTables2(sublist, sub_sitenames, sub_disp_title, supertopics, 1.0)
    biglist = makeInvertedTrainingList(fulllist)
    testdocsp = [tex[0] for tex in biglist]
    testdocs = [p[0] for p in testdocsp]
    testtitles = [p[1] for p in testdocsp]
    testlabel = [tex[1] for tex in biglist]
    testtarget = [tex[2] for tex in biglist]
    
    topl = []
    for top in supertopics:
        s = " "+top[0].lower()+". "
        for x in top[1]:
            s = s + x + " "
        topl.extend([(top[0], s)])
    topdict = {x[0]:x[1] for x in topl}  
    
    
    outTable = {gname:0 for gname in groupnames}
    outCount = {gname:0 for gname in groupnames}
    rfWin = {gname:0 for gname in groupnames}
    bestWin = {gname:0 for gname in groupnames}
    nextWin = {gname:0 for gname in groupnames}
    centWin = {gname:0 for gname in groupnames}
    falseposBes = {gname:0 for gname in groupnames}
    falseposRf = {gname:0 for gname in groupnames}
    bothAgr = {gname:0 for gname in groupnames}
    listTable = {gname:[] for gname in groupnames}
    num = 0.
    correct = 0.
    dataIn = []
    ans = []
    for i in range(0, len(testdocs)):
        statement = clean(testdocs[i])
        rf, best, nextb, cent = bigpredict(statement, km, vectorizer, lsi, dictionary, index_lsi, 
        lda, index_lda, Xtrain, traindocs, c_vectorizer, clfrf, tfidf_transformer, new_centroids, trainlabel, groupnames )
        #title is the text of abstract
        #sitename is the subtopic name 
        #disp_title is the actual title plus the arxiv key
        tup = (rf,best,cent, testdocs[i], testlabel[i], testtitles[i])
        #print "rf="+rf[0:5]+" best="+best[0:5]+" nextb="+nextb[0:5]+" cent="+cent[0:5]
        p = testlabel[i]
        num = num+1.0
        outCount[p] = outCount[p]+1
        if p == rf or p == best:
            correct = correct+1
            outTable[p] = outTable[p]+1
            #prstr = "best="+best[0:4] + "\tsecond="+nextb[0:4]+ "\trb="+t[0:4]+"\tsec="+sec[0:4]
            #print prstr + "\tfound = "+ findTopic(topdict, groupnames, q)[0:4] +" "+str(i)
            if p == best:
                bestWin[p] = bestWin[p]+1
            if p == rf:
                rfWin[p] = rfWin[p]+1
            if p == cent:
                centWin[p] = centWin[p]+ 1 
        #dataIn.extend([(nameIndex[best],nameIndex[nextb],nameIndex[t],nameIndex[sec])])
        #ans.extend([nameIndex[findTopic(topdict, groupnames, q)]])
        if( p == best and p == rf):
            bothAgr[p] = bothAgr[p]+ 1
        if (p != best):
            falseposBes[best] = falseposBes[best]+1 
        if (p != rf):
            falseposRf[rf] = falseposRf[rf]+1 
        # this will only work for the labled data.  save it under "true" category
        listTable[p].extend([tup])
        # the following is the way it should be done
        listTable[rf].extend([tup])
        if best != rf:
            listTable[best].extend([tup])
            
    print "correct rate = "+ str(100.0*correct/num)
    for nam in groupnames:
        if outCount[nam]> 0:
            ans = nam[0:9] + " \t= "+ str(100.0*outTable[nam]/outCount[nam])[0:5]
            ans = ans+  " \tboth agree ="+str(100.0*bothAgr[nam]/outCount[nam])[0:5]
            ans = ans+ " \t rf win = "+str(100.0*rfWin[nam]/outCount[nam])[0:5]
            ans = ans+ " \t best win = "+str(100.0*bestWin[nam]/outCount[nam])[0:5]
            ans = ans + "\t cent win " +str(100.0*centWin[nam]/outCount[nam])[0:5]
            ans = ans+ " \t false pos rf = "+str(100.0*falseposRf[nam]/num)[0:5]
            ans = ans+ " \t false pos best = "+str(100.0*falseposBes[nam]/num)[0:5]
            print ans
        print len(listTable[nam])
        path = "dump_"+subject+"_subtopic_"+nam+".p"
        pickle.dump(listTable[nam], open(path,"w"))
        print "done saving classifer results to dump files"

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
        if len(args) > 0:
            numsamples = int(args[1])
        else:
            numsamples = 600
        run(subj, numsamples)
        print "all done.  files now local"
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())
