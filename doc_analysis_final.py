#generic scifeed document analyzer
#read in the discipline specific config file and it will load the data and go from there.
#it will generate all the models used by the classifier.
#you invoke it with two arguments:  the topic which is one of
# all4, bio, compsci, finance, math, phy
#an integer which is the max size of the training set. note:
#if the set of documents in a subcategory is less than the max size the entire set
#of documents in that subcategory is selected as the training set.  

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
        if sitenames[i] in siteset:
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
	
def buildVectorizer(bio):
    nounlist = []
    for doc in bio:
        st = ""
        for (word, pos) in tag(doc):
            if pos in ["JJ", "NNS", "NN", "NNP"]:
                st = st+word+" "
            else:
                if st!= "":
                    st = st[0:-1]+" "
                    #print "got one"
        nounlist.extend([st])
    sciencestopwords = set([u'model','according', 'data', u'models', 'function', 'properties', 'approach', 'parameters', 
                    'systems', 'number', 'order', u'data', 'analysis', u'information', u'journal',
                    'results','using','research', 'consumers', 'scientists', 'model', 'models', 'journal',
                    'researchers','paper','new','study','time','case', 'simulation', u'simulation', 'equation',
                    'based','years','better', 'theory', 'particular','many','due','much','set', 'studies', 'systems',
                    'simple', 'example','work','non','experiments', 'large', 'small', 'experiment', u'experiments',
                    'provide', 'analysis', 'problem', 'method', 'used', 'methods'])
    #now doing the new vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    english = nltk.corpus.stopwords.words('english')
    newstop = english+list(sciencestopwords) 
    vectorizer = TfidfVectorizer(min_df=1, max_df=.5, stop_words=newstop, decode_error='ignore')
    X = vectorizer.fit_transform(nounlist)
    Xinv = vectorizer.inverse_transform(X)
        #X is a sparse matrix of docs x vocab size (7638). 
    #so X[doc_num] is the sparse vector of its words. 
    #the ||X[doc_num]|| = 1 there are 7638 unique words and 755 docs. with a total number of 38888 non-zeros.
    #Xinv[doc_num] is the list of words in the doc.
     
    return nounlist, vectorizer, X, Xinv

#returns a list consiting of tuples
#   (subtopic_name, [list of title-numbers in that subtopic])
def make_topic_lists_sets(nounlist, disp_title, supertopics):
    topic_lists = []
    for topic in supertopics:
        #print topic[0]
        subtop_list = []
        for i in range(0,len(nounlist)):
            t = disp_title[i]
            for j in topic[1]:
                x = t.find(j) 
                if x  > 0:
                    subtop_list.extend([i])
        topic_lists.extend([(topic[0], subtop_list)])
        #print subtop_list
    return topic_lists #returns the list of tuples for the training sets
 
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
       

# for each subtopic
#   (subtopic name, training set items, list of the ARXiV sub areas for this supertopic )
def fillTopicTables(nounlist, disp_title, supertopics, rate, maxsample ):
    toplsts = make_topic_lists_sets(nounlist, disp_title, supertopics)
    super_topics = []   
    for i in range(0,len(toplsts)):
        topl = toplsts[i]
        area = topl[0]
        items = topl[1]
        #print "maxsample = "+str(maxsample)
        z = min([maxsample, int(rate*len(items))])
        tupletitles = []
        tuplenums = []
        print "z ="+ str(z)
        for r in range(0,z):
            w = r
            tupletitles.extend([clean(nounlist[items[w]])])
            tuplenums.extend([items[w]])
        tup = (area, tupletitles, supertopics[i][1])
        #print "len titles ="+ str(len(tupletitles))
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
       
def compute_centroid(items, vectorizer):
    stmt = items[0]
    #stmt = titles[items[0]]
    count = len(items)
    #print stmt
    #print count
    vec = vectorizer.transform([stmt])[0]
    for i in range(1,count):
        stmt = items[i]
        vec2 = vectorizer.transform([stmt])[0]
        vec = vec+vec2
    z = np.linalg.norm(vec.toarray())
    return vec/z

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
    
def big_build_analizers( subj, maxsample):  
    print "doing subject "+ subj + "with max size of training set per category "+str(maxsample)
    base = "http://esciencegroup.blob.core.windows.net/scimlpublic"
    subject, loadset, supertopics =read_config_azure_blob(subj, base)
    #this version will write all the pickled model files to the local directory
    basepath = "./"
    cvectpath = basepath+"count_vectorizer-"+subject+".p"
    tfidftranpath = basepath+"tfidf_transformer-"+subject+".p"
    RFpath = basepath+"random_forest-"+subject+".p"
    namspath = basepath+"supertopic_names-"+subject+".p"
    GBpath = basepath+"gradientboosting-"+subject+".p"
    vectpath = basepath+"vectorizer-"+subject+".p"
    lsimodpath = basepath+"lsimod-"+subject+".p"
    lsiindpath = basepath+"lsiind-"+subject+".p"
    ldamodpath = basepath+"ldamod-"+subject+".p"
    ldaindpath = basepath+"ldaind-"+subject+".p"
    kmpath =  basepath+"km-"+subject+".p"
    ncentpath = basepath+"ncent-"+subject+".p"
    Xpath = basepath+"Xtrain-"+subject+".p"
    trainsetpath = basepath+"Tset-"+subject+".p"
    dictpath = basepath+"Dict-"+subject+".p"
    
    supertopic_names = [ x[0] for x in supertopics]
    pickle.dump( supertopic_names, open(namspath, "wb" ) )
     
    titles, sitenames, disp_title = load_data2(loadset)
    print "data loaded"
    #create a version of the docs "nounlist" where each item is filtered
    #through the stoplist.  Vectorizer is built from that and Xinv is a list
    #where each element is the an array of the words in that doc.
    nounlist, vectorizer, X, Xinv = buildVectorizer(titles)
    dictwords = [ item.tolist() for item in Xinv]
    dictionary = corpora.Dictionary(dictwords)
    print(dictionary)
    print "dictionary built"
    pickle.dump(dictionary, open(dictpath, "wb"))
   
    
    dictcorpus = [dictionary.doc2bow(text) for text in dictwords]
    tfidf = models.TfidfModel(dictcorpus)
    
    #create training set
    print "creating training set"
    trainingSets = fillTopicTables(titles, disp_title, supertopics, 0.75, maxsample)
    traininglist = makeInvertedTrainingList(trainingSets)
    traindocs = [tex[0] for tex in traininglist]
    trainlable = [tex[1] for tex in traininglist]
    traintarget = [tex[2] for tex in traininglist]
    corpus = [dictionary.doc2bow(text.lower().split()) for text in traindocs]
    corpus_tfidf = tfidf[corpus]
    
    #create lsi
    print "creating lsi model"
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10) # initialize an LSI transformation
    corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    index_lsi = similarities.MatrixSimilarity(corpus_lsi)

    #create lda
    print "creating lda model"
    lda = models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes = 10, iterations = 500) # initialize an LSI transformation
    corpus_lda = lda[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
    index_lda = similarities.MatrixSimilarity(corpus_lda)
    
    #create km for full list of documents 
    num_clusters = 10
    km = KMeans(n_clusters=num_clusters, init='random', n_init=1, verbose=1, tol=.00001)
    Xtrain = vectorizer.fit_transform(traindocs)
    Xinvtrain = vectorizer.inverse_transform(Xtrain)
    print Xtrain.shape
    print Xtrain[1].shape
    km.fit(Xtrain)
    #print Xtrain
    print "k-means analizer built"
    pickle.dump(Xtrain, open(Xpath, "wb"))
    pickle.dump(trainingSets, open(trainsetpath, "wb"))
    
    #dumping km, lda, lsi
    pickle.dump(km, open(kmpath, "wb"))
    pickle.dump(lsi, open(lsimodpath, "wb"))
    pickle.dump(index_lsi, open(lsiindpath, 'wb'))
    pickle.dump(lda, open(ldamodpath, "wb"))
    pickle.dump(index_lda, open(ldaindpath, "wb"))
    pickle.dump( vectorizer, open( vectpath, "wb" ) )
    
    #here is where we create the new centroid list
    new_centroids = []
    for ts in trainingSets:
        print "selecting friends for "+ts[0]
        newfr = []
        for x in ts[1]:
            gl = makeguess(x, km, vectorizer, lsi, dictionary, index_lsi, lda, index_lda, Xtrain, traindocs)
            l = min([10, len(gl)])
            items = [traindocs[ir[1]] for ir in gl[:l]]
            newfr = newfr + items
        sfr = set(newfr)
        newfr = list(sfr)
        print "len newfriends = "+ str(len(newfr))
        print "computing cetroid for " + ts[0]
        cent = compute_centroid(newfr, vectorizer)
        new_centroids.extend([cent])
        
    pickle.dump(new_centroids, open(ncentpath, "wb"))
      
    training_set_data = np.array(traindocs)
    training_set_target = traintarget
    print "total training set size = "+ str(len(training_set_data))
    
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(training_set_data)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clfrf = RandomForestClassifier(n_estimators = 100)
    clfrf.fit(X_train_tfidf, training_set_target)
    
    print "dumping vectorizer, transformer and clfrf"
    pickle.dump( count_vect, open( cvectpath, "wb" ) )
    pickle.dump( tfidf_transformer, open( tfidftranpath, "wb" ) )
    pickle.dump( clfrf, open( RFpath, "wb" ) )
    
    return

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
            maxsample = int(args[1])
        else:
            maxsample = 2000
        big_build_analizers(subj, maxsample)
        print "all done.  files now stored locally"
    except Usage, err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, "for help use --help"
        return 2

if __name__ == "__main__":
    sys.exit(main())

