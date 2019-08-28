# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""  

from pathlib import Path

import glob
import os

from bs4 import BeautifulSoup as bs
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import re
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import string
from sklearn.feature_extraction.text import TfidfVectorizer

path1 = Path("datasetSTKI/ekonomi/") 
path2 = Path("datasetSTKI/politik/")
path3 = Path("datasetSTKI/olahraga/")   

#frequency = {}
stem = {}

#file XML Ekonomi
for filename in glob.glob(os.path.join(path1, "*.xml")):
    with open(filename, 'r', encoding="utf8") as open_file:
        content = open_file.read()
        soup = bs(content,"lxml")
        label = soup.find("id").get_text()
        print (label)
        isi = soup.find("isi").get_text()
        #print (isi)
#case folding
        lowercase = isi.lower()
        #print (lowercase)
#symbol removal
        rmvsym = re.sub(r'[^\w\s]', '', lowercase)
        #print (rmvsym)        
#stopword removal      
        factory = StopWordRemoverFactory()
        stopword = factory.create_stop_word_remover()
        stoprmv = stopword.remove(str(rmvsym))
        #print (stoprmv)        
#Tokenization
        tokens = word_tokenize(stoprmv)
        #print (tokens)
#stemming
        stemFactory = StemmerFactory()
        stemmer = stemFactory.create_stemmer()
        for word in tokens:
            root = stemmer.stem(word)
            #print (root)
#count frequency
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(tokens)
    print(vectorizer.get_feature_names())
    print(X.shape)
            
from sklearn import svm
clf_svm = svm.LinearSVC()
X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, training_data.flag, test_size=0.25, random_state=42)
clf_svm.fit(X_train_tfidf, training_data.flag)
pickle.dump(clf_svm, open("svm.pkl", "wb"))
