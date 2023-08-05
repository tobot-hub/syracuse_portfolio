#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


from nltk.corpus import PlaintextCorpusReader
def get_df():
    spam_corpus = PlaintextCorpusReader('/kaggle/input/ham-and-spam-emails/corpus/spam', '.*', encoding='latin2')
    body = [spam_corpus.raw(fileids=[f]) for f in spam_corpus.fileids()]
    spam = [1 for f in spam_corpus.fileids()]
    name = spam_corpus.fileids()
    spam_df = pd.DataFrame(
        {'email': name,
        'body': body,
        'spam': spam})
    
    ham_corpus = PlaintextCorpusReader('/kaggle/input/ham-and-spam-emails/corpus/ham', '.*', encoding='latin2')
    body = [ham_corpus.raw(fileids=[f]) for f in ham_corpus.fileids()]
    spam = [0 for f in ham_corpus.fileids()]
    name = ham_corpus.fileids()
    ham_df = pd.DataFrame(
        {'email': name,
        'body': body,
        'spam': spam})
    df = pd.concat([spam_df,ham_df],ignore_index=True)
    df['date'] = pd.to_datetime(df.email.str.split('.',expand=True)[1])
    return df


# ## TfIDF vectorization

# In[4]:


from sklearn.feature_extraction.text import TfidfVectorizer
def vectorize_emails(df, max_features=200, ngram=1, max_df=1.0, sw=None, tokenizer=None):
    vectorizer = TfidfVectorizer(max_features=max_features, 
                            ngram_range=(1,ngram),
                            max_df=max_df,
                                stop_words=sw,
                                tokenizer=tokenizer)
    doc_vec = vectorizer.fit_transform(df.body)
    vocab_df = vocab_df = pd.DataFrame(data=doc_vec.toarray(), columns=vectorizer.get_feature_names())
    res = pd.concat([df, vocab_df], axis=1)
    return res


# ## Binary vectorization

# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
def bin_vectorize_emails(df, max_features=200, ngram=1, max_df=1.0, sw=None, tokenizer=None):
    vectorizer = CountVectorizer(max_features=max_features, 
                            ngram_range=(1,ngram),
                            max_df=max_df,
                            stop_words=sw,
                            tokenizer=tokenizer,
                            binary=True)
    doc_vec = vectorizer.fit_transform(df.body)
    vocab_df = vocab_df = pd.DataFrame(data=doc_vec.toarray(), columns=vectorizer.get_feature_names())
    res = pd.concat([df, vocab_df], axis=1)
    return res


# ## Predicting Spam or Ham

# In[6]:


from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer

from sklearn.naive_bayes import GaussianNB

from sklearn.inspection import permutation_importance

class spam_predicter:
    def __init__(self, df, target='spam'):
        self.X = df.select_dtypes(['number']).drop([target], axis=1)
        self.y = df[target]
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.features = X_train.columns

        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        
        self.pred = self.model.predict(self.X_test)
        
    def score(self):
        scoring = {'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
            'accuracy' : make_scorer(accuracy_score)}
        results = cross_validate(self.model,self.X,self.y,cv=10,scoring=scoring)
        print('Precision:\t',round(results['test_precision'].mean(),3))
        print('Recall:\t\t',round(results['test_recall'].mean(),3))
        print('F1 Score:\t',round(results['test_f1_score'].mean(),3))
        print('Accuracy:\t',round(results['test_accuracy'].mean(),3))
    
    def feature_importance(self, num=10):
        importances = permutation_importance(self.model, self.X_test, self.y_test).importances_mean
        indices = np.argsort(importances)[::-1]

        print("Feature ranking:")
        for i in range(min(num, self.X_test.shape[1])):
            print("%d. %s (%f)" % (i + 1, self.X_test.columns[indices[i]], importances[indices[i]]))
    


# In[7]:


from xgboost import XGBClassifier
class xg_spam_predicter:
    def __init__(self, df, target='spam'):
        for col in df.columns:
            if ('[' in col) or (']' in col) or ('<' in col):
                df.drop([col], inplace=True, axis=1)
        self.X = df.select_dtypes(['number']).drop([target], axis=1)
        self.y = df[target]
        X_train, self.X_test, y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42, stratify=self.y)
        self.features = X_train.columns

        self.model = XGBClassifier(max_depth=6,
                                  gamma=0.45,
                                  subsample=0.95,
                                  colsample_bytree=0.6,
                                  scale_pos_weight=1.4,
                                  n_estimators=500)
        self.model.fit(X_train, y_train)
        
        self.pred = self.model.predict(self.X_test)
        
    def score(self):
        scoring = {'precision' : make_scorer(precision_score),
           'recall' : make_scorer(recall_score), 
           'f1_score' : make_scorer(f1_score),
            'accuracy' : make_scorer(accuracy_score)}
        results = cross_validate(self.model,self.X,self.y,cv=10,scoring=scoring)
        print('Precision:\t',round(results['test_precision'].mean(),3))
        print('Recall:\t\t',round(results['test_recall'].mean(),3))
        print('F1 Score:\t',round(results['test_f1_score'].mean(),3))
        print('Accuracy:\t',round(results['test_accuracy'].mean(),3))
    
    def feature_importance(self, num=10):
        importances = permutation_importance(self.model, self.X_test, self.y_test).importances_mean
        indices = np.argsort(importances)[::-1]

        print("Feature ranking:")
        for i in range(min(num, self.X_test.shape[1])):
            print("%d. %s (%f)" % (i + 1, self.X_test.columns[indices[i]], importances[indices[i]]))


# In[8]:


#XGB parameters optimized on F1 score
{'colsample_bytree': 0.6,
 'gamma': 0.45,
 'max_depth': 5.0,
 'min_child_weight': 2.0,
 'n_estimators': 500.0,
 'scale_pos_weight': 1.4,
 'subsample': 0.95}


# ## Inspecting new features

# In[9]:


from tabulate import tabulate
def sample_new_features(old_df, new_df, size=5):
    col_set = set(old_df.columns)
    new_col_set = set(new_df.columns)
    new_cols = list(new_col_set.difference(col_set))[:size]
    print(tabulate(new_df[new_cols].head(), new_cols, tablefmt="fancy_grid"))


# In[10]:


df = get_df()


# In[11]:


df.head(5)


# ## Baseline - Top 200 words

# In[12]:


res = vectorize_emails(df)
cols = res.columns[50:55]
print(tabulate(res[cols].head(), cols, tablefmt="fancy_grid"))

baseline_model = spam_predicter(res)
baseline_model.score()


# ## Top 1000 words

# In[13]:


res = vectorize_emails(df, max_features=1000)
orig_df = res.copy()
sample_new_features(vectorize_emails(df), res)

model = spam_predicter(res)
model.score()
#model.feature_importance()


# ## Top 1000 with Bigrams and Trigrams

# In[14]:


res = vectorize_emails(df, max_features=1500, ngram=2)
sample_new_features(orig_df, res)

model = spam_predicter(res)
model.score()


# ## Top 1000 filtering stop words

# In[15]:


from nltk.corpus import stopwords
sw = set(stopwords.words("english"))
res = vectorize_emails(df, max_features=1000, sw=sw)
sample_new_features(orig_df, res)

model = spam_predicter(res)
model.score()
#model.feature_importance()


# ## Top 1000, lemmatized

# In[ ]:


from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
lemmatizer = WordNetLemmatizer()
def lemm_tokenizer(doc):
    return [lemmatizer.lemmatize(w) for w in word_tokenize(doc)]

res = vectorize_emails(df, max_features=1000, tokenizer=lemm_tokenizer)
sample_new_features(orig_df, res)

model = spam_predicter(res)
model.score()


# ## Top 1000, stemmed

# In[ ]:


from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
stemmer = PorterStemmer()
def stem_tokenizer(doc):
    return [stemmer.stem(w) for w in word_tokenize(doc)]

res = vectorize_emails(df, max_features=1000, tokenizer=stem_tokenizer)
sample_new_features(orig_df, res)

model = spam_predicter(res)
model.score()


# ## Top 1000 with doc embeddings
# #### Warning this takes 5 minutes

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_sm")

def add_doc_vecs(df):
    def get_doc_vec(doc):
        word_vecs = [token.vector for token in doc if token.has_vector]
        return pd.DataFrame(word_vecs).mean()
    doc_vecs = [get_doc_vec(doc) for doc in nlp.pipe(df.body)]
    cols = ['vec_' + str(i) for i in range(96)]
    vec_df = pd.DataFrame(doc_vecs)
    vec_df = (vec_df - vec_df.min())/(vec_df.max() - vec_df.min())
    vec_df.columns = cols
    res = pd.concat([df, vec_df], axis=1)
    return res


# In[ ]:


res = vectorize_emails(df, max_features=1000)
res = add_doc_vecs(res)
sample_new_features(orig_df, res)

model = spam_predicter(res)
model.score()


# ## 1000 with doc features

# In[ ]:


from nltk import word_tokenize
def get_doc_features(df):
    df['num_chars'] = df.body.str.len()
    df.num_chars = (df.num_chars - df.num_chars.min())/(df.num_chars.max() - df.num_chars.min())
    df['num_words'] = df.body.apply(lambda doc: len(word_tokenize(doc)))
    df['word_diversity'] = df.body.apply(lambda doc: len(set(word_tokenize(doc))))/df.num_words
    df.num_words = (df.num_words - df.num_words.min())/(df.num_words.max() - df.num_words.min())
    df['word_len'] = df.num_chars/df.num_words
    df.word_len = (df.word_len - df.word_len.min())/(df.word_len.max() - df.word_len.min())
    df.fillna(0, inplace=True)
    return df


# In[ ]:


res = vectorize_emails(df, max_features=1000)
res = get_doc_features(res)
sample_new_features(orig_df, res)

model = spam_predicter(res)
model.score()


# ## 200 with Binary Vectorization

# In[ ]:


res = bin_vectorize_emails(df)
cols = res.columns[50:55]
print(tabulate(res[cols].head(), cols, tablefmt="fancy_grid"))

model = spam_predicter(res)
model.score()


# ## POS with tfidf

# In[ ]:


from nltk import word_tokenize, pos_tag, sent_tokenize
def pos_features_norm(df, ngram=1):
    def tag_text(text):
        pos_sents = [pos_tag(word_tokenize(sent)) for sent in sent_tokenize(text)]
        return ' '.join([pair[1] for sent in pos_sents for pair in sent])
    df['tagged_sent'] = df.body.apply(tag_text)
    vectorizer = TfidfVectorizer(ngram_range=(1,ngram), min_df=2)
    pos_vec = vectorizer.fit_transform(df.tagged_sent)
    pos_feat = ['pos_' + feat for feat in vectorizer.get_feature_names()]
    pos_df = pd.DataFrame(data=pos_vec.toarray(), columns=pos_feat)
    res = pd.concat([df, pos_df], axis=1)
    return res


# In[ ]:


res = vectorize_emails(df, max_features=1000)
res = pos_features_norm(res)
model = spam_predicter(res)
model.score()


# In[ ]:


sample_new_features(orig_df, res)


# ## POS with ratios

# In[ ]:


#counts to number of words ratio approach
def pos_features_alt(df):

    tagged_text = lambda x: pos_tag(word_tokenize(x))
    df['tagged_sent'] = df['body'].apply(tagged_text)
    df['num_words'] = df.body.apply(lambda doc: len(word_tokenize(doc)))

    def NounCounter(x):
        nouns = []
        for (word, pos) in x:
            if pos.startswith("NN"):
                nouns.append(word)
        return nouns

    df["nouns"] = df["tagged_sent"].apply(NounCounter)
    df["noun_count"] = df["nouns"].str.len()
    df["noun_norm"] = df["noun_count"].div(df['num_words'], axis=0)

    def ProperNounCounter(x):
        propernouns = []
        for (word, pos) in x:
            if pos.startswith("NNP"):
                propernouns.append(word)
        return propernouns

    df["propernouns"] = df["tagged_sent"].apply(ProperNounCounter)
    df["propernoun_count"] = df["propernouns"].str.len()
    df["propernoun_norm"] = df["propernoun_count"].div(df['num_words'], axis=0).replace(np.inf, np.nan).fillna(0)


    def VerbCounter(x):
        verbs = []
        for (word, pos) in x:
            if pos.startswith("V"):
                verbs.append(word)
        return verbs

    df["verbs"] = df["tagged_sent"].apply(VerbCounter)
    df["verb_count"] = df["verbs"].str.len()
    df["verb_norm"] = df["verb_count"].div(df['num_words'], axis=0)

    def AdjectivesCounter(x):
        adjectives = []
        for (word, pos) in x:
            if pos.startswith("JJ"):
                adjectives.append(word)
        return adjectives

    df["adjectives"] = df["tagged_sent"].apply(AdjectivesCounter)
    df["adjective_count"] = df["adjectives"].str.len()
    df["adjective_norm"] = df["adjective_count"].div(df['num_words'], axis=0)

    def AdverbCounter(x):
        adverbs = []
        for (word, pos) in x:
            if pos.startswith("RB"):
                adverbs.append(word)
        return adverbs

    df["adverbs"] = df["tagged_sent"].apply(AdverbCounter)
    df["adverb_count"] = df["adverbs"].str.len()
    df["adverb_norm"] = df["adverb_count"].div(df['num_words'], axis=0)

    def WhCounter(x):
        wh = []
        for (word, pos) in x:
            if pos.startswith("W"):
                wh.append(word)
        return wh

    df["wh"] = df["tagged_sent"].apply(WhCounter)
    df["wh_count"] = df["wh"].str.len()
    df["wh_norm"] = df["wh_count"].div(df['num_words'], axis=0).replace(np.inf, np.nan).fillna(0)

    df['noun_verb_ratio'] = df["noun_count"].div(df["verb_count"], axis = 0).replace(np.inf, np.nan).fillna(0)
    df['noun_adj_ratio'] = df["noun_count"].div(df["adjective_count"], axis = 0).replace(np.inf, np.nan).fillna(0)
    df['propernoun_noun_ratio'] = df["propernoun_count"].div(df['noun_count'], axis=0).replace(np.inf, np.nan).fillna(0)
    return df


# In[ ]:


res = vectorize_emails(df, max_features=1000)
res = pos_features_alt(res)
model = spam_predicter(res)
model.score()


# ## Sentiment

# In[ ]:


import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def sentiment_features(df):
    def get_mean(text, feature):
        vals = [analyzer.polarity_scores(sent)[feature] for sent in sent_tokenize(text)]
        return sum(vals)/len(vals)
    df['compound'] = df.body.apply(get_mean, feature='compound')
    df['neg'] = df.body.apply(get_mean, feature='neg')
    df['neu'] = df.body.apply(get_mean, feature='neu')
    df['pos'] = df.body.apply(get_mean, feature='pos')
    return df


# In[ ]:


res = vectorize_emails(df, max_features=1000)
res = sentiment_features(res)
model = spam_predicter(res)
model.score()


# In[ ]:


sample_new_features(orig_df, res)


# ## Sentiment with textblob

# In[ ]:


from textblob import TextBlob

def sentiment_features2(df):
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
  
    #Create a function to get the polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
  
    #Create two new columns ‘Subjectivity’ & ‘Polarity’
    df['Subjectivity'] = df['body'].apply(getSubjectivity)
    df['Polarity'] = df['body'].apply(getPolarity)
    
    return df


# In[ ]:


res = vectorize_emails(df, max_features=1000)
res = sentiment_features2(res)
model = spam_predicter(res)
model.score()


# ## Basic Experiment 1
# #### Includes:
# * 1000 words
# * Lemmatization

# In[ ]:


res = vectorize_emails(df, max_features=1000, tokenizer=lemm_tokenizer)
model = spam_predicter(res)
model.score()


# ## Basic Experiment 2
# #### Includes:
# * 200 words
# * Document Features
# * Sentiment

# In[ ]:


res = vectorize_emails(df, max_features=200)
res = get_doc_features(res)
res = sentiment_features(res)
model = spam_predicter(res)
model.score()


# ## Basic Experiment 3
# #### Includes:
# * 1000 words
# * Document Features
# * lemmatization
# * sentiment

# In[ ]:


res = vectorize_emails(df, max_features=1000, tokenizer=lemm_tokenizer)
res = get_doc_features(res)
res = sentiment_features(res)
model = spam_predicter(res)
model.score()


# ## Advanced Experiment 1
# #### Includes:
# * 200 words
# * word embeddings
# * POS

# In[ ]:


res = vectorize_emails(df, max_features=200)
res = add_doc_vecs(res)
res = pos_features_norm(res)

model = spam_predicter(res)
model.score()


# ## Advanced Experiment 2
# #### Includes:
# * 200 words
# * XGBoost

# In[ ]:


res = vectorize_emails(df, max_features=200)

model = xg_spam_predicter(res)
model.score()


# ## Advanced Experiment 3
# #### Includes:
# * 1000 words
# * doc_features
# * lemmatization
# * sentiment
# * XGBoost

# In[ ]:


res = vectorize_emails(df, max_features=1000, tokenizer=lemm_tokenizer)
res = get_doc_features(res)
res = sentiment_features(res)

model = xg_spam_predicter(res)
model.score()


# ## Advanced Experiment 4
# #### Includes:
# * 200 words
# * doc features
# * lemmatization
# * sentiment
# * embeddings
# * POS
# * XGBoost

# In[ ]:


res = vectorize_emails(df, max_features=200, tokenizer=lemm_tokenizer)
res = get_doc_features(res)
res = sentiment_features(res)
res = add_doc_vecs(res)
res = pos_features_norm(res)

model = xg_spam_predicter(res)
model.score()


# ## Advanced Experiment 5
# #### Includes:
# * 1000 words
# * word embeddings
# * POS

# In[ ]:


res = vectorize_emails(df, max_features=1000, tokenizer=lemm_tokenizer)
res = get_doc_features(res)
res = sentiment_features(res)
res = add_doc_vecs(res)
res = pos_features_norm(res)

model = spam_predicter(res)
model.score()


# ## Advanced Experiment 6
# #### Includes:
# * 1000 words
# * doc features
# * lemmatization
# * sentiment
# * word embeddings
# * POS
# * XGBoost

# In[ ]:


res = vectorize_emails(df, max_features=1000, tokenizer=lemm_tokenizer)
res = get_doc_features(res)
res = sentiment_features(res)
res = add_doc_vecs(res)
res = pos_features_norm(res)

model = xg_spam_predicter(res)
model.score()


# ## Hyper param tuning

# In[ ]:


from hyperopt import hp, fmin, tpe
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


space = {
    'max_depth':        hp.quniform('max_depth', 3, 8, 1),
    'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'subsample':        hp.quniform('subsample', 0.5, 1, 0.05),
    'gamma':            hp.quniform('gamma', 0, 1, 0.05),
    'scale_pos_weight': hp.quniform('scale_pos_weight', 1, 5, 0.2),
    'n_estimators':     hp.quniform('n_estimators', 100, 1500, 100)
}


# In[ ]:


target='spam'
res = vectorize_emails(df, max_features=400)
X = res.select_dtypes(['number']).drop([target], axis=1)
y = res[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def f(params):
    
    model = XGBClassifier(learning_rate = 0.05,
        max_depth=int(params['max_depth']),
        min_child_weight=params['min_child_weight'],
        colsample_bytree=params['colsample_bytree'],
        subsample=params['subsample'],                 
        gamma=params['gamma'],
        scale_pos_weight=params['scale_pos_weight'],
        n_estimators=int(params['n_estimators']))
    model.fit(X_train, y_train)

    return 1-f1_score(y_test,model.predict(X_test))


# In[ ]:


#result = fmin(fn=f, space=space, algo=tpe.suggest, max_evals=100)


# In[ ]:


#result


# In[ ]:




