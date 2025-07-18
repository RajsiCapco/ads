import pandas as pd
import spacy
import re

nlp = spacy.load("en_core_web_sm")

df=pd.read_csv('emails_NLP.csv')

print(df.head())
print(df.info())

texts = df['message'].copy()

#remove lowercase text, email headers, urls, non-word characters,  punctuation, numbers, 
# tokenise words remaining
def clean_and_tokenize_spacy(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|<.*?>|[\r\n\t]',' ',text)
    text = re.sub(r'[^a-z\s]', '', text)

    doc=nlp(text)

    tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

    return tokens


sample_tokens = texts.iloc[0:5].apply(clean_and_tokenize_spacy)
print(sample_tokens)

from sklearn.feature_extraction.text import TfidfVectorizer

# Apply def-tokenizing to subset (increase speed of inital testing)
cleaned_texts = texts.iloc[0:1000].apply(clean_and_tokenize_spacy)

# Join tokens into strings
joined_texts = cleaned_texts.apply(lambda tokens: ' '.join(tokens))

vectorizer = TfidfVectorizer()

X_tfidf = vectorizer.fit_transform(joined_texts)

print("TF-IDF matrix shape:", X_tfidf.shape)

feature_names = vectorizer.get_feature_names_out()
print("Sample features:", feature_names[:20])

#most important words TF-IDF - term frequency-inverse document frequency
import numpy as np

def print_top_tfidf_words(row_index, tfidf_matrix, feature_names, top_n=10):
    row = tfidf_matrix[row_index].toarray().flatten()
    top_indices = row.argsort()[-top_n:][::-1]

    print(f"\nTop {top_n} words in document {row_index}:")
    for i in top_indices:
        print(f"{feature_names[i]}: {row[i]:.4f}")

print_top_tfidf_words(0,X_tfidf,feature_names, top_n=10)


#document similarity PCA - principal component analysis
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], alpha=0.5)
plt.title("Document Similarity (PCA on TF-IDF features)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.show()

#text classification model
import numpy as np

y = np.array([0]*500 + [1]*500)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train, y_test = train_test_split(X_tfidf, y, test_size = 0.2, random_state = 13)

# basic logistic regression classifier
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

#model eval
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("classification report:", classification_report(y_test,y_pred))
print("Confusion matrix:", confusion_matrix(y_test, y_pred))
# results indicate 78.5% accurate predictions of test data,84 true negs, 73 tru pos, 11 fals pos, 32 fals negs 

#Topic Modelling - reduce dimensions of text - basic topic extraction
from sklearn.decomposition import NMF

n_topics = 5

nmf_model = NMF(n_components=n_topics, random_state= 13)
W = nmf_model.fit_transform(X_tfidf)
H = nmf_model.components_
#w - document topic matrix (topic per doc), H - topic word matrix (word per topic)

#print keywords per topic
def print_topics(H, feature_names, top_n=10):
    for topic_idx, topic in enumerate(H):
        print(f"\nTopic {topic_idx + 1}:")
        top_indices = topic.argsort()[-top_n:][::-1]
        for i in top_indices:
            print(f"{feature_names[i]}", end=" ")

print_topics(H, feature_names, top_n=10)

# results show likely themes of 1: email metadata, 2: work coordination / calender updates , 3: real estate/finance, 4: rental mgmt 5, personal or time/date comms

# use topic model to improve classifer

W = nmf_model.fit_transform(X_tfidf)
X_Augmented = np.hstack((X_tfidf.toarray(), W))

pseudo_labels = W.argmax(axis=1) 
#label doc by main topic

# use topic scores for features in new model 
X_combined = np.hstack((X_tfidf.toarray(), W))

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_combined, y, test_size= 0.2, random_state=13)

modelC = LogisticRegression()
modelC.fit(X_train_c, y_train_c)

y_pred_c = modelC.predict(X_test_c)

print("Accuracy:", accuracy_score(y_test_c, y_pred_c))
print("Classification Report:", classification_report(y_test_c, y_pred_c))
#no visible difference once adding in the labeling and additional features

pca_topics = PCA(n_components=2)
W_2d = pca_topics.fit_transform(W)

plt.figure(figsize= (10,6))
plt.scatter(W_2d[:, 0], W_2d[:, 1], c=y, cmap='coolwarm', alpha=0.6)
plt.title("Document Clusters by NMF Topics")
plt.xlabel("Topic PCA 1")
plt.ylabel("Topic PCA 2")
plt.grid(True)
plt.show()


# try LDA - alternative to NMF 
#lda models how docs are gerenated based on topic dis, nmpf is decomp -finding parts in data

from sklearn.decomposition import LatentDirichletAllocation
lda_model = LatentDirichletAllocation(n_components=5, random_state=13)
lda_W = lda_model.fit_transform(X_tfidf)
lda_H = lda_model.components_

# Use the same print_topics function
print(" LDA Topics:")
print_topics(lda_H, feature_names, top_n=10)