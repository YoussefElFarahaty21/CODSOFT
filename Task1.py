import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords

train = pd.read_csv("train_data.txt", delimiter=':::', header=None, names=["id", "title", "genre", "plot_summary"], engine='python')# here i used engine = python as there is waning about using custom delimiter
X_train=train["plot_summary"]
Y_train=train["genre"]

test_data=pd.read_csv("test_data.txt",delimiter=':::',header=None,names=["id","title","plot_summary"],engine='python')

test_data_solution=pd.read_csv("test_data_solution.txt",delimiter=':::', header=None, names=['id','title','genre','plot_summary'],engine='python')

tfidf_vectorizer=TfidfVectorizer(max_features=5000, stop_words=stopwords.words('english'))
X_train_tfidf=tfidf_vectorizer.fit_transform(X_train)

clf=MultinomialNB()
clf.fit(X_train_tfidf,Y_train)

X_test_tfidf=tfidf_vectorizer.transform(test_data["plot_summary"])
y_pred=clf.predict(X_test_tfidf)
test_labels=test_data_solution["genre"]

accuracy=accuracy_score(test_labels,y_pred)
print(f"Accuracy:{accuracy}")