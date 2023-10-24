import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
data=pd.read_csv('spam.csv',encoding='ISO-8859-1')
X=data['v2']
y=data['v1']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
tfidf_vectorizer=TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
naive_bayes=MultinomialNB()
naive_bayes.fit(X_train_tfidf,y_train)
y_pred = naive_bayes.predict(X_test_tfidf)
acc=accuracy_score(y_test,y_pred)
print(f"Accuracy:{acc}")
print(f"Report:\n{classification_report(y_test,y_pred,target_names={'ham','spam'})}")