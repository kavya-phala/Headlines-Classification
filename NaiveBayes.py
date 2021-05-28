import string
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import train_test_split

#reads the csv file
headlines = pd.read_csv("SML_Headlines.csv")

#prints the csv file as category,headline
print(headlines.head())
#print(headlines.shape)#tells us how many rows,columns the data set has
#print(headlines['Category'].value_counts())#counts values in each category

#cleaning the text
def text_cleaning(a):
    remove_punctuations = [char for char in a if char not in string.punctuation]
    remove_punctuations = ''.join(remove_punctuations)
    stop_list = set(stopwords.words('english'))#stop_list contains the set of stop words
    return [word for word in remove_punctuations.split() if word.lower() not in stop_list]#headline without stop words

#print(headlines.iloc[:,1].apply(text_cleaning))#to check text cleaning on column 1 which has the headlines

#count vectorizer
BoW_transform = CountVectorizer(analyzer = text_cleaning).fit(headlines['Headline'])
#print(BoW_transform.vocabulary_)#tells us what value is associated with which word

#tranforming this data
Headlines_BoW = BoW_transform.transform(headlines['Headline'])
#print(Headlines_BoW)#prints the transformed Bag of Words

#finding most significant words using tf-idf transformer
tfidf_transform = TfidfTransformer().fit(Headlines_BoW)
Headlines_tfidf = tfidf_transform.transform(Headlines_BoW)
#print(Headlines_tfidf)#tells us tfidf values of the whole vocab in text

Category = headlines['Category']
#splitting dataset into train and test
Headlines_train, Headlines_test, Category_train, Category_test = train_test_split(Headlines_tfidf, Category, test_size = 0.2, random_state=13)
#Building the model
model = MultinomialNB().fit(Headlines_train, Category_train)
predictions = model.predict(Headlines_test)
#print(predictions)#tells us the predicted categories

#confusion matrix
print(confusion_matrix(Category_test,predictions))

#Accuracy score of classifier
print("Accuracy: ", accuracy_score(Category_test, predictions))

#classification report
print(metrics.classification_report(Category_test, predictions))