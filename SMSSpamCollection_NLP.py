import nltk
import string
from nltk.corpus import stopwords
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
message=[]
text=open('SMSSpamCollection','r')
for line in text:
     msg=line.rstrip()
     message.append(msg)



# ** we can remove all line above from 4 to 8
# and use that line
#message=[line.rstrip() for line in open('SMSSpamCollection','r')]

#***to add counter for the first 10 messages
#for count, message in enumerate(message[:10]):
    #print(count, message)
    #print('\n')


messages=pd.read_csv('SMSSpamCollection.csv',sep='\t',names=['label','msg'])
#print(messages.head(2))

#** get length for each msg
messages['length']=messages['msg'].apply(len)
#** to show all msg
#print(messages[messages['length']==910]['msg'].iloc[0])

def text(msg):
#1---- non punctuation
  mess="hello &ho"
  nopunc=[]
  for l in msg:
      if l not in string.punctuation:
         nopunc.append(l)
  nopunc=''.join(nopunc)

# ** one line
#nopunc=[c for c in mess: if c not in string.punctuation]

# split
  split_nopunc=nopunc.split()
#2----the word that not popular
  clean=[word for word in split_nopunc if word.lower() not in stopwords.words('english')]
#turn word into lower letter as stopword
  return (clean)

process=messages['msg'].head(4).apply(text)
#print(process)
"****************************************************"
#****** Vectors
#1--- count how many word occur in each msg(frequency)
"""
cv = CountVectorizer(vocabulary=['hot', 'cold', 'old'])
cv.fit_transform(['pease porridge hot', 'pease porridge cold', 'pease porridge in the pot', 'nine days old']).toarray()
array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 0],
       [0, 0, 1]],
"""
transformer=CountVectorizer(analyzer=text)
transformer.fit(messages['msg']) #train

msg4=messages['msg'][3]
freq_matrix_msg4=transformer.transform([msg4]) #msg4 test_set
#print(freq_matrix_msg4)
#bag of word
#get feature name
#print(transformer.get_feature_names()[4068])

"*********************************************"
#2--- inverse doc frequency TF-IDF
# M(tf-idf)=M(tf)*M(idf)

data=transformer.transform(messages['msg']) #matrix
TFidf=TfidfTransformer().fit(data) # to calculate the tf-idf weights
                                   #fit()calculated the idf for the matrix,

TF4=TFidf.transform(freq_matrix_msg4)
print(TF4)
"*********************************************"
#3---Training a model, detecting spam
spam_detect=MultinomialNB()
spam_detect.fit(TFidf.transform(data),messages['label'])
print(spam_detect.predict(TF4))