import nltk
import pandas as pd
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
print(messages[messages['length']==910]['msg'].iloc[0])
