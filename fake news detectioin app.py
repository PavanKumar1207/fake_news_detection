import pickle
import string
import nltk
import re
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import streamlit as st

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# # for nltk.pos_tag function
# nltk.download('averaged_perceptron_tagger')
# nltk.download('omw-1.4')


def get_wordnet_pos(word):
    """Map POS tag to the format accepted by the lemmatizer"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)


def data_preprocessing(text):
  text = text.lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  words = word_tokenize(text)
  stop_words = stopwords.words('english')
  words = [word for word in words if word not in stop_words]
  lemmatizer = WordNetLemmatizer()
  words = [lemmatizer.lemmatize(word,pos=get_wordnet_pos(word)) for word in words]
  text = ' '.join(words)
  return text


def predict(title,text):
  filename = 'D:/ml projects/project-3 fake news detection/trained_model_for_fake_news_detection.sav'
  logistic_model = pickle.load(open(filename, 'rb'))
  filename1 = 'D:/ml projects/project-3 fake news detection/tfidfmodel.sav'
  filename2 = 'D:/ml projects/project-3 fake news detection/countvectmodel.sav'
  tfidf = pickle.load(open(filename1,'rb'))
  countvectorizer = pickle.load(open(filename2,'rb'))
  text = title + ' ' + text
  text = data_preprocessing(text)
  text = tfidf.transform(countvectorizer.transform([text]))
  result = logistic_model.predict(text)[0]
  if result == 1:
    return 'Fake News'
  else:
    return 'True News'
# title = "President Trump Just Gave A HUGE Gift to #HurricaneHarvey Reliefâ€¦Walking the Walk!"
# text =  'Yea Baby! President Trump walks the walk!President Trump just put his money where his mouth is! He donated $1 million to the Harvey Hurricane relief! What a great thing to do! He has a huge heart!Is this empathetic enough for the left? The media decided to hammer Trump after his incredible visit to storm ravaged Texas. They claimed he wasn t empathetic enough to the victims which is total bs. Real Americans aren t buying this bs line from the media because we all watched the First Couple in action. They care and we know it that s all that matters!Fox News reported:The White House announced Thursday that President Donald Trump pledged to donate $1 million in personal funds to Harvey relief efforts.Trump visited Corpus Christi, Texas, and Austin on Tuesday for briefings on Harvey s devastation. He praised first responders, telling everyone who has been affected by the storm that  we are here with you today, we are with you tomorrow and we will be with you every single day after to restore, recover and rebuild. First responders have been doing heroic work. Their courage & devotion has saved countless lives   they represent the very best of America! pic.twitter.com/I0gvCQLTKO  Donald J. Trump (@realDonaldTrump) August 31, 2017On Wednesday, Trump tweeted that on his Tuesday trip to Texas he had witnessed  first hand the horror & devastation  wrought by Harvey. He wrote that after seeing the widespread damage,  my heart goes out even more so to the great people of Texas! He plans on returning to Texas on Saturday. VP Pence and the Second Lady Karen Pence visited the storm victims today and pitched in to clear debris. '
# result = predict(title,text)
# print(result)

def main():

  # giving title for app
  st.title('Fake News Detection Web app')

  # taking inputs from user

  title = st.text_input('Enter the title of the News')
  content = st.text_input('Enter the content of the News')

  #prediction

  if(st.button('Predict fake or Real News')):
      result = predict(title,content)
      st.success(result)

if __name__ == '__main__':
  main()
    
