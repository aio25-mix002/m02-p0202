import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords,wordnet
from sklearn.preprocessing import LabelEncoder

def lowercase(text):
    return text.lower()

def punctuation_removal(text):
    #translator = {k: '' for k in list(string.punctuation)}
    translator = str.maketrans('','',string.punctuation)
    return text.translate(translator)

def tokenize(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = stopwords.words('english')
    return [token for token in tokens if token not in stop_words]

def stemming(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_text(text):
    text = lowercase(text)
    text = punctuation_removal(text)
    tokens = tokenize(text)
    tokens = remove_stopwords(tokens)
    stem = stemming(tokens)
    return stem

#messages = [preprocess_text(message) for message in messages]

def create_dictionary(messages):
    dictionary = []
    for tokens in messages:
        if tokens not in dictionary:
            dictionary.append(tokens)
    return dictionary
dictionary = create_dictionary(messages)
#dictionary

def create_features(tokens, dictionary):
    features = np.zeros(len(dictionary))
    for token in tokens:
        if token in dictionary:
            features[dictionary.index(token)] += 1
    return features
# X = np.array([create_features(tokens,dictionary) for tokens in messages])


# Label encoder
# le = LabelEncoder()
# y = le.fit_transform(labels)


