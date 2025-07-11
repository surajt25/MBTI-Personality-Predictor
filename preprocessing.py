import pandas as pd
import nltk
import re
import string

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Loading dataset
df = pd.read_csv('mbti_1.csv')

# Optional: List of MBTI types to remove from posts
mbti_types = [
    'INFJ','INFP','INTJ','INTP','ISFJ','ISFP','ISTJ','ISTP',
    'ENFJ','ENFP','ENTJ','ENTP','ESFJ','ESFP','ESTJ','ESTP'
]

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # Lowercase text
    text = text.lower()

    # Removing MBTI types
    for mbti in mbti_types:
        text = re.sub(mbti.lower(), '', text)
    # Removing URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)

    # Removing punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] #this is list comprehension 
    return ' '.join(tokens)


df['clean_posts'] = df['posts'].head().apply(clean_text)

# View cleaned data
print(df[['type', 'clean_posts']].head())
# Create 4 binary labels
df['IE'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)  # 1 = I, 0 = E
df['NS'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)  # 1 = N, 0 = S
df['TF'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)  # 1 = T, 0 = F
df['JP'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)  # 1 = J, 0 = P
print(df[['type', 'IE', 'NS', 'TF', 'JP']].head())
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # Emoticons
                           u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # Flags
                           u"\U00002700-\U000027BF"  # Dingbats
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)

# Check how many posts contain emojis
emoji_counts = df['posts'].apply(lambda x: bool(emoji_pattern.search(x)))
print("Posts with emojis:", emoji_counts.sum())
# Optionally view few examples
df[emoji_counts].head(3)['posts']
df.to_csv('mbti_preprocessed.csv', index=False)
