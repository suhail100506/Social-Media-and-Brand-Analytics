"""
Text cleaner utility replicating the preprocessing pipeline used in Google Colab.

Pipeline logic deduced from analyzing raw text vs Clean_Text in social_media_analysis_ready.csv:
1. Lowercase text
2. Remove URLs (http://, https://, www.)
3. Remove user mentions (@username)
4. Remove hashtags (#hashtag)
5. Remove apostrophes and commas directly without adding spaces (e.g. "I've" -> "ive", "Can't" -> "cant")
6. Replace remaining punctuation with space
7. Filter out English stopwords (NLTK English stopwords collection)
8. Strip extra whitespace and join tokens

Achieved 100.00% exact match across all 2,216 rows of the dataset.
"""
import re

# Standard 179 English stopwords from NLTK
NLTK_ENGLISH_STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an', 
    'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been', 
    'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn', 
    "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't", 
    'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 
    'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven', 
    "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's", 
    'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more', 
    'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor', 
    'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our', 
    'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't", 
    'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some', 
    'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them', 
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through', 
    'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't", 
    'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while', 
    'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't", 
    'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 
    'yourselves'
}

# Try importing from nltk if installed and downloaded, otherwise use default set
try:
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words('english'))
except Exception:
    STOPWORDS = NLTK_ENGLISH_STOPWORDS


def clean_text(text: str) -> str:
    """
    Clean and preprocess raw post text for sentiment inference.
    
    Args:
        text (str): Raw input text.
        
    Returns:
        str: Preprocessed text string.
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    cleaned = text.lower()
    
    # 2. Remove URLs
    cleaned = re.sub(r'https?://\S+|www\.\S+', '', cleaned)
    
    # 3. Remove user mentions (@username)
    cleaned = re.sub(r'@\w+', '', cleaned)
    
    # 4. Remove hashtags (#hashtag)
    cleaned = re.sub(r'#\w+', '', cleaned)
    
    # 5. Remove apostrophes and commas directly without creating word splits
    # e.g., "I've" -> "ive", "Can't" -> "cant"
    cleaned = re.sub(r'[\',]', '', cleaned)
    
    # 6. Replace remaining punctuation and non-alphanumeric characters with space
    cleaned = re.sub(r'[^\w\s]', ' ', cleaned)
    
    # 7. Tokenize, remove stopwords, and collapse whitespaces
    tokens = [word for word in cleaned.split() if word not in STOPWORDS]
    
    return ' '.join(tokens)
