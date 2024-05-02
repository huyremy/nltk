import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist
from nltk.tokenize import MWETokenizer

def extract_keywords(text):
    nltk.download('stopwords')
    
    stop_words = set(stopwords.words('english'))
    
    words = word_tokenize(text)
    
    # Xác định các từ ghép bằng cách sử dụng BigramCollocationFinder
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 10)  # Chỉ xác định 10 từ ghép
    
    # Tạo tokenizer với danh sách từ ghép
    tokenizer = MWETokenizer(bigrams)
    
    words = tokenizer.tokenize(words)
    
    keywords = [word.lower().replace("_", " ") for word in words if word.lower() not in stop_words]
    
    fdist = FreqDist(keywords)
    
    return fdist.most_common()

text = """
    Hello! I’m Soumil Nitin Shah, a Software and Hardware Developer based 
    in New York City. I have completed by Bachelor in Electronic Engineering and
    my Double master’s in Computer and Electrical Engineering. I Develop Python Based Cross 
    Platform Desktop Application , Webpages , Software, REST API, Database and much more I 
    have more than 2 Years of Experience in Python
    """

keywords = extract_keywords(text)

# In ra màn hình những từ khóa có tần suất xuất hiện cao nhất
for keyword, count in keywords:
    keyword = keyword.replace(",", "").replace(".", "")
    print(keyword)
