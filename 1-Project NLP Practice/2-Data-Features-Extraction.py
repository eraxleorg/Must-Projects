from sklearn.feature_extraction.text import TfidfVectorizer


corpus = ["May i know more ", "I am looking for a job"]
unique_words = set(word.lower() for sentence in corpus for word in sentence.split())
print(unique_words)

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
y = X.toarray()
y
y.shape




from gensim.models import Word2Vec

# Example corpus
sentences = [
    ['may', 'i', 'know', 'more'],
    ['i', 'am', 'looking', 'for', 'a', 'job']
]

# Train a Word2Vec model
model = Word2Vec(sentences, vector_size=50, window=2, min_count=1, sg=1)

# Get the word vector for 'job'
word_vector = model.wv['more']
print(word_vector)
word_vector.shape

# Find most similar words to 'job'
similar_words = model.wv.most_similar('more')
print(similar_words)
