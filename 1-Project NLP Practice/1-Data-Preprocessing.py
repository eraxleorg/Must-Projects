# This exercise is about text preprocessing techniques
# - data cleaning
# -lowering the text .case

# - data tokenization
# - stop words removal
# - lemmatization
# - stemming


import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

stopwords.words("english")
nltk.download("stopwords")


nltk.download("stopwords")


import nltk

nltk.download("stopwords")


text = "Excited to, to express interest in the Procurement Officer position at Emirates Logistics LLC. With solid experience in strategic sourcing, supplier management, and cost control, there is confidence in adding value to your procurement team."

text = text.lower()
text

text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
text

text = re.sub(r"\d+", "", text)
text


import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download()

nltk.data.path.append("stopwords")
stop_words = set(stopwords.words("english"))

text = text.split()
filter_words = [word for word in text if word not in stop_words]
filter_words


tokens = " ".join(filter_words).split()
tokens


Practice  # 1:


import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords

text = "May i know more about your cv writing services and the charges for the same and how long does it take to get the cv done. also please make sure cv is ats compliant."


# Step 1: Lower the text
text = text.lower()

# Step 2: Remove punctuation
text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

# Step 3: Remove digits
text = re.sub(r"\d+", "", text)

# Step 4: Tokenization
text = text.split()

# Step 5: Remove stop words
stop_words = set(stopwords.words("english"))
filter_words = [word for word in text if word not in stop_words]

# Step 6: Join the words
tokens = " ".join(filter_words).split()


def preprocess_text(text, remove_digits=True, remove_stopwords=True, tokenize=True):

    # Lower the text
    text = text.lower()

    # Remove punctuation
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)

    # Remove digits
    if remove_digits:
        text = re.sub(r"\d+", "", text)

    # Tokenization
    if tokenize:
        text = text.split()

    # Remove stop words
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        text = [word for word in text if word not in stop_words]

    # Join the words
    if tokenize:
        text = " ".join(text).split()

    return text


text = "May i 6 know. more about your cv writing 777  services and the charges for the same and how long does it take to get the cv done. also please make sure cv is ats compliant."
after_preprocessing = preprocess_text(
    sample_text, remove_digits=True, remove_stopwords=True, tokenize=True
)
print(after_preprocessing)



