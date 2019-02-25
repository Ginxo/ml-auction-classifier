from string import punctuation

from nltk.corpus import stopwords

FEATURE_VECTOR = 'feature-vector'
LABEL = 'label'
TECH_LABEL = 'Tech'
NON_TECH_LABEL = 'Non-Tech'
ENGLISH_STOP_WORDS = set(stopwords.words('english') + list(punctuation) + ["'s", '"', "'", "", '—', '’', "”", "“", '``', '‘'])
