from string import punctuation

from nltk.corpus import stopwords

FEATURE_VECTOR = 'feature-vector'
LABEL = 'label'
TECH_LABEL = 'Tech'
NON_TECH_LABEL = 'Non-Tech'
ENGLISH_STOP_WORDS = set(stopwords.words('english') + list(punctuation) + ["'s", '"', "'", "", '—', '’', "”", "“", '``', '‘'])
LABELS_COLOR_MAP = {
    0: '#20b2aa', 1: '#ff7373', 2: '#ffe4e1', 3: '#005073', 4: '#4d0404',
    5: '#ccc0ba', 6: '#4700f9', 7: '#f6f900', 8: '#00f91d', 9: '#da8c49'
}
