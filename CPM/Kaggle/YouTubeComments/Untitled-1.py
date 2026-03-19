# %%
# !unzip -q -o ./data/you-tube-comments-signal-vsosai.zip -d ./data/

# %%
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# %% [markdown]
# ## Dependencies

# %%
import pandas as pd
import numpy as np

from catboost import CatBoostClassifier

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
import seaborn

import emoji

from tqdm.cli import tqdm
from tqdm import trange

tqdm.pandas()

import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('averaged_perceptron_tagger_eng', quiet=True)
nltk.download('names', quiet=True)


# %%
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_train.head()

# %% [markdown]
# ## EDA

# %%
df_train.shape

# %%
df_train['target'].value_counts()

# %%
df_train['comment_text'][:10]

# %%
df_train.isna().sum()

# %%
df_train = df_train.fillna("NA")
df_test = df_test.fillna("NA")

# %% [markdown]
# ## Solutions

# %%
def text_feature_extraction(df: pd.DataFrame, col: str):
    df['emoji_count'] = df[col].progress_apply(lambda x: len(emoji.emoji_list(x)))
    df['happy'] = df[col].progress_apply(lambda x: x.count(':)') + x.count(';)') + x.count('(:') + x.count('(;'))
    df['sad'] = df[col].progress_apply(lambda x: x.count(':(') + x.count(';(') + x.count('):') + x.count(');'))
    df['is_na'] = df[col] == "NA"
    return df

# %%
def text_normalize(text: str):
    text = text.lower()
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')
    text = text.replace('\t', ' ')
    text = "".join([ch for ch in text if ch not in string.punctuation])
    text = emoji.replace_emoji(text, replace='')

    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    text = ' '.join(words)
    return text
    
text_normalize(df_train['comment_text'][4])

# %%
df_extra = pd.read_csv('./data/extra_unlabeled.csv').dropna()

# %%
df_train_clean = df_train.copy()
df_test_clean = df_test.copy()
df_extra_clean = df_extra.copy()

# df_train_clean = text_feature_extraction(df_train_clean, 'comment_text')
df_train_clean['comment_text'] = df_train_clean['comment_text'].progress_apply(text_normalize)

# df_test_clean = text_feature_extraction(df_test_clean, 'comment_text')
df_test_clean['comment_text'] = df_test_clean['comment_text'].progress_apply(text_normalize)

# df_extra_clean = text_feature_extraction(df_extra_clean, 'comment_text')
df_extra_clean['comment_text'] = df_extra_clean['comment_text'].progress_apply(text_normalize)

df_train_clean.head()

# %%
df_train_clean = text_feature_extraction(df_train_clean, 'comment_text')
df_test_clean = text_feature_extraction(df_test_clean, 'comment_text')

# %% [markdown]
# ### CatBoost

# %%
X = df_train_clean.drop(columns=['id', 'target'])
y = df_train_clean['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# %%
model = CatBoostClassifier(
    iterations=50000,
    learning_rate=0.01,
    eval_metric='MultiClass',
    random_seed=42,
    early_stopping_rounds=1000,
    use_best_model=True,
    thread_count=-1,
    task_type='GPU',
    auto_class_weights='Balanced',
    text_features=['comment_text']
)

model.fit(
   X_train,
   y_train,
   eval_set=(X_val, y_val),
   verbose=1000
)

# %%
subm = df_test_clean.copy()
subm['target'] = model.predict(df_test_clean.drop(columns=['id']))
subm[['id', 'target']].to_csv("subm.csv", index=False)
subm.head()

# %% [markdown]
# ### Linear Models

# %%
df_train.head()

# %%
all_text = df_train_clean['comment_text'].tolist() + df_extra_clean['comment_text'].tolist() + df_test_clean['comment_text'].tolist()
all_text[:10]

# %%
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 3), stop_words='english')
tfidf.fit(all_text)

# %%
X_tfidf = tfidf.transform(df_train_clean['comment_text'])
X_tfidf_test = tfidf.transform(df_test_clean['comment_text'])
y = df_train_clean['target']

X_tfidf_train, X_tfidf_val, y_train, y_val = train_test_split(X_tfidf, y, test_size=0.3, random_state=42, stratify=y)

# svd = TruncatedSVD(n_components=2000, random_state=42)
# X_tfidf_train = svd.fit_transform(X_tfidf_train)
# X_tfidf_val = svd.transform(X_tfidf_val)
# X_tfidf_test = svd.transform(X_tfidf_test)

# sc = StandardScaler()
# X_tfidf_train = sc.fit_transform(X_tfidf_train)
# X_tfidf_val = sc.transform(X_tfidf_val)
# X_tfidf_test = sc.transform(X_tfidf_test)

print(f"TF-IDF матрица: {X_tfidf.shape}")
print(f"Ненулевых элементов: {X_tfidf.nonzero()[0].shape[0] / np.prod(X_tfidf.shape) * 100:.2f}%")

# %%
df_test.shape, df_test_clean.shape, X_tfidf_test.shape

# %%
model = LogisticRegression(
    max_iter=10000,
    n_jobs=-1,
    class_weight='balanced',
    random_state=42
)
# model.fit(X_tfidf_train, y_train)

# y_pred = model.predict(X_tfidf_val)
# f1_score(y_val, y_pred, average='macro')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X_tfidf, y, cv=cv, scoring='f1_macro', n_jobs=-1)
scores.mean()

# %%
# subm = df_test.copy()
# subm['target'] = model.predict(X_tfidf_test)
# subm.head()

# subm[['id', 'target']].to_csv("subm.csv", index=False)

# %%
model = LogisticRegression(
    max_iter=1000000,
    class_weight='balanced',
    n_jobs=-1,
    random_state=42
)
model.fit(X_tfidf_train, y_train)

y_pred = model.predict(X_tfidf_val)
f1_score(y_val, y_pred, average='macro')

# %%
subm = df_test.copy()
subm['target'] = model.predict(X_tfidf_test)
subm.head()

subm[['id', 'target']].to_csv("subm.csv", index=False)

# %%



