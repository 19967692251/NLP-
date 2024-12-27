import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC  # 可以使用其它机器学习模型
from sklearn.linear_model import RidgeClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

train_df = pd.read_csv('./NLP新闻分类/train_set.csv', sep='\t', nrows=200000)
test_df = pd.read_csv('./NLP新闻分类/test_a.csv', sep='\t')

tfidf = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1,1),
    max_features=1000)

tfidf.fit(pd.concat([train_df['text'], test_df['text']]))
train_word_features = tfidf.transform(train_df['text'])
test_word_features = tfidf.transform(test_df['text'])

X_train = train_word_features
y_train = train_df['label']
X_test = test_word_features

KF = KFold(n_splits=5,shuffle=True, random_state=7)
clf = LinearSVC()
#clf = RidgeClassifier()
#clf = LGBMClassifier()
# 存储测试集预测结果 行数：len(X_test) ,列数：1列
test_pred = np.zeros((X_test.shape[0], 1), int)
f1 = 0
for KF_index, (train_index,valid_index) in enumerate(KF.split(X_train)):
    print('第', KF_index+1, '折交叉验证开始...')
    # 训练集划分
    x_train_, x_valid_ = X_train[train_index], X_train[valid_index]
    y_train_, y_valid_ = y_train[train_index], y_train[valid_index]
    # 模型构建
    clf.fit(x_train_, y_train_)
    # 模型预测
    val_pred = clf.predict(x_valid_)
    _f1 = f1_score(y_valid_, val_pred, average='macro')
    print("LinearSVC准确率为：", _f1)
    f1 += _f1
    # 保存测试集预测结果
    test_pred = np.column_stack((test_pred, clf.predict(X_test)))  # 将矩阵按列合并
# 多数投票
preds = []
for i, test_list in enumerate(test_pred):
    preds.append(np.argmax(np.bincount(test_list)))
preds = np.array(preds)
f1 = f1 / 5
print(f1)
#submission = pd.read_csv('./NLP新闻分类/test_a_sample_submit.csv')
#submission['label'] = preds
#submission.to_csv('./LinearSVC_submission.csv', index=False)