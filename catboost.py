import catboost as cb
cat_1 = cb.CatBoostClassifier(iterations=1000, eval_metric='F1')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(vectors, targets, stratify = targets)

model_2= cat_1.fit(X_train, y_train, eval_set=(X_test,y_test), plot=True)

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

k = 5 # number of Kfolds

list_train_index = []
list_valid_index = []

skf = StratifiedKFold(n_splits=k, shuffle = True)
for train_index, valid_index in skf.split(X = vectors, y=targets):
    list_train_index.append(train_index)
    list_valid_index.append(valid_index)

list_train_index

cat = cb.CatBoostClassifier(iterations=300, eval_metric='F1')


for i in range(k):
    X_t = np.asarray(vectors)[list_train_index[i],:]
    X_v = np.asarray(vectors)[list_valid_index[i],:]
    
    y_t = np.asarray(targets)[list_train_index[i]]
    y_v = np.asarray(targets)[list_valid_index[i]]
    
    cat.fit(X = X_t, y = y_t, eval_set=(X_v, y_v));
    print(accuracy_score(y_t, cat.predict(X_t)))

pred_cat = cat.predict(np.asarray(vectors_test))

"""
# import csv
id_1 = twitter_df_test.id
with open('cat_twitter.csv', 'w', newline='\n') as f:
    writer = csv.writer(f)
    
    writer.writerow(['id', 'target'])
    for id_1, target in zip(id_1, pred_cat):
        writer.writerow([id_1, target])
"""

# catboost accuracy on test set was 66.043%