from sklearn.datasets import fetch_openml
X, Y = fetch_openml(data_id=42851, return_X_y=True, as_frame=False)
Y = Y.astype(float)

print(Y)

from sklr.ensemble._bagging import BaggingLabelRanker
from sklr.ensemble._forest import RandomForestLabelRanker
from sklr.ensemble._weight_boosting import AdaBoostLabelRanker
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.7, random_state=0)
print(BaggingLabelRanker(random_state=0).fit(X_train, y_train).score(X_test, y_test))
