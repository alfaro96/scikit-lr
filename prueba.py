from sklearn.datasets import fetch_openml
X, Y = fetch_openml(data_id=42851, return_X_y=True, as_frame=False)
Y = Y.astype(float)

from sklr.ensemble._bagging import BaggingLabelRanker
from sklr.ensemble._forest import RandomForestLabelRanker
from sklr.ensemble._weight_boosting import AdaBoostLabelRanker
print(AdaBoostLabelRanker(random_state=0).fit(X, Y).score(X, Y))
