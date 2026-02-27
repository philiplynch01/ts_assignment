from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from mrsqm import MrSQMClassifier
from mrsqm.mrsqm_wrapper import MrSQMTransformer  # import transformer directly
from experiments.utils import get_cmj_data

x_train, y_train, x_test, y_test = get_cmj_data()

# Mirror MrSQMClassifier.fit() but with fixed LogisticRegression
transformer = MrSQMTransformer(strat='RS', features_per_rep=500, selection_per_rep=2000, nsax=0, nsfa=5)
train_x = transformer.fit_transform(x_train, y_train)

clf = LogisticRegression(
    solver='newton-cg',
    class_weight='balanced',
    random_state=0,
    max_iter=1000
).fit(train_x, y_train)

# Transform test data and predict
test_x = transformer.transform(x_test)
predictions = clf.predict(test_x)

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(accuracy_score(y_test, predictions))