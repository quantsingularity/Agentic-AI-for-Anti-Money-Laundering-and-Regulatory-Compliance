from aml.models.xgboost_classifier import XGBoostClassifier


def test_xgboost_n_estimators_param():
    clf = XGBoostClassifier(task="binary", seed=42, n_estimators=5)
    assert hasattr(clf, "n_estimators")
    assert clf.n_estimators == 5

    # quick smoke train on small synthetic data
    import numpy as np

    X = np.random.rand(50, 5)
    y = np.random.randint(0, 2, size=50)

    clf.train(X, y)
    preds = clf.predict(X)
    assert len(preds) == 50
