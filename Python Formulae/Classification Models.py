# Import packages
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, normalize, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier

def classificationModels(data, dep_var):
    """
    Runs a dataset and chosen dependant variable through list of agnostic machine classification models. Intention
    is to produce a summary of results based on plain-vanilla models. Models do not include GridSearchCV

    list of models include Multinomial Naive Bayes, Stochastic Gradient Descent, Random Forest, Adaboost,
    Gradient Boosting, and Extreme Gradient Boost

    :param data: Dataset
    :param dep_var: Dependent variable
    :return: Returns average accuracy, a confusion matrix, and classification report
    """

    # Data entry and processing
    Y = data[dep_var]  # Pull only column of dependent variable
    X = MinMaxScaler().fit_transform(data.drop(lavels=dep_var, axis=1))  # Normalize dataset with dep_var omitted
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=.2, stratify=Y)
    feat_count = len(X.columns)  # Number of features in input dataset

    # Multinomial Naive Bayes
    nb = MultinomialNB().fit(X_train, Y_train)  # Build model
    pred_nb = nb.predict(X_validation)
    print("Multinomial Naive Bayes")
    print(np.mean(pred_nb == Y_validation))  # Average prediction accuracy
    print(confusion_matrix(Y_validation, pred_nb))  # Confusion matrix
    print(classification_report(Y_validation, pred_nb))  # Classification report

    # Stochastic Gradient Descent
    sgd = SGDClassifier(loss='hinge', alpha=1e-3, random_state=0).fit(X_train, Y_train)  # Build model
    pred_sgd = sgd.predict(X_validation)
    print("Stochastic Gradient Descent")
    print(np.mean(pred_sgd == Y_validation))  # Average prediction accuracy
    print(confusion_matrix(Y_validation, pred_sgd))  # Confusion matrix
    print(classification_report(Y_validation, pred_sgd))  # Classification report

    # Random Forest
    rf = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, Y_train)  # Build model
    pred_rf = rf.predict(X_validation)
    print("Random Forest")
    print(np.mean(pred_rf == Y_validation))  # Average prediction accuracy
    print(confusion_matrix(Y_validation, pred_rf))  # Confusion matrix
    print(classification_report(Y_validation, pred_rf))  # Classification report

    # Adaboost Classifier
    ab = AdaBoostClassifier(n_estimators=40, random_state=0).fit(X_train, Y_train)  # Build model
    pred_ab = ab.predict(X_validation)
    print("Adaboost Classifier")
    print(np.mean(pred_ab == Y_validation))  # Average prediction accuracy
    print(confusion_matrix(Y_validation, pred_ab))  # Confusion matrix
    print(classification_report(Y_validation, pred_ab))  # Classification report

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=50, random_state=0).fit(X_train, Y_train)  # Build model
    pred_gb = gb.predict(X_validation)
    print("Gradient Boosting Classifier")
    print(np.mean(pred_gb == Y_validation))  # Average prediction accuracy
    print(confusion_matrix(Y_validation, pred_gb))  # Confusion matrix
    print(classification_report(Y_validation, pred_gb))  # Classification report

    # Extreme Gradient Boosting
    xgb = XGBClassifier().fit(X_train, Y_train)  # Build model
    pred_xgb = xgb.predict(X_validation)
    print("Extreme Gradient Boosting")
    print(np.mean(pred_xgb == Y_validation))  # Average prediction accuracy
    print(confusion_matrix(Y_validation, pred_xgb))  # Confusion matrix
    print(classification_report(Y_validation, pred_xgb))  # Classification report
