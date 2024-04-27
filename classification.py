import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer, LabelEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# Load diagnosis data
diagnosis_data = pd.read_csv(r'C:\Diplomatiki\Datasets\Worf\actual.csv')

# Load gene expression data
expression_data = pd.read_csv(r"C:\Diplomatiki\Datasets\Worf\actual_train.csv")

# Merge the diagnosis and expression data on patient ID
data = pd.merge(diagnosis_data, expression_data, left_on='patient', right_on='patient')

# Separate features (gene expression) and labels (diagnosis)
# X = data.drop(['patient', 'diagnosis'], axis=1)
features = [col for col in data.columns if col not in ['patient', 'diagnosis']]
cols = ['patient', 'diagnosis']
classifiers = {
    # SVC(): {
    #     'estimator__C': [0.1, 1, 10, 100, 1000, 10000, 100000],
    #     'estimator__kernel': ['linear', 'rbf', 'poly'],
    #     'estimator__gamma': ['scale', 'auto']
    # },
    GradientBoostingClassifier(): {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__learning_rate': [0.05, 0.1, 0.2],
        'estimator__max_depth': [3, 4, 5],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4]
    },
    RandomForestClassifier(): {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__max_features': ['auto', 'sqrt'],
        'estimator__bootstrap': [True, False],
        'estimator__class_weight': [None, 'balanced'],
        'estimator__random_state': [42]  # You can adjust the random state as needed
    },
    MLPClassifier(max_iter=1000): {
        'estimator__hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
        'estimator__activation': ['relu', 'logistic', 'tanh'],
        'estimator__solver': ['adam', 'sgd', 'lbfgs'],
        'estimator__alpha': [0.0001, 0.001, 0.01],
        'estimator__learning_rate': ['constant', 'adaptive', 'invscaling'],
        'estimator__learning_rate_init': [0.001, 0.01, 0.1],
        'estimator__max_iter': [100, 200, 300],
        'estimator__tol': [1e-4, 1e-3, 1e-2],
        'estimator__early_stopping': [True, False],
        'estimator__validation_fraction': [0.1, 0.2, 0.3],
        'estimator__beta_1': [0.9, 0.95, 0.99],
        'estimator__beta_2': [0.999, 0.9999],
        'estimator__epsilon': [1e-8, 1e-7, 1e-6]
    },
    # KNeighborsClassifier(): {
    #     'estimator__n_neighbors': [3, 5, 7],
    #     'estimator__weights': ['uniform', 'distance'],
    #     'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    #     'estimator__leaf_size': [30, 40, 50],
    #     'estimator__p': [1, 2]  # for Minkowski distance
    # },
    # GaussianNB(): {
    #     'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]  # Variance smoothing parameter
    # }
}
max_accuracy = dict()
max_features = dict()
for i in range(1, len(features)):
    feature_cols = features[0:i]
    feature_cols.extend(['patient', 'diagnosis'])
    print(f"dropping {feature_cols}")
    X = data.drop(feature_cols, axis=1)
    y = data['diagnosis']

    # One-hot encode the labels
    label_binarizer = LabelBinarizer()
    y_encoded = label_binarizer.fit_transform(y)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Feature scaling
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)

    # Define classifier and param grid options for hyperturning
 
    # Train classifiers
    for classifier in classifiers:
        print(f"training {classifier}")
        multi_target_classifier = MultiOutputClassifier(classifier, n_jobs=-1)
        multi_target_classifier.fit(X_train, y_train)
        # multi_target_classifier.fit(X_train_scaled, y_train)

        # Predict
        y_pred = multi_target_classifier.predict(X_test)
        # y_pred = multi_target_classifier.predict(X_test_scaled)

        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)
        if max_accuracy.get(classifier, 0) < accuracy:
            max_accuracy[classifier] = accuracy
            max_features[classifier] = feature_cols

        # report = classification_report(y_test, y_pred)
        # print("Classification Report:")
        # print(report)

        # Perform grid search with cross-validation
        # param_grid = classifiers.get(classifier)
        # grid_search = GridSearchCV(estimator=multi_target_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
        # # grid_search.fit(X_train_scaled, y_train)
        # grid_search.fit(X_train, y_train)
        #
        # # Get the best parameters and estimator
        # best_params = grid_search.best_params_
        # best_estimator = grid_search.best_estimator_
        #
        # # Use the best estimator to predict
        # # y_pred = best_estimator.predict(X_test_scaled)
        # y_pred = best_estimator.predict(X_test)
        #
        # # Inverse transform the one-hot encoded predictions to get the original labels
        # y_pred_inverse = label_binarizer.inverse_transform(y_pred)
        #
        # # Inverse transform the one-hot encoded true labels to get the original labels
        # y_test_inverse = label_binarizer.inverse_transform(y_test)
        #
        # # Evaluation
        # accuracy = accuracy_score(y_test_inverse, y_pred_inverse)
        # print("Best Parameters:", best_params)
        # print("Accuracy:", accuracy)
        #
        # report = classification_report(y_test_inverse, y_pred_inverse)
        # print("Classification Report:")
        # print(report)

print(max_accuracy)
print(max_features)
