import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle

df = pd.read_csv('mbti_tfidf_features.csv', low_memory=False)
feature_cols = [col for col in df.columns if col not in ['type', 'IE', 'NS', 'TF', 'JP', 'original_posts', 'clean_posts']]
X = df[feature_cols]
y = df['IE']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

y_pred = best_rf.predict(X_test)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.2f}")
print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.2f}")
print(f"Recall:    {recall_score(y_test, y_pred, zero_division=0):.2f}")
print(f"F1-score:  {f1_score(y_test, y_pred, zero_division=0):.2f}")

with open('best_random_forest_IE.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
