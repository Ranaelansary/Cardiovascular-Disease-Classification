import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
# 1. Load Dataset
df = pd.read_csv("data.csv", sep=";")
print("First 5 rows:")
print(df.head())
print("\nOriginal Dataset Shape:")
print(df.shape)
# 2. Data Cleaning
# Remove duplicate rows
print("\nDuplicates before cleaning:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplicates after cleaning:", df.duplicated().sum())
# Convert age from days to years
df["age"] = df["age"] / 365.25
# 3. Remove Impossible / Extreme Values
df = df[
    (df["ap_hi"] >= 90) &
    (df["ap_hi"] <= 250) &
    (df["ap_lo"] >= 60) &
    (df["ap_lo"] <= 150) &
    (df["ap_hi"] > df["ap_lo"]) &
    (df["height"] >= 140) &
    (df["height"] <= 210) &
    (df["weight"] >= 40) &
    (df["weight"] <= 200)
].copy()

print("\nDataset Shape After Cleaning:")
print(df.shape)
# 4. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())
# 5. Feature Engineering
# BMI
df["bmi"] = df["weight"] / ((df["height"] / 100) ** 2)
# Pulse Pressure
df["pulse_pressure"] = df["ap_hi"] - df["ap_lo"]
# Mean Arterial Pressure
df["map"] = (df["ap_hi"] + 2 * df["ap_lo"]) / 3
# High systolic blood pressure
df["ap_hi_high"] = (df["ap_hi"] >= 140).astype(int)
# High diastolic blood pressure
df["ap_lo_high"] = (df["ap_lo"] >= 90).astype(int)
# Any high blood pressure
df["bp_high_any"] = (
    (df["ap_hi"] >= 140) |
    (df["ap_lo"] >= 90)
).astype(int)

# Age × Systolic Blood Pressure
df["age_x_aphi"] = df["age"] * df["ap_hi"]
# 6. Separate Features and Targe
# Remove ID because it does not represent
# a meaningful medical feature
X = df.drop(columns=["id", "cardio"])
y = df["cardio"]
print("\nFeature Shape:")
print(X.shape)
print("\nTarget Distribution:")
print(y.value_counts())
# 7. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data:")
print(X_train.shape)

print("Testing Data:")
print(X_test.shape)
# 8. Define Features
numeric_features = [
    "age",
    "height",
    "weight",
    "ap_hi",
    "ap_lo",
    "bmi",
    "pulse_pressure",
    "map",
    "age_x_aphi"
]

categorical_features = [
    "cholesterol",
    "gluc"
]
# 9. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 10. Logistic Regression
# ==========================================

logistic_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            random_state=42,
            max_iter=2000
        )
    )
])


logistic_params = {
    "model__C": [
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        2,
        5,
        10,
        20
    ]
}
# 11. KNN

knn_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        KNeighborsClassifier()
    )
])


knn_params = {
    "model__n_neighbors": [
        5,
        9,
        13,
        17,
        21,
        25,
        31,
        41,
        51
    ],
    "model__weights": [
        "uniform",
        "distance"
    ],
    "model__p": [
        1,
        2
    ]
}
# 12. Decision Tree
dt_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        DecisionTreeClassifier(
            random_state=42
        )
    )
])


dt_params = {
    "model__max_depth": [
        3,
        4,
        5,
        6,
        7,
        8,
        10,
        12,
        15
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
        20,
        30
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4,
        6,
        8,
        10
    ],
    "model__criterion": [
        "gini",
        "entropy"
    ]
}
# 13. Random Forest

rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1
        )
    )
])


rf_params = {
    "model__n_estimators": [
        100,
        150,
        200,
        250,
        300
    ],
    "model__max_depth": [
        6,
        8,
        10,
        12,
        15,
        20,
        None
    ],
    "model__min_samples_split": [
        2,
        5,
        10,
        15
    ],
    "model__min_samples_leaf": [
        1,
        2,
        4,
        6
    ],
    "model__max_features": [
        "sqrt",
        "log2"
    ]
}


# ==========================================
# 14. Tuning Settings
# ==========================================

TUNE_SIZE = 20000

X_tune, _, y_tune, _ = train_test_split(
    X_train,
    y_train,
    train_size=TUNE_SIZE,
    random_state=42,
    stratify=y_train
)
# 15. Tune Logistic Regression

print("Tuning Logistic Regression...")
logistic_search = RandomizedSearchCV(
    estimator=logistic_pipeline,
    param_distributions=logistic_params,
    n_iter=8,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    refit=False
)

logistic_search.fit(X_tune, y_tune)

best_logistic_params = logistic_search.best_params_

print("Best Logistic Regression Parameters:")
print(best_logistic_params)


# ==========================================
# 16. Tune KNN
# ==========================================

print("\n==========================================")
print("Tuning KNN...")
print("==========================================")

knn_search = RandomizedSearchCV(
    estimator=knn_pipeline,
    param_distributions=knn_params,
    n_iter=8,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    refit=False
)

knn_search.fit(X_tune, y_tune)
best_knn_params = knn_search.best_params_
print("Best KNN Parameters:")
print(best_knn_params)
# 17. Tune Decision Tree
print("\n==========================================")
print("Tuning Decision Tree...")
print("==========================================")

dt_search = RandomizedSearchCV(
    estimator=dt_pipeline,
    param_distributions=dt_params,
    n_iter=12,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    refit=False
)

dt_search.fit(X_tune, y_tune)
best_dt_params = dt_search.best_params_
print("Best Decision Tree Parameters:")
print(best_dt_params)
# 18. Tune Random Forest

print("Tuning Random Forest...")
rf_search = RandomizedSearchCV(
    estimator=rf_pipeline,
    param_distributions=rf_params,
    n_iter=12,
    scoring="accuracy",
    cv=3,
    random_state=42,
    n_jobs=-1,
    refit=False
)

rf_search.fit(X_tune, y_tune)

best_rf_params = rf_search.best_params_

print("Best Random Forest Parameters:")
print(best_rf_params)
# 19. Create Final Models
final_logistic = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        LogisticRegression(
            random_state=42,
            max_iter=2000,
            **{
                key.replace("model__", ""): value
                for key, value in best_logistic_params.items()
            }
        )
    )
])


final_knn = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        KNeighborsClassifier(
            **{
                key.replace("model__", ""): value
                for key, value in best_knn_params.items()
            }
        )
    )
])


final_dt = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        DecisionTreeClassifier(
            random_state=42,
            **{
                key.replace("model__", ""): value
                for key, value in best_dt_params.items()
            }
        )
    )

])
final_rf = Pipeline([
    ("preprocessor", preprocessor),
    (
        "model",
        RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **{
                key.replace("model__", ""): value
                for key, value in best_rf_params.items()
            }
        )
    )
])

# 20. Train Final Models
print("\n==========================================")
print("Training Final Models...")
print("==========================================")
final_logistic.fit(X_train, y_train)
final_knn.fit(X_train, y_train)
final_dt.fit(X_train, y_train)
final_rf.fit(X_train, y_train)
print("All models trained successfully.")
# 21. Predictions
y_pred_logistic = final_logistic.predict(X_test)
y_pred_knn = final_knn.predict(X_test)
y_pred_dt = final_dt.predict(X_test)
y_pred_rf = final_rf.predict(X_test)
# 22. Logistic Regression Evaluation=
accuracy_logistic = accuracy_score(
    y_test,
    y_pred_logistic
)

precision_logistic = precision_score(
    y_test,
    y_pred_logistic
)

recall_logistic = recall_score(
    y_test,
    y_pred_logistic
)

f1_logistic = f1_score(
    y_test,
    y_pred_logistic
)

cm_logistic = confusion_matrix(
    y_test,
    y_pred_logistic
)

print("Logistic Regression Results")
print("Accuracy:", accuracy_logistic)
print("Precision:", precision_logistic)
print("Recall:", recall_logistic)
print("F1-Score:", f1_logistic)
print("\nConfusion Matrix:")
print(cm_logistic)
# 23. KNN Evaluation
accuracy_knn = accuracy_score(
    y_test,
    y_pred_knn
)

precision_knn = precision_score(
    y_test,
    y_pred_knn
)

recall_knn = recall_score(
    y_test,
    y_pred_knn
)

f1_knn = f1_score(
    y_test,
    y_pred_knn
)

cm_knn = confusion_matrix(
    y_test,
    y_pred_knn
)

print("KNN Results")
print("Accuracy:", accuracy_knn)
print("Precision:", precision_knn)
print("Recall:", recall_knn)
print("F1-Score:", f1_knn)
print("\nConfusion Matrix:")
print(cm_knn)
# 24. Decision Tree Evaluation

accuracy_dt = accuracy_score(
    y_test,
    y_pred_dt
)

precision_dt = precision_score(
    y_test,
    y_pred_dt
)

recall_dt = recall_score(
    y_test,
    y_pred_dt
)

f1_dt = f1_score(
    y_test,
    y_pred_dt
)
cm_dt = confusion_matrix(
    y_test,
    y_pred_dt
)
print("Decision Tree Results")
print("Accuracy:", accuracy_dt)
print("Precision:", precision_dt)
print("Recall:", recall_dt)
print("F1-Score:", f1_dt)
print("\nConfusion Matrix:")
print(cm_dt)
# 25. Random Forest Evaluation

accuracy_rf = accuracy_score(
    y_test,
    y_pred_rf
)
precision_rf = precision_score(
    y_test,
    y_pred_rf
)
recall_rf = recall_score(
    y_test,
    y_pred_rf
)
f1_rf = f1_score(
    y_test,
    y_pred_rf
)
cm_rf = confusion_matrix(
    y_test,
    y_pred_rf
)
print("Random Forest Results")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-Score:", f1_rf)

print("\nConfusion Matrix:")
print(cm_rf)
# 26. Final Results Comparison
results = pd.DataFrame({

    "Algorithm": [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest"
    ],

    "Accuracy": [
        accuracy_logistic,
        accuracy_knn,
        accuracy_dt,
        accuracy_rf
    ],

    "Precision": [
        precision_logistic,
        precision_knn,
        precision_dt,
        precision_rf
    ],

    "Recall": [
        recall_logistic,
        recall_knn,
        recall_dt,
        recall_rf
    ],

    "F1-Score": [
        f1_logistic,
        f1_knn,
        f1_dt,
        f1_rf
    ]
})
print("FINAL RESULTS")
print(results)

# 27. Accuracy Plot
plt.figure(figsize=(8, 5))
plt.bar(
    results["Algorithm"],
    results["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.ylim(0.6, 0.8)

plt.xticks(rotation=15)

plt.tight_layout()

plt.show()