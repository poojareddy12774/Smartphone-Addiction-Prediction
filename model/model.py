import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("teen_smartphone_addiction_dataset.csv")

print("Dataset Loaded Successfully\n")
df.head()

print("\nDataset Shape:", df.shape)

df.drop(columns=[
    'ID',
    'Name',
    'Location',
    'School_Grade',
    'Academic_Performance'
], inplace=True)

print("\nColumns after dropping:")
print(df.columns)

df['Addiction_Level'] = pd.cut(
    df['Addiction_Level'],
    bins=[0,4,7,10],
    labels=['Low','Medium','High']
)

print("\nTarget Distribution:")
print(df['Addiction_Level'].value_counts())

target_encoder = LabelEncoder()

df['Addiction_Level'] = target_encoder.fit_transform(df['Addiction_Level'])

for i, label in enumerate(target_encoder.classes_):
    print(f"{label} -> {i}")

label_encoders = {}

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    
    label_encoders[col] = le
    
    print(f"\nEncoding for {col}")
    
    for i, label in enumerate(le.classes_):
        print(f"{label} -> {i}")

X = df.drop("Addiction_Level", axis=1)

y = df["Addiction_Level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain Shape:", X_train.shape)
print("Test Shape:", X_test.shape)

test_dataset = X_test.copy()
test_dataset["Addiction_Level"] = y_test

test_dataset.to_csv("test_dataset.csv", index=False)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "scaler.pkl")

base_models = [

    ("rf", RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        random_state=42
    )),
    
    ("gb", GradientBoostingClassifier()),
    
    ("svm", SVC(probability=True))

]

meta_model = LogisticRegression()

stack_model = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

stack_model.fit(X_train_scaled, y_train)

joblib.dump(stack_model, "stacking_model.pkl")

y_pred = stack_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n==============================")
print("MODEL PERFORMANCE")
print("==============================")

print("\nAccuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix\n")
print(confusion_matrix(y_test, y_pred))

joblib.dump(label_encoders, "label_encoders.pkl")

joblib.dump(target_encoder, "target_encoder.pkl")