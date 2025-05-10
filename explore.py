import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

# Load your dataset
df = pd.read_csv("C:/Users/revee/OneDrive/Desktop/medicine-disease-predictor/MEDICAL_DATASET.csv")

# Show initial data
print(df.head())
print(df.info())

# Drop rows with missing values and duplicates
df_cleaned = df.dropna().drop_duplicates()

# Standardize column names
df_cleaned.columns = df_cleaned.columns.str.lower().str.replace(' ', '_')

# Convert strength to numeric
df_cleaned['strength'] = df_cleaned['strength'].str.extract(r'(\d+)', expand=False).astype(float)


# Encode all categorical columns
label_encoders = {}
for column in ['name', 'category', 'dosage_form', 'manufacturer', 'indication']:
    le = LabelEncoder()
    df_cleaned[column] = le.fit_transform(df_cleaned[column])
    label_encoders[column] = le

# Save cleaned dataset
df_cleaned.to_csv("cleaned_medical_dataset.csv", index=False)
print("Cleaned dataset saved as 'cleaned_medical_dataset.csv'.")

# Features and target
X = df_cleaned.drop("indication", axis=1)
y = df_cleaned["indication"]

# SMOTE for class balancing
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save label encoders
with open("label_encoders.pkl", "wb") as le_file:
    pickle.dump(label_encoders, le_file)

# Visualizations
plt.figure(figsize=(12, 6))
sns.countplot(data=df_cleaned, x='indication')
plt.xticks(rotation=45)
plt.title('Distribution of Diseases (Target Variable)')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 4))
sns.countplot(data=df_cleaned, x='dosage_form')
plt.xticks(rotation=45)
plt.title('Distribution of Dosage Forms')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df_cleaned['strength'], bins=20, kde=True, ax=axes[0])
axes[0].set_title('Histogram of Strength')
sns.boxplot(x=df_cleaned['strength'], ax=axes[1])
axes[1].set_title('Boxplot of Strength')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(data=df_cleaned, x='indication', y='strength', errorbar=None)
plt.xticks(rotation=45)
plt.title('Average Strength per Disease')
plt.xlabel('Disease (Indication)')
plt.ylabel('Average Strength')
plt.tight_layout()
plt.show()
