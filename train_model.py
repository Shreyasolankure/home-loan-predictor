import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Generate Synthetic Dataset
print("Generating synthetic loan dataset...")
np.random.seed(42)
n_samples = 1500

monthly_income = np.random.randint(20000, 200000, n_samples)
age = np.random.randint(21, 65, n_samples)
cibil_score = np.random.randint(300, 900, n_samples)
existing_loan = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])

# Define a realistic, probabilistic logic for loan approval
# Higher CIBIL and Income = Higher approval chance. Existing loan = Penalty.
score = (cibil_score - 300) / 600 * 0.5 + (monthly_income / 200000) * 0.4 - (existing_loan * 0.2)
loan_status = np.where(score > 0.35, 1, 0)

df = pd.DataFrame({
    'monthly_income': monthly_income,
    'age': age,
    'cibil_score': cibil_score,
    'existing_loan': existing_loan,
    'loan_status': loan_status
})
df.to_csv('loan_data.csv', index=False)

# 2. Preprocess Data
X = df.drop('loan_status', axis=1)
y = df['loan_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train Random Forest Model
print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 4. Save Model and Scaler
joblib.dump(model, 'loan_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and Scaler successfully saved!")