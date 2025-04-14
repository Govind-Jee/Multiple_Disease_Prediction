# train_models.py

import os
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

# 1. Load datasets
df_d = pd.read_csv('D.csv')    # Diabetes
df_h = pd.read_csv('H.csv')    # Heart Disease
df_p = pd.read_csv('P.csv')    # Parkinson's

# 2. Prepare features (X) and labels (y)
X_d = df_d.drop('Outcome', axis=1)
y_d = df_d['Outcome']

X_h = df_h.drop('target', axis=1)
y_h = df_h['target']

X_p = df_p.drop(['name','status'], axis=1)
y_p = df_p['status']

# 3. Split into train/test sets (20% test)
Xd_train, Xd_test, yd_train, yd_test = train_test_split(
    X_d, y_d, test_size=0.2, random_state=42, stratify=y_d
)
Xh_train, Xh_test, yh_train, yh_test = train_test_split(
    X_h, y_h, test_size=0.2, random_state=42, stratify=y_h
)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_p, y_p, test_size=0.2, random_state=42, stratify=y_p
)

# 4. Feature scaling
scaler_d = StandardScaler().fit(Xd_train)
Xd_train_scaled = scaler_d.transform(Xd_train)
Xd_test_scaled  = scaler_d.transform(Xd_test)

scaler_h = StandardScaler().fit(Xh_train)
Xh_train_scaled = scaler_h.transform(Xh_train)
Xh_test_scaled  = scaler_h.transform(Xh_test)

scaler_p = StandardScaler().fit(Xp_train)
Xp_train_scaled = scaler_p.transform(Xp_train)
Xp_test_scaled  = scaler_p.transform(Xp_test)

# 5. Train models
diabetes_model = SVC(kernel='linear', probability=True)
diabetes_model.fit(Xd_train_scaled, yd_train)
print("Diabetes test accuracy:", diabetes_model.score(Xd_test_scaled, yd_test))

heart_model = LogisticRegression(max_iter=1000)
heart_model.fit(Xh_train_scaled, yh_train)
print("Heart disease test accuracy:", heart_model.score(Xh_test_scaled, yh_test))

parkinsons_model = SVC(kernel='rbf', probability=True)
parkinsons_model.fit(Xp_train_scaled, yp_train)
print("Parkinson’s test accuracy:", parkinsons_model.score(Xp_test_scaled, yp_test))

# 6. Create folder for saved models
os.makedirs('saved_models', exist_ok=True)

# 7. Serialize (pickle) each model to .sav
with open('saved_models/diabetes_model.sav', 'wb') as f:
    pickle.dump(diabetes_model, f)

with open('saved_models/heart_disease_model.sav', 'wb') as f:
    pickle.dump(heart_model, f)

with open('saved_models/parkinsons_model.sav', 'wb') as f:
    pickle.dump(parkinsons_model, f)

print("All models have been trained and saved to 'saved_models/'")
