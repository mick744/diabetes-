Python 3.13.0 (tags/v3.13.0:60403a5, Oct  7 2024, 09:38:07) [MSC v.1941 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> # Step 1: Import Libraries
... import pandas as pd
... import numpy as np
... from sklearn.model_selection import train_test_split
... from sklearn.preprocessing import StandardScaler
... from sklearn.ensemble import RandomForestClassifier
... from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
... 
... # Step 2: Load Dataset
... data = pd.read_csv('diabetes.csv')  # replace with your path
... 
... # Step 3: Preprocess Data
... X = data.drop('Outcome', axis=1)
... y = data['Outcome']
... X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
... 
... scaler = StandardScaler()
... X_train = scaler.fit_transform(X_train)
... X_test = scaler.transform(X_test)
... 
... # Step 4: Train Model
... model = RandomForestClassifier()
... model.fit(X_train, y_train)
... 
... # Step 5: Evaluate Model
... y_pred = model.predict(X_test)
... print("Accuracy:", accuracy_score(y_test, y_pred))
... print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
