import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
import numpy as np
import joblib

processed_path = "data/processed/"
files = os.listdir(processed_path)

for file in files:
    if file.endswith(".csv"):
        print(f"\nTraining Ensemble Model for {file}...")

        df = pd.read_csv(os.path.join(processed_path, file))

        features = ['Close', 'Volume', 'MA_7', 'MA_14', 'RSI', 'MACD']
        target = 'Target'

        X = df[features]
        y = df[target]

        # Time-series split (no shuffle)
        split_index = int(len(df) * 0.8)
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        # Models
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
        xgb = XGBRegressor(n_estimators=100, random_state=42)

        # Voting Ensemble
        ensemble = VotingRegressor([
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb)
        ])

        ensemble.fit(X_train, y_train)
        
        os.makedirs("models/ensemble", exist_ok=True)
        joblib.dump(ensemble, f"models/ensemble/{file.replace('.csv','')}_ensemble.pkl")
        
        predictions = ensemble.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)

        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")

print("\nAll Ensemble Models Trained Successfully!")

# An Ensemble Learning approach was implemented using Random Forest, Gradient Boosting, and XGBoost. A Voting Regressor was applied to combine predictions, improving robustness and reducing variance.