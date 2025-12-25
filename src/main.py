import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    return df

def clean_data(df):
    print("Cleaning data...")
    # Check for missing values
    if df.isnull().sum().sum() > 0:
        print("Found missing values. Dropping...")
        df = df.dropna()
    else:
        print("No missing values found.")
    
    # Remove outliers (simple z-score or IQR could be used, here just manual check on surface/price)
    # Keeping it simple for now as generated data is relatively clean but with noise.
    return df

def perform_eda(df, output_dir):
    print("Performing EDA...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Price Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Price'], kde=True, bins=30)
    plt.title('Distribution of Apartment Prices')
    plt.xlabel('Price')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'price_distribution.png'))
    plt.close()
    
    # Price vs Surface
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='Surface', y='Price', hue='Location', alpha=0.6)
    plt.title('Price vs Surface by Location')
    plt.savefig(os.path.join(output_dir, 'price_vs_surface.png'))
    plt.close()
    
    # Correlation Heatmap
    # Select only numeric columns for correlation matrix
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    print(f"EDA plots saved to {output_dir}")

def train_and_evaluate(df):
    print("Training and evaluating models...")
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    # Preprocessing
    categorical_features = ['Location']
    numerical_features = ['Surface', 'Rooms', 'Bedrooms', 'Floor', 'Elevator', 'Balcony']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"\n{name} Results:")
        print(f"MAE: {mae:,.2f}")
        print(f"RMSE: {rmse:,.2f}")
        print(f"R² Score: {r2:.4f}")
        
        if name == 'Random Forest':
             # Feature Importance
            rf_model = pipeline.named_steps['model']
            
            # Get feature names from one-hot encoder
            ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat']
            cat_feature_names = ohe.get_feature_names_out(categorical_features)
            feature_names = numerical_features + list(cat_feature_names)
            
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\nFeature Importances (Random Forest):")
            for f in range(len(feature_names)):
                print(f"{feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

            # Plot Feature Importance
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importances")
            plt.bar(range(len(feature_names)), importances[indices], align="center")
            plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join("output", 'feature_importance.png'))
            plt.close()

    return results

if __name__ == "__main__":
    data_path = "data/apartments.csv"
    output_dir = "output"
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found. Please run src/generate_data.py first.")
        exit(1)
        
    df = load_data(data_path)
    df = clean_data(df)
    perform_eda(df, output_dir)
    results = train_and_evaluate(df)
    
    print("\nProcessing complete.")
