# Apartment Price Prediction

## Project Overview

This project aims to predict the selling price of an apartment based on its characteristics (surface, number of rooms, location, etc.). It implements a full data science workflow including data generation, cleaning, exploratory data analysis (EDA), and machine learning modeling.

## Project Structure

```
├── data/
│   └── apartments.csv       # Generated synthetic dataset
├── output/
│   ├── price_distribution.png
│   ├── price_vs_surface.png
│   ├── correlation_heatmap.png
│   └── feature_importance.png
├── src/
│   ├── generate_data.py     # Script to generate synthetic data
│   └── main.py              # Main script for analysis and modeling
├── README.md                # Project documentation
└── requirements.txt         # List of dependencies
```

## Setup

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Generate Data**

   Run the data generation script to create a synthetic dataset of 1000 apartments.

   ```bash
   python src/generate_data.py
   ```

3. **Run Analysis and Modeling**

   Execute the main script to perform EDA and train models.

   ```bash
   python src/main.py
   ```

## Methodology

1.  **Data Generation**: We created a synthetic dataset with features like Surface, Rooms, Bedrooms, Floor, Location, Elevator, and Balcony. Prices were generated using a base price per sqm depending on location, with adjustments for floor, elevator, and balcony, plus random noise.
2.  **EDA**: Visualized price distribution, relationship between price and surface, and feature correlations.
3.  **Modeling**:
    *   **Linear Regression**: Used as a baseline.
    *   **Random Forest Regressor**: Used to capture non-linear relationships.
4.  **Evaluation**: Models were evaluated using MAE, RMSE, and R² Score.

## Results

(Example results from a run)

*   **Linear Regression**: R² ~ 0.91
*   **Random Forest**: R² ~ 0.96

Random Forest performed better, likely due to its ability to capture the specific pricing rules (e.g., floor premiums, elevator interactions) used in the data generation process.

**Key Drivers**:
*   Surface area is the most dominant factor.
*   Location is the second most important determinant.
