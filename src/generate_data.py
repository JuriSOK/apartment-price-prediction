import pandas as pd
import numpy as np
import random
import os

def generate_apartment_data(n_samples=1000):
    """
    Generates a synthetic dataset of apartment listings.
    """
    np.random.seed(42)
    random.seed(42)

    # Features
    surfaces = np.random.normal(loc=70, scale=30, size=n_samples)
    surfaces = np.clip(surfaces, 15, 200)  # Clip between 15 and 200 sqm
    surfaces = np.round(surfaces, 1)

    rooms = []
    bedrooms = []
    
    for surf in surfaces:
        # Number of rooms roughly correlated with surface
        n_rooms = max(1, int(surf / 25) + np.random.randint(-1, 2))
        rooms.append(n_rooms)
        # Bedrooms usually rooms - 1 or rooms - 2 (min 0)
        n_beds = max(0, n_rooms - np.random.randint(1, 3))
        bedrooms.append(n_beds)

    floors = np.random.randint(0, 15, size=n_samples) # 0 to 14
    
    locations = np.random.choice(['City Center', 'Suburbs', 'Countryside', 'Downtown'], 
                                 size=n_samples, p=[0.3, 0.4, 0.1, 0.2])
    
    has_elevator = []
    for f in floors:
        if f > 4:
            has_elevator.append(np.random.choice([0, 1], p=[0.05, 0.95])) # High chance of elevator
        else:
            has_elevator.append(np.random.choice([0, 1], p=[0.6, 0.4]))
            
    has_balcony = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    # Price Calculation (Synthetic Formula)
    # Base price per sqm varies by location
    base_price_map = {
        'City Center': 8000,
        'Downtown': 7500,
        'Suburbs': 4500,
        'Countryside': 2500
    }
    
    prices = []
    for i in range(n_samples):
        loc = locations[i]
        surf = surfaces[i]
        floor = floors[i]
        elev = has_elevator[i]
        balc = has_balcony[i]
        
        sqm_price = base_price_map[loc]
        
        # Adjustments
        # Floor: higher is usually better, but ground floor (0) might be lower
        if floor == 0:
            sqm_price *= 0.95
        elif floor > 1 and floor < 5:
            sqm_price *= 1.02
        elif floor >= 5:
            sqm_price *= 1.05
            
        # Elevator bonus for high floors
        if floor > 3 and elev == 1:
            sqm_price *= 1.05
        elif floor > 3 and elev == 0:
            sqm_price *= 0.85 # Penalty for no elevator on high floor
            
        # Balcony bonus
        if balc == 1:
            sqm_price *= 1.03
            
        # Random noise
        noise = np.random.normal(0, 0.05) # +/- 5% noise
        sqm_price *= (1 + noise)
        
        total_price = int(sqm_price * surf)
        prices.append(total_price)

    df = pd.DataFrame({
        'Surface': surfaces,
        'Rooms': rooms,
        'Bedrooms': bedrooms,
        'Floor': floors,
        'Location': locations,
        'Elevator': has_elevator,
        'Balcony': has_balcony,
        'Price': prices
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic dataset...")
    df = generate_apartment_data(1000)
    
    output_path = "data/apartments.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(df.head())
    print(df.info())
