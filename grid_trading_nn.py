import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def prepare_data(df, look_back=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i])
        y.append(scaled_data[i])
    
    return np.array(X), np.array(y), scaler

def create_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=input_shape[1])
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def train_model(crypto1_data, crypto2_data, look_back=60, epochs=50, batch_size=32):
    # Combinar datos de ambas criptomonedas
    combined_data = pd.concat([crypto1_data, crypto2_data], axis=1)
    
    X, y, scaler = prepare_data(combined_data, look_back)
    
    model = create_model((X.shape[1], X.shape[2]))
    model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    return model, scaler

def predict_grid_parameters(model, scaler, last_data, num_grids=10):
    last_scaled = scaler.transform(last_data.reshape(1, -1))
    prediction = model.predict(last_scaled.reshape(1, 1, -1))
    predicted_prices = scaler.inverse_transform(prediction)[0]
    
    # Calcular parámetros del grid
    min_price = min(predicted_prices)
    max_price = max(predicted_prices)
    grid_size = (max_price - min_price) / num_grids
    
    return min_price, max_price, grid_size

def evaluate_grid_performance(actual_prices, min_price, max_price, grid_size, initial_investment=1000):
    num_grids = int((max_price - min_price) / grid_size)
    grid_levels = np.linspace(min_price, max_price, num_grids + 1)
    
    portfolio = initial_investment
    crypto_holdings = 0
    trades = 0
    
    for i in range(1, len(actual_prices)):
        prev_price = actual_prices[i-1]
        current_price = actual_prices[i]
        
        for level in grid_levels:
            if prev_price < level <= current_price:  # Precio subió por encima del nivel
                if portfolio >= level:
                    buy_amount = portfolio * 0.1  # Compra 10% del portfolio
                    crypto_holdings += buy_amount / level
                    portfolio -= buy_amount
                    trades += 1
            elif current_price <= level < prev_price:  # Precio bajó por debajo del nivel
                if crypto_holdings > 0:
                    sell_amount = crypto_holdings * 0.1  # Vende 10% de las criptomonedas
                    portfolio += sell_amount * level
                    crypto_holdings -= sell_amount
                    trades += 1
    
    final_value = portfolio + crypto_holdings * actual_prices[-1]
    roi = (final_value - initial_investment) / initial_investment * 100
    
    return {
        "ROI": roi,
        "Número de operaciones": trades,
        "Valor final del portfolio": final_value
    }

# Ejemplo de uso
crypto1_data = pd.read_csv('crypto1_data.csv')
crypto2_data = pd.read_csv('crypto2_data.csv')

model, scaler = train_model(crypto1_data, crypto2_data)

last_data = np.concatenate([crypto1_data.iloc[-1], crypto2_data.iloc[-1]])
min_price, max_price, grid_size = predict_grid_parameters(model, scaler, last_data)

print(f"Parámetros recomendados para el grid trading:")
print(f"Precio mínimo: {min_price}")
print(f"Precio máximo: {max_price}")
print(f"Tamaño del grid: {grid_size}")

# Evaluación del rendimiento
test_data = crypto1_data['close'].values[-100:]  # Últimos 100 precios de cierre para evaluación
performance = evaluate_grid_performance(test_data, min_price, max_price, grid_size)

print("\nRendimiento del Grid Trading:")
for metric, value in performance.items():
    print(f"{metric}: {value}")
