import pandas as pd


file_path = 'E:/Quant_Project/data/features_data.csv' 
data = pd.read_csv(file_path)


data['Date'] = pd.to_datetime(data['Date'], errors='coerce')


data = data.dropna(subset=['Date'])

# Extract Day, Month, Year, and Weekday from the 'Date' column
data['Day'] = data['Date'].dt.day
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year
data['Weekday'] = data['Date'].dt.dayofweek 


data = data.drop(columns=['Date'])


print("Updated DataFrame with Date-Based Features:")
print(data.head())


output_file_path = 'E:/Quant_Project/data/u_dataset.csv'  
data.to_csv(output_file_path, index=False)

print(f"\nUpdated dataset saved to: {output_file_path}")
