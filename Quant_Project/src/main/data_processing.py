import pandas as pd


file_path = 'E:/Quant_Project/data/data.csv'
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')


dollar_columns = ['Close/Last', 'Open', 'High', 'Low']


for col in dollar_columns:
    data[col] = data[col].str.replace('[^\d.]', '', regex=True) 
    data[col] = pd.to_numeric(data[col], errors='coerce')       


data[dollar_columns] = data[dollar_columns].fillna(0)


data = data.sort_values(by='Date').reset_index(drop=True)


print(data.head())


data.to_csv('E:/Quant_Project/data/cleaned_data.csv', index=False)
