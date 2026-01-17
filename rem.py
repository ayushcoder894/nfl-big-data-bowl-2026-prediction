import pandas as pd

# Data extracted from the image
years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
crime_rates = [181.5, 181.4, 187.6, 192.2, 196.7, 215.5, 229.2, 234.2, 233.6, 237.7, 236.7, 241.2, 314.3, 268.0, 258.1, 270.3]

# Create DataFrame with four columns - two empty columns
df = pd.DataFrame({
    'Year': years,
    'Unemployment Rate (%)': [''] * len(years),  # Empty column
    'Crime Rate (per 100,000 population)': crime_rates,
    'Crime against Property': [''] * len(years)  # Empty column
})

# Save to Excel file
df.to_excel('crime_unemployment_data.xlsx', index=False)

print("Excel file 'crime_unemployment_data.xlsx' updated successfully!")
print("\nPreview of the data:")
print(df)
