import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft
#import scrfft

# Read the dataset
data = pd.read_csv('airline5.csv')
data['Date'] = pd.to_datetime(data['Date'], format='%d/%m/%Y')  

# Extract year, month, and day
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
data['Day'] = data['Date'].dt.day


print(data.info())

#  Perform Fourier Transform on daily passenger numbers
passenger_data = data['Number'].values
time = np.arange(len(passenger_data))

# Fourier Transform
fourier_transform = fft(passenger_data)
frequencies = np.fft.fftfreq(len(time), d=1)

# Power spectrum
power_spectrum = np.abs(fourier_transform)**2

#Compute monthly averages
monthly_avg = data.groupby('Month')['Number'].mean()

# Create a bar chart for monthly averages
plt.figure(figsize=(10, 6))
months = np.arange(1, 13)
plt.bar(months, monthly_avg, label='Monthly Average Passengers')
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.title('Average Daily Passengers Monthly Student Id:23081554')
plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid()



sorted_indices = np.argsort(-power_spectrum)
selected_terms = sorted_indices[:8]


reconstructed_signal = np.zeros_like(passenger_data, dtype=float)
for idx in selected_terms:
    reconstructed_signal += (np.real(fourier_transform[idx]) * np.cos(2 * np.pi * frequencies[idx] * time) -
                             np.imag(fourier_transform[idx]) * np.sin(2 * np.pi * frequencies[idx] * time)) / len(passenger_data)

#Add Fourier approximation to plot
width=0.4
plt.figure(figsize=(10, 6))
months = np.arange(1, 13)
plt.bar(months+width, monthly_avg,width, label='Monthly Average Passengers')
plt.bar(data['Month'][:365], reconstructed_signal[:365],width, color='red', label='Fourier Approximation (8 terms)')



#Formatting the plot
plt.xlabel('Month')
plt.ylabel('Passengers')
plt.xticks(months, ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend()
plt.title('Average Daily Passengers and Fourier Approximation')
plt.text(0.9, 0.9, 'Student Id:23081554', transform=plt.gca().transAxes, fontsize=10)
plt.grid()
plt.show()

#Plot the Power Spectrum
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], power_spectrum[:len(power_spectrum)//2])
plt.xlabel('Frequency (day)')
plt.ylabel('Power')
plt.title('Power Spectrum Student Id:23081554')
plt.grid()
plt.show()

#Calculate X and Y (Average Ticket Prices for 2021 and 2022)
data_2021 = data[data['Year'] == 2021]
data_2022 = data[data['Year'] == 2022]

average_price_2021 = data_2021['Price'].sum() / data_2021['Number'].sum()
average_price_2022 = data_2022['Price'].sum() / data_2022['Number'].sum()

print(f"Average Ticket Price in 2021 (X): {average_price_2021:.2f}")
print(f"Average Ticket Price in 2022 (Y): {average_price_2022:.2f}")

#Adding point to Figure 2
plt.figure(figsize=(10, 6))
plt.plot(frequencies[:len(frequencies)//2], power_spectrum[:len(power_spectrum)//2])
plt.xlabel('Frequency (day)')
plt.ylabel('Power')
plt.title('Power Spectrum and Average Price Student Id:23081554')
plt.grid()
plt.scatter(average_price_2021,average_price_2022,color='red')##I tried many time but the scatter point y donot show on the plot together with power spectrum
plt.show()
