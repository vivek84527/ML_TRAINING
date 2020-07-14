from sklearn.preprocessing import MinMaxScaler
import numpy as np

data = np.array([[-1, 2],
                 [-0.5, 6],
                 [0, 10],
                 [1, 18]])

print(data)

scaler = MinMaxScaler(feature_range=(1, 5))
print(scaler.fit(data))
print("Minimum Values : ", scaler.data_min_)
print("Maximum Values : ", scaler.data_max_)

rescaledX = scaler.transform(data)
print(rescaledX)
