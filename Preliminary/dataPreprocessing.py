import os
import pandas as pd
import torch

data_file = os.path.join('..', 'Preliminary', 'data', 'house_tiny.csv')
data = pd.read_csv(data_file)
print(data, '\n')

# print(data['NumRooms'])
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]  # use iloc for slicing in "DataFrame" datatype
# values = {'NumRooms': inputs.mean().item(), 'Alley': 'Road'}
inputNew = inputs.fillna(inputs.mean())
print(inputNew, '\n')

inputNew = pd.get_dummies(inputNew, dummy_na=True)
print(inputNew, '\n')

X, y = torch.tensor(inputNew.values), torch.tensor(outputs.values)
print(X)
print(y)

############# Practice #############
data.drop(columns=[data.isna().sum().idxmax()], inplace=True)
print("\nAfter deleting: \n", data)
data = data.fillna(data.mean())
print("\nAfter Filling the NaN: \n", data)
