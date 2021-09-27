import os

os.makedirs(os.path.join('..', 'Preliminary', 'data'), exist_ok=True)
data_file = os.path.join('..', 'Preliminary',  'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write("NumRooms,Alley,Price\n")
    f.write("NA,Pave,127500\n")
    f.write("2,NA,210900\n")
    f.write("4,NA,178100\n")
    f.write("NA,NA,140000\n")
