import json
import numpy as np

# Your NumPy array
my_numpy_array = np.array([1, 2, 3, 4, 5])

# Convert NumPy array to Python list
my_list = my_numpy_array.tolist()

# Specify the file path where you want to save the JSON file
file_path = 'my_numpy_array.json'

# Open the file in write mode ('w')
with open(file_path, 'w') as json_file:
    # Use json.dump to write the list to the file
    json.dump(my_list, json_file)
