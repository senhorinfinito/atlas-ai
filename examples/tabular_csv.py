import atlas
import os
import lance
import pandas as pd

import atlas
import os
import lance
import pandas as pd

# Define the path to the CSV and Lance datasets
csv_path = "examples/data/titanic.csv"
lance_path = "examples/data/tabular.lance"
os.makedirs(os.path.dirname(lance_path), exist_ok=True)

# Create a dummy CSV file
data = {
    'pclass': [1, 2, 3, 1, 2, 3, 1, 2, 3, 1],
    'survived': [1, 1, 0, 0, 0, 1, 1, 0, 1, 0],
    'name': [
        "Allen, Miss. Elisabeth Walton",
        "Caldwell, Mr. Albert Francis",
        "Sage, Miss. Stella Anna",
        "Allison, Miss. Helen Loraine",
        "Mallet, Mr. Albert",
        "McGowan, Miss. Anna",
        "Baxter, Mr. Quigg Edmond",
        "Corey, Mrs. Percy C",
        "Johnson, Miss. Eleanor Ileen",
        "Bonnell, Miss. Elizabeth"
    ],
    'sex': ['female', 'male', 'female', 'female', 'male', 'female', 'male', 'female', 'female', 'female'],
    'age': [29, 26, 4, 2, 31, 15, 45, 35, 1, 58],
    'sibsp': [0, 1, 8, 1, 1, 0, 0, 0, 1, 0],
    'parch': [0, 1, 2, 2, 1, 0, 2, 0, 1, 0],
    'ticket': [24160, 248738, 'CA. 2343', 113781, 'S.C./PARIS 2079', 330923, 113572, 113789, 347742, 113783],
    'fare': [211.3375, 29.0, 69.55, 151.55, 37.0042, 8.0292, 227.525, 26.55, 11.1333, 26.55],
    'cabin': ['B5', 'D', '', 'C22 C26', '', '', 'B58 B60', '', 'G3', 'C103'],
    'embarked': ['S', 'C', 'S', 'S', 'S', 'Q', 'C', 'S', 'S', 'S'],
    'boat': ['2', '13', '', '', '', '6', 'D', '', '15', '8'],
    'body': ['', '', '', '', '', '', '', '', '', ''],
    'home.dest': ['St Louis, MO', 'Bangkok, Thailand / Roseville, IL', 'London / Kaggle', 'Montreal, PQ / Chesterville, ON', 'Paris / Montreal, PQ', 'Kaggle', 'Montreal, PQ', 'Kaggle', 'Kaggle', 'Birkdale, England Cleveland, OH']
}
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)


# Sink the CSV file to a Lance dataset
print(f"Sinking {csv_path} to {lance_path}...")
atlas.sink(csv_path, lance_path, mode="overwrite")

# Verify that the dataset was created and is not empty
print("Verifying dataset...")
dataset = lance.dataset(lance_path)
assert dataset.count_rows() == 10, "The number of rows in the dataset does not match the number of rows in the CSV file"

# Verify the contents of the dataset
table = dataset.to_table()
original_df = pd.read_csv(csv_path)
assert table.num_columns == len(original_df.columns), "Column count mismatch"
assert table.to_pydict()['pclass'][0] == 1, "Data mismatch in 'pclass' column"
assert table.to_pydict()['name'][0] == "Allen, Miss. Elisabeth Walton", "Data mismatch in 'name' column"

print("Tabular CSV dataset sinking example executed successfully.")

