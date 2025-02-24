import pandas as pd
from sklearn.datasets import load_iris

# Load the Iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

# Save as CSV
df.to_csv("iris_dataset.csv", index=False)
print("CSV file saved as 'iris_dataset.csv'")
