import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler

linnerud = load_linnerud()
X = linnerud.data
y = linnerud.target
df_X = pd.DataFrame(X, columns=linnerud.feature_names)
df_y = pd.DataFrame(y, columns=linnerud.target_names)

print(df_X.describe())
print(df_y.describe())

scaler_X = StandardScaler()
X_std = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
X_std_y = scaler_y.fit_transform(y)