import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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


pca_y = PCA(n_components=1)
y_pca = pca_y.fit_transform(X_std_y)

y_manual = X_std_y[:, 1]