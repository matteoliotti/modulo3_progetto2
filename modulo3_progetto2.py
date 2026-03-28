import pandas as pd
from sklearn.datasets import load_linnerud
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

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



models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1)
}

results = {}

for target_name, y_target in [('PCA', y_pca), ('Manuale', y_manual)]:
    results[target_name] = {}
    for model_name, model in models.items():
        model.fit(X_std, y_target)
        y_pred = model.predict(X_std)
        mse = mean_squared_error(y_target, y_pred)
        r2 = r2_score(y_target, y_pred)
        results[target_name][model_name] = {'MSE': mse, 'R2': r2}

print(results)



pca_X = PCA(n_components=2)
X_pca = pca_X.fit_transform(X_std)

plt.scatter(X_pca[:, 0], y_pca)

for model_name, model in models.items():
    model.fit(X_pca[:, 0].reshape(-1, 1), y_pca)
    y_pred = model.predict(X_pca[:, 0].reshape(-1, 1))
    plt.plot(X_pca[:, 0], y_pred, label=f"{model_name} (R2={r2_score(y_pca, y_pred):.2f})")

plt.xlabel('PC1 delle Feature')
plt.ylabel('Target (PCA)')
plt.legend()
plt.show()