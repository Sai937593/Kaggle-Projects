from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

mae_scores = []
for i in range(10):
    pca = PCA(n_components=i+1)
    X_train = pca.fit_transform(X_scaled)
    X_test = pca.fit_transform(X_test_scaled)
    model = LinearRegression(n_jobs=-1)
    model.fit(X_train, y_full)
    y_pred = model.predict(X_test)
    mae_scores.append(mean_absolute_error(y_test, y_pred))
