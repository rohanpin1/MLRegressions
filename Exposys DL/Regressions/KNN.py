from PackageClass import ModelImports

class KNN:
    @staticmethod
    def get_knn(X_train,y_train):
        model_libs = ModelImports.get_imports()
        knn_model = model_libs['KNeighborsRegressor'](n_neighbors=7);
        knn_model.fit(X_train, y_train)
        return knn_model