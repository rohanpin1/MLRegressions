from PackageClass import ModelImports

class LinearRegression:
    @staticmethod
    def get_linear_regression(X_train, y_train):
            model_libs =  ModelImports.get_imports()
            linear_model = model_libs['LinearRegression']()
            linear_model.fit(X_train, y_train)
            return linear_model        