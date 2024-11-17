from PackageClass import ModelImports

class RandomForest:
    @staticmethod
    def get_random_forest(X_train,y_train):
        model_libs = ModelImports.get_imports();
        forest_model = model_libs['RandomForestRegressor'](n_estimators=100, random_state=42)
        forest_model.fit(X_train, y_train)
        return forest_model