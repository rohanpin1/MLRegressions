from PackageClass import ModelImports

class GradientBoosting:
    @staticmethod
    def get_gradientboosting(X_train,y_train):
        model_libs = ModelImports.get_imports()
        gb_model = model_libs['GradientBoostingRegressor'](n_estimators=100,learning_rate=0.1,max_depth=3,random_state=42)
        gb_model.fit(X_train,y_train)
        return gb_model