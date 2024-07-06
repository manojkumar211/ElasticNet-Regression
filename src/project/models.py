from feature_selection import X,y
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from best_values import lr_best_test
from sklearn.linear_model import LinearRegression,Lasso,LassoCV,Ridge,RidgeCV,ElasticNet,ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.preprocessing import PolynomialFeatures




# ElasticNet Regression Algorithm:-

class ElasticNet_cv:

    lr_best_train=[]
    lr_best_test=[]

    try:
        for i in range(0,20):
            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=i)
            lr=LinearRegression()
            lr.fit(X_train,y_train)
            lr_train_pred=lr.predict(X_train)
            lr_test_pred=lr.predict(X_test)
            lr_best_train.append(lr.score(X_train,y_train))
            lr_best_test.append(lr.score(X_test,y_test))

    except Exception as e:
        raise Exception(f'Best RandomSate Error in ElasticNet :\n'+str(e))

    
    elastic_cv=ElasticNetCV(alphas=None,cv=5)
    elastic_cv.fit(X_train, y_train)
    elastic_alpha=elastic_cv.alpha_
    elastic_l1=elastic_cv.l1_ratio_

    try:

        def __init__(self,elastic_cv,elastic_alpha,elastic_l1):

            self.elastic_cv = elastic_cv
            self.elastic_alpha = elastic_alpha
            self.elastic_l1 = elastic_l1

        def elastic_cv_regression(self):
            return self.elastic_cv
        def elastic_alpha_regression(self):
            return self.elastic_alpha
        def elastic_l1_regression(self):
            return self.elastic_l1
        
    except Exception as e:
        raise Exception(f'Alpha Error in ElasticNet :\n'+str(e))

class ElasticNet_regression(ElasticNet_cv):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=np.argmax(lr_best_test))

    try:

        try:

            elastic_model=ElasticNet(alpha=ElasticNet_cv.elastic_alpha,l1_ratio=ElasticNet_cv.elastic_l1) # type: ignore
            elastic_model.fit(X_train,y_train)
            elastic_train_pred=elastic_model.predict(X_train)
            elastic_test_pred=elastic_model.predict(X_test)
            elastic_tr_score=elastic_model.score(X_train,y_train)
            elastic_te_score=elastic_model.score(X_test,y_test)
            elastic_train_mae=mean_absolute_error(y_train,elastic_train_pred)
            elastic_train_mse=mean_squared_error(y_train,elastic_train_pred)
            elastic_train_rmse=np.sqrt(mean_squared_error(y_train,elastic_train_pred))
            elastic_test_mae=mean_absolute_error(y_test,elastic_test_pred)
            elastic_test_mse=mean_squared_error(y_test,elastic_test_pred)
            elastic_test_rmse=np.sqrt(mean_squared_error(y_test,elastic_test_pred))

        except Exception as e:
            raise Exception(f'Error find in ElasticNet Regression :\n'+str(e))

        try:

            def __init__(self,elastic_cv,elastic_alpha,elastic_l1,elastic_model,elastic_train_pred,elastic_test_pred,elastic_tr_score,elastic_te_score,
                        elastic_train_mae,elastic_train_mse,elastic_train_rmse,elastic_test_mae,elastic_test_mse,elastic_test_rmse,lr_best_train,lr_best_test):
                    
                try:
                
                    self.elastic_cv=elastic_cv
                    self.elastic_alpha=elastic_alpha
                    self.elastic_l1=elastic_l1
                    self.elastic_model=elastic_model
                    self.elastic_train_pred=elastic_train_pred
                    self.elastic_test_pred=elastic_test_pred
                    self.elastic_tr_score=elastic_tr_score
                    self.elastic_te_score=elastic_te_score
                    self.elastic_train_mae=elastic_train_mae
                    self.elastic_train_mse=elastic_train_mse
                    self.elastic_train_rmse=elastic_train_rmse
                    self.elastic_test_mae=elastic_test_mae
                    self.elastic_test_mse=elastic_test_mse
                    self.elastic_test_rmse=elastic_test_rmse
                    self.lr_best_train=lr_best_train
                    self.lr_best_test=lr_best_test

                except Exception as e:
                    raise Exception(f'Error find in ElasticNet at Initiate level :\n'+str(e))

            try:


                def elastic_cv_regression(self):
                    return super().elastic_cv
                def elastic_alpha_regression(self):
                    return super().elastic_alpha
                def elastic_l1_regression(self):
                    return super().elastic_l1
                def elastic_model_regression(self):
                    return self.elastic_model
                def elastic_train_pred_regression(self):
                    return self.elastic_train_pred
                def elastic_test_pred_regression(self):
                    return self.elastic_test_pred
                def elastic_train_score_regression(self):
                    return self.elastic_tr_score
                def elastic_test_score_regression(self):
                    return self.elastic_te_score
                def elastic_train_mae_regression(self):
                    return self.elastic_train_mae
                def elastic_train_mse_regression(self):
                    return self.elastic_train_mse
                def elastic_train_rmse_regression(self):
                    return self.elastic_train_rmse
                def elastic_test_mae_regression(self):
                    return self.elastic_test_mae
                def elastic_test_mse_regression(self):
                    return self.elastic_test_mse
                def elastic_test_rmse_regression(self):
                    return self.elastic_test_rmse
                def lr_best_train_poly(self):
                    return super().lr_best_train
                def lr_best_test_poly(self):
                    return super().lr_best_test
                
            except Exception as e:
                raise Exception(f'Error find in ElasticNet at Defining level :\n'+str(e))
            
        except Exception as e:
            raise Exception(f'Error find in ElasticNet at Initiate and Defining level :\n'+str(e))

    except Exception as e:
        raise Exception(f'Total Error in ElasticNet Regression :\n'+str(e))

        



