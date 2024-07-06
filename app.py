from joblib import load
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures

# x_new=float(input('Enter the spend value on TV :'))

# y_new=float(input('Enter the spend value on Radio :'))

df_new=pd.DataFrame({'tv_power':[45689.02313546,89765.56967864],'radio_root':[89657.0098456,99978.442699]})

#

model_file='file_elasticnet.pickle'

final_model=pickle.load(open(model_file,'rb'))
print(final_model.predict(df_new))



