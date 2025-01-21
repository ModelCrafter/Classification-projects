import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt 
import numpy as np
import zipfile
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score , GridSearchCV 
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import  ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler , FunctionTransformer
from sklearn.metrics import accuracy_score , precision_score , f1_score , recall_score

 
# load data after I downloaded it before
def load_data() -> pd.DataFrame:
    tarball_path = Path("D:/python/datasets/archive.zip")
    if not tarball_path.exists():
        print('in if')
        Path('datasets').mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(tarball_path) as data:
            data.extractall(path='datasets/crop_recommendation')
    
    return pd.read_csv(Path('datasets/crop_recommendation/Crop_recommendation.csv'))


# init the data (split (train , test) then split (X , Y) )
main_data = load_data()
train , test = train_test_split(main_data , test_size = 0.2 , random_state=42)

X_train : pd.DataFrame = train.drop('label' , axis = 1)
Y_train = train['label']

X_test = test.drop('label' , axis = 1)
Y_test = test[ 'label']



# make preprocessing

# make func can caluc ph
def determine_ph(ph_value):
    if (ph_value == 7):
        return 0
    elif (ph_value > 7):
        return 1
    else:
        return -1



def add_ph_type(df:pd.DataFrame):
    df['ph_type'] = df['ph'].apply(func=determine_ph)
    return df

def add_sum_all(df:pd.DataFrame):
    num = df.select_dtypes(include=['number']).columns
    df['sum_all'] = df[num].sum(axis=1)
    return df

preprocess = ColumnTransformer([
    ('ph_pro' , FunctionTransformer(add_ph_type) , ['ph']),
    ('sum_pro' , FunctionTransformer(add_sum_all) , X_train.columns)],remainder=StandardScaler())

finally_pip = Pipeline([
    ('processing' , preprocess),
    ('classifier' , RandomForestClassifier(random_state=42))
])
'''
#X_train['all'] = X_train.sum(axis=1)
#X_train['ph_type'] = X_train['ph'].apply(func=determine_ph)
'''

#fine-tine
param_grid = (
    {'classifier__max_features': [5] , 'classifier__n_estimators':[300] , 'classifier__max_depth':[ 20]}
    )

grid_sh = GridSearchCV(estimator= finally_pip , cv=3 , param_grid=param_grid , scoring='accuracy')
grid_sh.fit(X_train , Y_train)
print(f'best_score_on_train_set{grid_sh.best_score_}')
y_pred = grid_sh.best_estimator_.predict(X_test)
print(f'accuracy: {accuracy_score(Y_test , y_pred)}')
print(f'f1: {f1_score(Y_test , y_pred , average="weighted")}')
print(f'recall: {recall_score(Y_test , y_pred , average="weighted")}')
print(f'precision: {precision_score(Y_test , y_pred , average=" weighted")}')



# 94 with out stander
# after stander 
# 0.9687512718688769 with all_column
# 0.9670476943561 without all_column 
# after append all_column
# 0.9715915367664586 with ph_type_column