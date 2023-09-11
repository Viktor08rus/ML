# <YOUR_IMPORTS>
import dill
import os
import pandas as pd
import json
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')
#Загрузка обученной модели
def load_model():
    files = os.listdir(f'{path}/data/models')
    with open(f'{path}/data/models/{files[0]}', 'rb') as file:
        model = dill.load(file)
    return model

#Предсказания для объектов в папке data/test
def predict():
    directory =f'{path}/data/test/'
    files = os.listdir(directory)

    df_pred = pd.DataFrame(columns = ['car_id', 'pred'])
    model = load_model()

    for filename in files:
        with open(directory + filename) as f:
            form = json.load(f)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            df_pred.loc[len(df_pred)] = [df.id[0], y[0]]

    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index = False)



if __name__ == '__main__':
    predict()
