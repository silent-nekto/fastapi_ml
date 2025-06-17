from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI()


class Item(BaseModel):
    sep_length: float
    sep_width: float
    pet_length: float
    pet_width: float


class IrisWrapper:
    def __init__(self):
        data = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2)

        self.model = RandomForestClassifier()
        self.model.fit(X_train, y_train)

    def predict(self, item: Item):
        predictions = self.model.predict([[item.sep_width, item.sep_length, item.pet_width, item.pet_length]])
        return predictions[0]


IrisModel = IrisWrapper()


@app.get('/items/')
async def get_items(s: Item):
    print(f'===={s=}')
    res = IrisModel.predict(s)
    return f'Result is {res}'
