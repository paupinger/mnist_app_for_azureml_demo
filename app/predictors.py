import pandas as pd
import requests

def predict_by_endpoint(x_predict):
    x_predict = x_predict.reshape(1, 28, 28, 1)

    df = pd.DataFrame([x_predict.flatten()])   

    data = df.to_json(orient="split")

    # add endpoint URL here
    url = ""
    headers = {'Content-Type':'application/json'}

    resp = requests.post(url, data, headers=headers)

    values = resp.json()["predict_proba"][0]

    return values
