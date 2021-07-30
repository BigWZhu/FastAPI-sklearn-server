import numpy as np
import requests
import json

url_train = "http://127.0.0.1:8000/train_PLS"
url_test = "http://127.0.0.1:8000/use_PLS"

# Training the model

x = np.random.rand(100, 5).tolist()
y = np.random.rand(100, 2).tolist()

train_data = {
    "X": x,
    "Y": y
}

response = requests.post(url_train, json.dumps(train_data))
# print(model_para)
model_para = json.loads(response.content.decode('utf-8'))


# Test the model

test_piece = {
        # x test
        "X": np.random.rand(1, 5).tolist(),
        # model parameters
        "model_param": model_para
    }


response1 = requests.post(url_test, json.dumps(test_piece))
y_pred = json.loads(response1.content.decode('utf-8'))['y_pred']

