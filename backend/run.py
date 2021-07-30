from fastapi import FastAPI
import uvicorn
from sklearn.cross_decomposition import PLSRegression
from pydantic import BaseModel, Field
from typing import List
import numpy as np
import json
from typing import ClassVar


app = FastAPI()

# model parameters in sklearn PLS model
all_attr = ['x_weights_','y_weights_','x_loadings_','y_loadings_',
            'x_scores_','y_scores_','x_rotations_','y_rotations_','coef_',
            'n_iter_','n_features_in_','_x_mean','_x_std','_y_mean', '_y_std']

def serialize_model(pls_model):
# save model parameters into dict format
    param = {}
    
    for attr in all_attr:
        param_att = getattr(pls_model, attr)
        if type(param_att) is int:
            param[attr] = [param_att]
        elif type(param_att) is list:
            param[attr] = param_att
        else:
            param[attr] = param_att.tolist()

    return param

def load_model(param):
# create the PLS model from given parameters
    n_components = np.shape(param['x_weights_'])[1]
    model = PLSRegression(n_components=n_components, scale=True) 
    for attr in all_attr:
        param_att = param[attr]

        try:
            setattr(model, attr, np.array(param_att))
        except AttributeError:
            pass
        # getattr(model, attr) = param_att
        
    return model
        

'''
Data structure schema
'''
    
class XSchema(BaseModel):
    X: List[List[float]] = Field(None, title="X Input")

class YSchema(BaseModel):
    Y: List[List[float]] = Field(None, title="Y Input")

class XYSchema(XSchema, YSchema):
    pass

class PlsParamSchema(BaseModel):
    x_weights_: List[List[float]]
    y_weights_: List[List[float]]
    x_loadings_: List[List[float]]
    y_loadings_: List[List[float]]
    x_scores_: List[List[float]]
    y_scores_: List[List[float]]
    x_rotations_: List[List[float]]
    y_rotations_: List[List[float]]
    coef_: List[List[float]]
    n_iter_: List[float]
    n_features_in_: List[float]

    x_mean:List[float]= Field(alias="_x_mean")
    x_std: List[float]=Field(alias="_x_std")
    y_mean: List[float]=Field(alias="_y_mean")
    y_std: List[float]=Field(alias="_y_std")
    

class TestSchema(XSchema):
    model_param: PlsParamSchema


'''
API urls
'''

@app.post('/train_PLS', response_model=PlsParamSchema)
def train_PLS(data: XYSchema):
    x_data = np.array(data.X)
    y_data = np.array(data.Y)
    
    assert np.shape(x_data)[0] == np.shape(y_data)[0], "dimension of X and Y should equal"
        
    pls_model = PLSRegression(n_components=2)
    pls_model.fit(x_data, y_data)
    j_dict = serialize_model(pls_model)
    
    return j_dict

@app.post('/use_PLS')
def use_PLS(model_input: TestSchema):
    n_samples = np.shape(model_input.X)[0]
    
    x_dim = np.shape(model_input.X)[1]
    
    param_dim = np.shape(model_input.model_param.x_weights_)[0]
    
    assert x_dim == param_dim, "dimension is incorrect: input dimension %d not equal to %d"%(x_dim, param_dim)
    
    model = load_model(model_input.model_param.dict(by_alias=True))
    y_pred = model.predict(np.reshape(model_input.X,(n_samples,x_dim)))
    
    return {'y_pred': y_pred.tolist()}



if __name__ == '__main__':

    # url 127.0.0.1:8000
    # API docs: http://127.0.0.1:8000/docs
    uvicorn.run(app='run:app', host="0.0.0.0", port=8000)


