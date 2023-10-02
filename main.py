from fastapi import FastAPI
import torch
import uvicorn
from model import AnomalyTransformer, predict
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json

model = AnomalyTransformer(win_size=100, enc_in=50, c_out=50, e_layers=3)
model_path = '/content/drive/MyDrive/Anomaly-Transformer/checkpoints/BGL__checkpoint_tf_50.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

app = FastAPI()
app = FastAPI()

class Item(BaseModel):
  Time: int

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/predict')
def predict_n(item : Item):
  x = item.json()
  x = json.loads(x)
  x = x['Time']
  y = predict(x)
  return y

if __name__=='__main__':
  uvicorn.run(app, host='localhost', port=8000)
