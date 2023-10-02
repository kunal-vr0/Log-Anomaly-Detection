from model import AnomalyTransformer
from model import predict_anomaly

model = AnomalyTransformer(win_size=100, enc_in=50, c_out=50, e_layers=3)

if torch.cuda.is_available():
  model.cuda()

model_path = '/content/drive/MyDrive/Anomaly-Transformer/checkpoints/BGL__checkpoint_tf_50.pth'
model.load_state_dict(torch.load(model_path))
model.eval()
a, p, r, f1 = predict_anomaly()