from gmm_trans_package import plot_eva as pe
from gmm_trans_package import GMMsTransPipeline, load_valdata_example
import torch

pipeline = GMMsTransPipeline()
encoder, para_emb, token_emb = pipeline.from_pretrained()

# -----------------------------Inference use real ECP data--------------------------------
# load the validation data for inference
num_of_shot = 5
dataloader = load_valdata_example()
_val_data = dataloader.load_vali_data(size=1)
gmm_parameters, t_samples, _ = pipeline.inference(encoder, para_emb, token_emb, _val_data, num_of_shot) # 2 is the number of samples
pe.plot_results(t_samples, _val_data[0][:,:-1], _)
# -----------------------------Inference use real ECP data--------------------------------


# -----------------------------Inference use toy data--------------------------------
# we want to test the inference result if we randomly creaste a toy data    
# feel free to adjust the window_size, num_of_shot to test the Zero-shot time series modeling
num_of_shot = 5
window_size = 15

_x = torch.randn(250, 25)
_y = torch.sin(_x)+1 # the model's output range is [0, +inf], need to scale the data to this range

# a smoothing function to smooth the data, otherwise the y data is random noise
def smooth(y, window_size=window_size):
    conv_filter = torch.ones(window_size) / window_size
    smooth_y = torch.nn.functional.conv1d(y.unsqueeze(1), conv_filter.unsqueeze(0).unsqueeze(0), padding=window_size//2)
    return torch.exp(smooth_y.squeeze(1)) 

_y_smooth = smooth(_y.T).T
toy_samples = _y_smooth.unsqueeze(0)

toy_samples = torch.tensor(toy_samples[0], dtype=torch.float64).unsqueeze(0)
gmm_parameters_toy, t_samples_toy, _toy = pipeline.inference(encoder, para_emb, token_emb, toy_samples, num_of_shot) # 2 is the number of samples

# plot the toy samples
pe.plot_results(t_samples_toy, toy_samples[0][:,:-1], _toy)
# -----------------------------Inference use toy data--------------------------------
