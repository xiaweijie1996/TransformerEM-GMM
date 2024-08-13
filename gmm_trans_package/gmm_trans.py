from typing import Union

import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from huggingface_hub import hf_hub_download
    
import gmm_trans_package.model.gmm_transformer as gmm_model
import gmm_trans_package.asset.dataloader as dl
import gmm_trans_package.asset.em_pytorch as ep
import gmm_trans_package.asset.random_sampler as rs
import gmm_trans_package.asset.gmm_train_tool as gtt
import gmm_trans_package.asset.plot_eva as pe

def load_model(
    n_components = 6,
    hidden_d = 24 * 4,
    out_d = 24,
    n_heads = 4,
    mlp_ratio = 8,
    n_blocks = 6,
    _encoder = r'_encoder_25_4537398.pth',
    _para = r'_embedding_25_4537398.pth',
    _token = r'_emb_empty_token_25_4537398.pth',
    random_sample_num = None
    ):

    chw = (1, random_sample_num,  25)
    
    # Set device to GPU if available, otherwise use CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the transformer model
    encoder = gmm_model.ViT_encodernopara(chw, hidden_d, out_d, n_heads, mlp_ratio, n_blocks).to(device)
    _model_scale = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('Number of parameters of encoder:', _model_scale)

    # Download the pre-trained model state from huggingface hub
    encoder_path = hf_hub_download(repo_id="Weijie1996/gmms_transformer_models", filename=_encoder)
    path_para = hf_hub_download(repo_id="Weijie1996/gmms_transformer_models", filename=_para)
    path_token = hf_hub_download(repo_id="Weijie1996/gmms_transformer_models", filename=_token)
    
    # Load the pre-trained model state
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))

    state_dict_para = torch.load(path_para, map_location=device)
    state_dict_token = torch.load(path_token, map_location=device)
    
    return encoder, state_dict_para, state_dict_token

def load_valdata_example():
    val_data_path = hf_hub_download(repo_id="Weijie1996/gmms_transformer_models", filename='val_data_example.pkl')
    batch_size = 64
    split_ratio = (0, 0, 1)
    dataloader = dl.Dataloader_nolabel(val_data_path, batch_size, split_ratio)
    return dataloader

class GMMsTransPipeline:
    def from_pretrained(self,
        n_components: int = 6,
        resolution: int = 24):
        
        # check the model type
        if resolution == 24:
            model, para_emb, token_emb = load_model(
                n_components = n_components,
                hidden_d = 24 * 4,
                out_d = 24,
                n_heads = 4,
                mlp_ratio = 8,
                n_blocks = 6,
                _encoder = r'_encoder_25_4537398.pth',
                _para = r'_embedding_25_4537398.pth',
                _token = r'_emb_empty_token_25_4537398.pth',
                random_sample_num = None
                )
        elif resolution == 48:
            pass
            
        return model, para_emb, token_emb 
    
    def _pre_parameter(self, encoder, para_emb, token_emb, _val_data, sample_num, n_components=6, device='cpu'):

        encoder.eval()
        # _val_data = dataloader.load_vali_data(size=1)
        _val_data = torch.tensor(_val_data, dtype=torch.float64).to(device)
        
        # normalize the input data
        _val_min,_ = _val_data[:,:, :-1].min(axis=1, keepdim=True)
        _val_max,_ = _val_data[:,:, :-1].max(axis=1, keepdim=True)
        _val_data[:,:, :-1] = (_val_data[:,:, :-1] - _val_min)/(_val_max-_val_min+1e-15)
        
        _val_sample_part = rs.random_sample(_val_data, 'random', sample_num)
        _val_sample_part[:, :, -1] = _val_sample_part[:, :, -1]/365 # simple date embedding
        
        # padding the empty token to _train_sample_part to have shape (b, random_sample_num, 25)
        _val_sample_part_emb = gtt.pad_and_embed(_val_sample_part, 25, sample_num, token_emb, device)
        
        # use ep to do one iteration of the EM algorithm
        _ms, _covs = ep.GMM_PyTorch_Batch(n_components, 24).fit(_val_sample_part[:,:, :-1], 1)
        
        # concatenate the mean and variance to have (b, n_components*2, 25)
        _param_emb, _param = gtt.concatenate_and_embed_params(_ms, _covs, n_components, para_emb, device)
        
        # feed into the encoder
        _val_sample_part_emb = torch.cat((_param_emb, _val_sample_part_emb), dim=1)
        encoder_out = encoder(_val_sample_part_emb)
        
        _new_para = encoder_out[:, :n_components*2, :]
        _new_para = encoder.output_adding_layer(_new_para, _param)
            
        return _new_para, _param, _val_data[:, :, :-1], _val_sample_part[:, :, :-1], (_val_min, _val_max)

    def _sample_and_scale(
        self,
        _new_para = None,
        _mm = None,
        _val_data = None,
        _val_sample_part = None,
        sample_amount = 250,
        n_components = 6,
        ):

        _min = _mm[0].cpu().detach().numpy()
        _max = _mm[1].cpu().detach().numpy()
        _samples, _gmm = pe.sample_from_gmm(n_components=n_components , _new_para=_new_para, _num=0, _num_samples=sample_amount)
        t_samples = _samples * (_max - _min) + _min
        t_samples[t_samples < 0] = 0
        r_samples_scale_back =_val_data * (_max - _min) + _min
        r_samples_part_scale_back = _val_sample_part * (_max - _min) + _min
        return t_samples[0], r_samples_scale_back[0], r_samples_part_scale_back[0]
    
    def inference(self, encoder, para_emb, token_emb, _val_data, sample_num):
        # use pre-trained model to do inference
        _new_para, _, _val_data, _val_sample_part, _mm = self._pre_parameter(encoder, para_emb, token_emb, _val_data, sample_num)
        
        # sample and scale back
        t_samples, _, r_samples_part_scale_back = self._sample_and_scale(_new_para, _mm, _val_data, _val_sample_part)
        return _new_para, t_samples, r_samples_part_scale_back


if __name__ == '__main__':
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
    