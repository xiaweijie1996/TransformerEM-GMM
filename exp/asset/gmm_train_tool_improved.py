import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import asset.em_pytorch_improved as ep
import asset.random_sampler as rs
import asset.le as le

def pad_and_embed(train_sample_part, random_sample_num, random_num, emb_empty_token, device):
    # create an index tensor for the empty token
    embedded_token_idx = torch.tensor([0], dtype=torch.long).to(device)
    
    # get the embedded token
    embedded_token = emb_empty_token(embedded_token_idx)
    
    # if no padding is needed, return the original part
    if random_sample_num - random_num == 0:
        return train_sample_part
    
    # repeat the embedded token to match the required padding shape
    embedded_token = embedded_token.repeat(train_sample_part.shape[0], random_sample_num - random_num, 1)
    
    # concatenate the original samples with the padding
    train_sample_part_emb = torch.cat((train_sample_part, embedded_token), dim=1)
    
    return train_sample_part_emb

def concatenate_and_embed_params(ms, covs, n_components, embedding_layer, device):
    # concatenate the mean and variance to have (b, n_components*2, 25)
    param = torch.cat((ms, covs), dim=1)
    
    # create indices for embedding and move to the appropriate device
    embed_indices = torch.tensor([list(range(n_components * 2))], dtype=torch.long).to(device)
    
    # get the embeddings
    embed = embedding_layer(embed_indices)
    
    # repeat the embedding to match the batch size
    embed = embed.repeat(param.shape[0], 1, 1)
    
    # concatenate the parameters and the embeddings
    param_emb = torch.cat((param, embed), dim=2)
    
    return param_emb,  param

def get_loss_le(dataset, encoder, random_sample_num, min_random_sample_num, n_components, embedding, emb_empty_token, train='True', device='cpu'):
    if train == 'True':
        train_sample = dataset.load_train_data()
    else:
        train_sample = dataset.load_test_data()
  
    train_sample = torch.tensor(train_sample, dtype=torch.float64).to(device)
    
    # normalize the input data
    _train_min,_ = train_sample[:,:, :-1].min(axis=1, keepdim=True)
    _train_max,_ = train_sample[:,:, :-1].max(axis=1, keepdim=True)
    train_sample[:,:, :-1] = (train_sample[:,:, :-1] - _train_min)/(_train_max-_train_min+1e-15)
    
    # random_sample a number between min_random_sample_num and random_sample_num
    _random_num = torch.randint(min_random_sample_num, random_sample_num+1, (1,)).item()
    _train_sample_part = rs.random_sample(train_sample, 'random', _random_num)
    _train_sample_part[:, :, -1] = _train_sample_part[:, :, -1]/365 # simple data embedding
    
    # padding the empty token to _train_sample_part to have shape (b, random_sample_num, 25)
    _train_sample_part_emb = pad_and_embed(_train_sample_part, random_sample_num, _random_num,
                                           emb_empty_token, device)

    # use ep to do one iteration of the EM algorithm
    _ms, _covs = ep.GMM_PyTorch_Batch(n_components, _train_sample_part[:,:, :-1].shape[-1]).fit(_train_sample_part[:,:, :-1], 1) # _ms: (b, n_components, 24), _covs: (b, n_components, 24)

    # concatenate the mean and variance to have (b, n_components*2, 25)
    _param_emb, _param = concatenate_and_embed_params(_ms, _covs, n_components, embedding, device)
    
    # feed into the encoder
    _train_sample_part_emb = torch.cat((_param_emb, _train_sample_part_emb), dim=1)
    encoder_out = encoder(_train_sample_part_emb)
    
    _new_para = encoder_out[:, :n_components*2, :]
    _new_para = encoder.output_adding_layer(_new_para, _param)
    _loss = le.le_loss(train_sample[:,:, :-1], n_components, _new_para)
    
    return _loss, _random_num, _new_para, _param, train_sample[:, :, :-1], _train_sample_part[:, :, :-1], (_train_min, _train_max) 

