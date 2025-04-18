import torch


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
