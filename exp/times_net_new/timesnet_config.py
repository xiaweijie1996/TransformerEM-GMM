class TimesBlockConfig:
    # Parameters only related to data and task----------------
    seq_len = 366  # Sequence length, 48 steps 
    pred_len = 0  # Prediction length, predict 48 steps
    c_out = 24 # Output dimension
    enc_in = 24 # Input data dimension (T, N, C) en_in is c channels
    
    # Parameters for can be optimized----------------
    top_k = 10  # Top k frequencies
    d_model = 18  # Hidden dimension
    d_ff = 32 # Hidden dimension
    num_kernels = 10 # number of kernels for inception block
    dropout = 0.1 # Dropout rate
    e_layers = 3 # Number of TimesNet block
    
    # Do not change--------------------------------
    embed = 'fixed' # Embedding type
    freq = 'h' # Do not change Frequency
    task_name = 'imputation' #  task name
    
    
    