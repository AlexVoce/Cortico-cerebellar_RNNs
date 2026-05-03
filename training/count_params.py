def count_rnn_params(input_size, hidden_size, bias=True):
    """
    Elman RNN core:
      input -> hidden
      hidden -> hidden
    """
    b = 1 if bias else 0

    inp = input_size * hidden_size + b * hidden_size
    rec = hidden_size * hidden_size + b * hidden_size

    return inp + rec


def count_cb_params(hidden_size, gc_dim, bias=True):
    """
    Cerebellar bias module:
      GC: hidden -> gc_dim
      PC: gc_dim -> hidden
    """
    b = 1 if bias else 0

    gc = hidden_size * gc_dim + b * gc_dim
    pc = gc_dim * hidden_size + b * hidden_size
    dcn = hidden_size * hidden_size + b * hidden_size

    return gc + pc + dcn


def count_readout_params(hidden_size, num_classes, num_heads, bias=True):
    """
    Multi-head linear readout
    """
    b = 1 if bias else 0
    per_head = hidden_size * num_classes + b * num_classes
    return num_heads * per_head


def count_total_params(
    input_size=1,
    hidden_size=64,
    gc_dim=512,
    num_classes=2,
    num_heads=1,
    rnn_bias=True,
    cb_bias=True,
    readout_bias=True,
):
    rnn = count_rnn_params(input_size, hidden_size, rnn_bias)
    cb = count_cb_params(hidden_size, gc_dim, cb_bias)
    ro = count_readout_params(hidden_size, num_classes, num_heads, readout_bias)

    return {
        "RNN": rnn,
        "CB": cb,
        "Readout": ro,
        "Total": rnn + cb + ro,
    }

def rnn_only_params(input_size, hidden_size, num_classes, num_heads, bias=True):
    return (
        count_rnn_params(input_size, hidden_size, bias)
        + count_readout_params(hidden_size, num_classes, num_heads, bias)
    )


def find_matching_hidden_size(
    target_params,
    input_size=1,
    num_classes=2,
    num_heads=1,
    bias=True,
    h_max=2000,
):
    """Find hidden size that gives closest match to target params"""
    best_h = None
    best_diff = float('inf')
    
    for h in range(1, h_max):
        params = rnn_only_params(input_size, h, num_classes, num_heads, bias)
        diff = abs(params - target_params)
        
        if diff < best_diff:
            best_diff = diff
            best_h = h
        
        # Early exit if we've passed target and diff is increasing
        if params > target_params and diff > best_diff:
            break
    
    return best_h