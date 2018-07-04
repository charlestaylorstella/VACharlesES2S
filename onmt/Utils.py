import torch


def aeq(*args):
    """
    Assert all arguments have the same value
    """
    arguments = (arg for arg in args)
    first = next(arguments)
    assert all(arg == first for arg in arguments), \
        "Not all arguments have the same value: " + str(args)


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


def use_gpu(opt):
    return (hasattr(opt, 'gpuid') and len(opt.gpuid) > 0) or \
        (hasattr(opt, 'gpu') and opt.gpu > -1)


def myMatrixDivVector(matrix, vector):
    """
       matrix(N,M) / vector(N) = matrix(N,M)
       for each i,j: 
           matrix_result[i][j] = matrix_source[i][j] / vector[i]
    """
    duplicate_size = matrix.size()[-1]
    vector_duplicate = vector.repeat(duplicate_size, 1).permute(1, 0)
    matrix = matrix / vector_duplicate
    return matrix

def sum_with_axis(input, axes, keepdim=False):
    # probably some check for uniqueness of axes
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input
