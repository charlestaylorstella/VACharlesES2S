import torch
import sys

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

def print_matrix(x, filestream=sys.stderr):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    ''' 
    There are some attempts. x.datacpu().numpy() also works.
    print("x size:", x.size()[0])
    print("x:", x)
    print("x data:", x[0].data) # do not change
    print("x[0][0] data0:", x[0][0].data[0]) # do not change
    print("x[0][0].data.cpu():", x[0][0].data.cpu()) # do not change much
    print("x[0][0].data.cpu().numpy():", x[0][0].data.cpu().numpy()) # dealed!
    print("x[0].data.cpu().numpy():", x[0].data.cpu().numpy()) # dealed!
    print("x.data.cpu().numpy():", x.data.cpu().numpy()) # dealed!
    print("x.data.tolist:", x.data.tolist()) # dealed!
    print("x.data.tolist[0]:", x.data.tolist()[0]) # dealed!
    '''
    print("x:", x)
    print("x.data.tolist:", x.data.tolist()) # dealed!
    print("raw:", raw, "col:", col, "len(x_list)", len(x_list), "len(x_list[0]) and 1:", len(x_list[0]), len(x_list[1]))
    for i in range(raw):
        assert len(x_list[i]) == col
        print(" ".join(list(map(str, x_list[i]))), file=filestream)
    print("", file=filestream)

def print_matrix_with_text(x, text, filestream=sys.stderr):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    print("text:", text)
    print("len(text):", len(text))
    for i in range(raw):
        assert len(x_list[i]) == col
        print(text[i] + "\t" + " ".join(list(map(str, x_list[i]))), file=filestream)
    print("", file=filestream)

def print_matrix_with_text(x, ids, filestream=sys.stderr):
    raw = x.size()[0]
    col = x.size()[1]
    x_list = x.data.tolist()
    print("text:", text)
    print("len(text):", len(text))
    for i in range(raw):
        assert len(x_list[i]) == col
        print(text[i] + "\t" + " ".join(list(map(str, x_list[i]))), file=filestream)
    print("", file=filestream)
