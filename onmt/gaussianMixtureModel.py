import math
import numpy

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from torch.nn.modules import Module

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


class gaussianMixtureModel(Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.
    Attributes:
        weight: the learnable weights of the module of shape
            `(out_features x in_features)`
        bias:   the learnable bias of the module of shape `(out_features)`
    Examples::
        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, latent_dim, cluster_num, batch_size, opt=None, bias=False):
        super(gaussianMixtureModel, self).__init__()
        self.latent_dim = latent_dim
        self.cluster_num = cluster_num
        self.batch_size = batch_size
        if opt == None:
            self.opt = None 
        self.opt = opt
        self.is_first_ff = True
        #self.input_data_dim = input_data_dim
        #self.alpha = alpha
        #self.target_dict_size = target_dict_size
        #weight4loss = torch.ones(target_dict_size) 
        #self.cross_entropy_loss = nn.NLLLoss(weight, size_average=False)
        self.cluster_mean = Parameter(torch.Tensor(latent_dim, cluster_num))
        self.cluster_variance_sq_unnorm = Parameter(torch.Tensor(latent_dim, cluster_num))
        self.cluster_prior = Parameter(torch.Tensor(cluster_num))
        if bias:
            self.cluster_bias = Parameter(torch.Tensor(cluster_num))
        else:
            self.register_parameter('cluster_bias', None)
        self.reset_parameters()
        print("init cluster_prior:", self.cluster_prior)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.cluster_num)
        torch.nn.init.constant_(self.cluster_mean, 0)
        torch.nn.init.constant_(self.cluster_variance_sq_unnorm, 1.0)
        torch.nn.init.constant_(self.cluster_prior, 1.0 / self.cluster_num)
        print("in reset_parameters init cluster_prior:", self.cluster_prior)
        #self.cluster_mean.data.constant_(0)
        #self.cluster_mean.data.uniform_(-stdv, stdv)
        #self.cluster_variance_sq.data.constant_(1.0)
        #self.cluster_prior.data.uniform_(-stdv, stdv)
        #self.cluster_prior.data.constant_(1.0 / self.cluster_num)
        if self.cluster_bias is not None:
            self.cluster_bias.data.uniform_(-stdv, stdv)

    def forward(self, z_mean, z_log_variance_sq, z):
        if self.is_first_ff:
            torch.nn.init.constant_(self.cluster_mean, 0)
            torch.nn.init.constant_(self.cluster_variance_sq_unnorm, 1.0)
            torch.nn.init.constant_(self.cluster_prior, 1.0 / self.cluster_num)
            self.is_first_ff = False
        if self.opt != None and self.opt.debug_mode >= 4:
            print("ff cluster_prior:", self.cluster_prior)
            print("ff cluster_mean:", self.cluster_mean)
            print("ff cluster_variance_sq_unnorm:", self.cluster_variance_sq_unnorm)
        #print("z_mean:", z_mean.size(), "z_log_variance_sq:", z_log_variance_sq.size(), "z:", z.size())
        # shape
        self.batch_size = z_mean.size()[0]
        cluster_mean_duplicate = self.cluster_mean.repeat(self.batch_size, 1, 1)
        cluster_prior_prob = torch.nn.functional.sigmoid(self.cluster_prior)
        cluster_variance_sq = torch.nn.functional.sigmoid(self.cluster_variance_sq_unnorm)
        #cluster_variance_sq = torch.nn.functional.relu(self.cluster_variance_sq_unnorm) # soft relu is the best
        cluster_variance_sq_duplicate = cluster_variance_sq.repeat(self.batch_size, 1, 1)
        cluster_prior_duplicate = cluster_prior_prob.repeat(self.latent_dim, 1).repeat(self.batch_size, 1, 1)
        cluster_prior_duplicate_2D = cluster_prior_prob.repeat(self.batch_size, 1)
        
        z_mean_duplicate = z_mean.repeat(self.cluster_num, 1, 1).permute(1, 2, 0)
        z_log_variance_sq_duplicate = z_log_variance_sq.repeat(self.cluster_num, 1, 1).permute(1, 2, 0)
        z_duplicate = z.repeat(self.cluster_num, 1, 1).permute(1, 2, 0)
        # prob
        #print("z_duplicate:", z_duplicate)
        #print("cluster_mean_duplicate:", cluster_mean_duplicate)
        if self.opt != None and self.opt.debug_mode >= 3:
            print("z_duplicate size:", z_duplicate.size())
            print("z_mean_duplicate size:", z_mean_duplicate.size())
            print("z_log_variance_sq_duplicate size:", z_log_variance_sq_duplicate.size())

            print("cluster_mean_duplicate size:", cluster_mean_duplicate.size())
            print("cluster_variance_sq_duplicate size:", cluster_variance_sq_duplicate.size())
            print("cluster_prior_duplicate size:", cluster_prior_duplicate.size())
            print("cluster_prior_duplicate_2D size:", cluster_prior_duplicate_2D.size())
        if self.opt != None and self.opt.debug_mode >= 4:
            print("z_duplicate:", z_duplicate)
            print("z_mean_duplicate:", z_mean_duplicate)
            print("z_log_variance_sq_duplicate:", z_log_variance_sq_duplicate)

            print("cluster_mean_duplicate:", cluster_mean_duplicate)
            print("cluster_variance_sq_duplicate:", cluster_variance_sq_duplicate)
            print("cluster_prior:", self.cluster_prior)
            print("cluster_prior_prob:", cluster_prior_prob)
            print("cluster_prior_duplicate:", cluster_prior_duplicate)
            print("cluster_prior_duplicate_2D:", cluster_prior_duplicate_2D)
        #tmpa = cluster_mean_duplicate - z_log_variance_sq_duplicate.cuda()
        tmpa = z_duplicate - cluster_mean_duplicate
        tmpb = tmpa * tmpa
        terms = torch.log(cluster_prior_duplicate) \
            - 0.5 * torch.log(2 * math.pi * cluster_variance_sq_duplicate) \
            - tmpb / (2 * cluster_variance_sq_duplicate)
        P_c_given_x_unnorm = torch.exp(sum_with_axis(terms, [1])) + 1e-10
        #print(P_c_given_x_unnorm)
        #print(sum_with_axis(P_c_given_x_unnorm, [-1]))
        P_c_given_x = myMatrixDivVector(P_c_given_x_unnorm, \
            sum_with_axis(P_c_given_x_unnorm, [-1]))

        # loss
        P_c_given_x_duplicate = P_c_given_x.repeat(self.latent_dim, 1, 1).permute(1, 0, 2)
        #cross_entropy_loss = alpha * self.input_data_dim * self.cross_entropy_loss()
        tmp1 = 0.5 * P_c_given_x_duplicate * (self.latent_dim * math.log(math.pi * 2))
        tmp2 = torch.log(cluster_variance_sq_duplicate)
        tmp3 = torch.exp(z_log_variance_sq_duplicate) / cluster_variance_sq_duplicate
        tmp4 = z_mean_duplicate - cluster_mean_duplicate
        tmp5 = tmp4 * tmp4 / cluster_variance_sq_duplicate
        #tmp111 = tmp1 + tmp2
        #tmp112 = tmp111 + tmp3
        #tmp113 = tmp112 + tmp5
        #second_term = sum_with_axis(tmp113, [1, 2])
        second_term = sum_with_axis(tmp1 + tmp2 + tmp3 + tmp5, [1, 2])
        tmp6 = sum_with_axis(P_c_given_x * torch.log(P_c_given_x), [1])
        tmp7 = sum_with_axis(P_c_given_x * torch.log(cluster_prior_duplicate_2D), [1])
        third_term_KL_div = tmp6 - tmp7
        forth_term = 0.5 * sum_with_axis(z_log_variance_sq + 1, [1]) 
        tmp211 = 0 - second_term + third_term_KL_div
        tmp212 = tmp211 + forth_term
        loss_without_reconstruct = tmp212
        #loss_without_reconstruct = 0 - second_term + third_term_KL_div + forth_term
        nagetive_loss_without_reconstruct = 0 - loss_without_reconstruct
        
        if self.opt != None and self.opt.debug_mode >= 3:
            print("size terms:", terms.size(), "P_c_given_x_unnorm:", P_c_given_x_unnorm.size(), "P_c_given_x", P_c_given_x.size(), "second_term:", second_term.size(), "third_term_KL_div:", third_term_KL_div.size(), "forth_term:", forth_term.size(), "nagetive_loss_without_reconstruct:", nagetive_loss_without_reconstruct.size())
            print("tmp1:", tmp1.size(), "tmp2:", tmp2.size(), "tmp3:", tmp3.size(), "tmp4:", tmp4.size(), "tmp5:", tmp5.size(), "tmp6:", tmp6.size(), "tmp7:", tmp7.size(), "second_term:", second_term.size(), "third_term_KL_div:", third_term_KL_div.size(), "z_log_variance_sq:", z_log_variance_sq.size())
            print("tmp211:", tmp211.size(), "forth_term:", forth_term.size())
        if self.opt != None and self.opt.debug_mode >= 5:
            print("sum_with_axis(terms, [1]):", sum_with_axis(terms, [1]))
        if self.opt != None and self.opt.debug_mode >= 4:
            print("terms:", terms, "P_c_given_x_unnorm:", P_c_given_x_unnorm, "P_c_given_x", P_c_given_x, "second_term:", second_term, "third_term_KL_div:", third_term_KL_div, "forth_term:", forth_term, "nagetive_loss_without_reconstruct:", nagetive_loss_without_reconstruct)
            print("tmp1:", tmp1, "tmp2:", tmp2, "tmp3:", tmp3, "tmp4:", tmp4, "tmp5:", tmp5, "tmp6:", tmp6, "tmp7:", tmp7, "second_term:", second_term, "third_term_KL_div:", third_term_KL_div, "z_log_variance_sq:", z_log_variance_sq)
            print("tmp211:", tmp211, "forth_term:", forth_term)
        return P_c_given_x, nagetive_loss_without_reconstruct

    '''def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )'''

latent_dim = 2
cluster_num = 3
batch_size = 4
opt = {"debug_mode":4}
gmm = gaussianMixtureModel(latent_dim, cluster_num, batch_size)

z_mean = torch.rand(batch_size, latent_dim)
z_log_variance_sq = torch.rand(batch_size, latent_dim)
z = torch.rand(batch_size, latent_dim)
print("z_mean:", z_mean)
print("z_log_variance_sq:", z_log_variance_sq)
print("z:", z)
print("will gmm")
p, loss = gmm(z_mean, z_log_variance_sq, z)
print("P:", p)
print("Loss:", loss)

# define network
#fc = nn.Linear(3, 4)

# call network
#input = torch.rand(3)
#output = fc(input)
#print("input of FC(3*4): ", input)
#print("output of FC(3*4): ", output)
#print("output of FC(3*4): ", output.size())
