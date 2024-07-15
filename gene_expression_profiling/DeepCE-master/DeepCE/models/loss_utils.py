import torch
import torch.nn as nn
import numpy as np


class LogCumsumExp(torch.autograd.Function):
    # The PyTorch OP corresponding to the operation: log{ |sum_k^m{ exp{pred_k} } }
    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a context object and a Tensor containing the input;
        we must return a Tensor containing the output, and we can use the context object to cache objects for use in the backward pass.
        Specifically, ctx is a context object that can be used to stash information for backward computation.
        You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
        :param ctx:
        :param input: i.e., batch_preds of [batch, ranking_size], each row represents the relevance predictions for documents within a ranking
        :return: [batch, ranking_size], each row represents the log_cumsum_exp value
        """
        m, _ = torch.max(input, dim=1, keepdim=True)    #a transformation aiming for higher stability when computing softmax() with exp()
        y = input - m
        y = torch.exp(y)
        y_cumsum_t2h = torch.flip(torch.cumsum(torch.flip(y, dims=[1]), dim=1), dims=[1])    #row-wise cumulative sum, from tail to head
        fd_output = torch.log(y_cumsum_t2h) + m # corresponding to the '-m' operation
        ctx.save_for_backward(input, fd_output)
        return fd_output


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and
        a Tensor containing the gradient of the loss with respect to the output produced during the forward pass (i.e., forward's output).
        We can retrieve cached data from the context object, and
        must compute and return the gradient of the loss with respect to the input to the forward function.
        Namely, grad_output is the gradient of the loss w.r.t. forward's output. Here we first compute the gradient (denoted as grad_out_wrt_in) of forward's output w.r.t. forward's input.
        Based on the chain rule, grad_output * grad_out_wrt_in would be the desired output, i.e., the gradient of the loss w.r.t. forward's input
        :param ctx:
        :param grad_output:
        :return:
        """
        input, fd_output = ctx.saved_tensors
        # chain rule
        bk_output = grad_output * (torch.exp(input) * torch.cumsum(torch.exp(-fd_output), dim=1))
        return bk_output


apply_LogCumsumExp = LogCumsumExp.apply


def tor_batch_triu(batch_mats=None, k=0, device=None):
    """
    :param batch_mats: [batch, m, m]
    :param k: the offset w.r.t. the diagonal line For k=0 means including the diagonal line, k=1 means upper triangular part without the diagonal line
    :return:
    """
    assert batch_mats.size(1) == batch_mats.size(2)
    m = batch_mats.size(1)
    row_inds, col_inds = np.triu_indices(m, k=k)
    tor_row_inds = torch.LongTensor(row_inds).to(device)
    tor_col_inds = torch.LongTensor(col_inds).to(device)
    batch_triu = batch_mats[:, tor_row_inds, tor_col_inds]
    return batch_triu


def idcg_std(sorted_labels, device):
    """
    :param sorted_labels:
    :return:
    nums = np.power(2, sorted_labels) - 1.0
    denoms = np.log2(np.arange(len(sorted_labels)) + 2)
    idcgs = np.sum(nums/denoms, axis=1)
    return idcgs
    """
    nums = torch.pow(2.0, sorted_labels) - 1.0
    a_range = torch.arange(sorted_labels.size(1), dtype=torch.double).to(device)
    denoms = torch.log2(2.0 + a_range)
    idcgs = torch.sum(nums / denoms, dim=1)
    return idcgs


def tor_get_approximated_ranks(batch_pred_diffs, alpha, tor_zero):
    batch_indicators = torch.where(batch_pred_diffs < 0, 1.0 / (1.0 + torch.exp(alpha * batch_pred_diffs)), tor_zero)  # w.r.t. negative Sxy
    batch_tmps = torch.exp(torch.mul(batch_pred_diffs, -alpha))
    batch_indicators = torch.where(batch_pred_diffs > 0, torch.div(batch_tmps, batch_tmps + 1.0), batch_indicators)  # w.r.t. positive Sxy
    batch_hat_pis = torch.sum(batch_indicators, dim=2) + 0.5  # get approximated rank positions, i.e., hat_pi(x)
    return batch_hat_pis


class ApproxNDCG_OP(torch.autograd.Function):
    DEFAULT_ALPHA = 50

    @staticmethod
    def forward(ctx, input, batch_std_labels):
        """
        In the forward pass we receive a context object and a Tensor containing the input;
        we must return a Tensor containing the output, and we can use the context object to cache objects for use in the backward pass.
        Specifically, ctx is a context object that can be used to stash information for backward computation.
        You can cache arbitrary objects for use in the backward pass using the ctx.save_for_backward method.
        :param ctx:
        :param input: [batch, ranking_size], each row represents the relevance predictions for documents within a ranking
        :return: [batch, ranking_size], each row value represents the approximated nDCG metric value
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        tor_zero = torch.Tensor([0.0]).to(device).double()
        alpha = ApproxNDCG_OP.DEFAULT_ALPHA
        batch_pred_diffs = torch.unsqueeze(input, dim=2) - torch.unsqueeze(input, dim=1)        #computing pairwise differences, i.e., Sij or Sxy
        # stable version of the above two lines
        batch_hat_pis = tor_get_approximated_ranks(batch_pred_diffs, alpha=alpha, tor_zero=tor_zero)
        # used for later back propagation
        bp_batch_exp_alphaed_diffs = torch.where(batch_pred_diffs<0, torch.exp(alpha*batch_pred_diffs), tor_zero) # negative values
        bp_batch_exp_alphaed_diffs = torch.where(batch_pred_diffs>0, torch.exp(-alpha*batch_pred_diffs), bp_batch_exp_alphaed_diffs) # positive values
        batch_gains = torch.pow(2.0, batch_std_labels) - 1.0
        sorted_labels, _ = torch.sort(batch_std_labels, dim=1, descending=True)                 #for optimal ranking based on standard labels
        batch_idcgs = idcg_std(sorted_labels, device)                                                   # ideal dcg given standard labels
        batch_dcg = torch.sum(torch.div(batch_gains, torch.log2(batch_hat_pis + 1)), dim=1)
        batch_ndcg = torch.div(batch_dcg, batch_idcgs)
        ctx.save_for_backward(batch_hat_pis, batch_pred_diffs, batch_idcgs, batch_gains, bp_batch_exp_alphaed_diffs)
        return batch_ndcg


    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive the context object and
        a Tensor containing the gradient of the loss with respect to the output produced during the forward pass (i.e., forward's output).
        We can retrieve cached data from the context object, and
        must compute and return the gradient of the loss with respect to the input to the forward function.
        Namely, grad_output is the gradient of the loss w.r.t. forward's output. Here we first compute the gradient (denoted as grad_out_wrt_in) of forward's output w.r.t. forward's input.
        Based on the chain rule, grad_output * grad_out_wrt_in would be the desired output, i.e., the gradient of the loss w.r.t. forward's input
        i: the i-th rank position
        Si: the relevance prediction w.r.t. the document at the i-th rank position
        Sj: the relevance prediction w.r.t. the document at the j-th rank position
        Sij: the difference between Si and Sj
        :param ctx:
        :param grad_output:
        :return:
        """
        alpha = ApproxNDCG_OP.DEFAULT_ALPHA
        batch_hat_pis, batch_pred_diffs, batch_idcgs, batch_gains, bp_batch_exp_alphaed_diffs = ctx.saved_tensors
        # the coefficient, which includes ln2, alpha, gain value, (1+hat_pi), pow((log_2_{1+hat_pi} ), 2)
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        log_base = torch.tensor([2.0]).to(device).double()
        batch_coeff = (alpha/torch.log(log_base))*(batch_gains/((batch_hat_pis + 1.0) * torch.pow(torch.log2(batch_hat_pis + 1.0), 2.0)))  #coefficient part
        #here there is no difference between 'minus-alpha' and 'alpha'
        batch_gradient_Sijs = torch.div(bp_batch_exp_alphaed_diffs, torch.pow((1.0 + bp_batch_exp_alphaed_diffs), 2.0)) # gradients w.r.t. Sij, i.e., main part of delta(hat_pi(d_i))/delta(s_i)
        batch_weighted_sum_gts_i2js = batch_coeff * torch.sum(batch_gradient_Sijs, dim=2)   #sum_{i}_{delta(hat_pi(d_i))/delta(s_j)}
        batch_weighted_sum_gts_js2i = torch.squeeze(torch.bmm(torch.unsqueeze(batch_coeff, dim=1), batch_gradient_Sijs), dim=1) #sum_{j}_{delta(hat_pi(d_j))/delta(s_i)}
        batch_gradient2Sis = torch.div((batch_weighted_sum_gts_i2js - batch_weighted_sum_gts_js2i), torch.unsqueeze(batch_idcgs, dim=1))    #normalization coefficent
        #chain rule
        grad_output.unsqueeze_(1)
        target_gradients = grad_output * batch_gradient2Sis
        #target_gradients.unsqueeze_(2)
        # it is a must that keeping the same number w.r.t. the input of forward function
        return target_gradients, None


apply_ApproxNDCG_OP = ApproxNDCG_OP.apply
mse = nn.MSELoss()
ce = nn.CrossEntropyLoss()
cos = nn.CosineSimilarity(dim=1)
