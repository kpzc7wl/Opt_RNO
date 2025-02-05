#!/usr/bin/env python  
#-*- coding:utf-8 _*-
import os
import operator
import matplotlib.pyplot as plt
import torch
import numpy as np
import torch.special as ts
import dgl

from scipy import interpolate
from functools import reduce




def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all({s})
    '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("=" * 50)
        print(message)
        print("=" * 50)




def get_num_params(model):
    '''
    a single entry in cfloat and cdouble count as two parameters
    see https://github.com/pytorch/pytorch/issues/57518
    '''
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # num_params = 0
    # for p in model_parameters:
    #     # num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
    #     num_params += p.numel() * (1 + p.is_complex())
    # return num_params

    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))  #### there is complex weight
    return c



### x: list of tensors
class MultipleTensors():
    def __init__(self, x):
        self.x = x

    def to(self, device):
        self.x = [x_.to(device) for x_ in self.x]
        return self

    def __len__(self):
        return len(self.x)

    def append(self, y):
        x = [x_ for x_ in self.x] + [y]
        return x

    def float(self):
        self.x = [x_.float() for x_ in self.x]
        return self

    def __getitem__(self, item):
        return self.x[item]


# whether need to transpose
def plot_heatmap(
    x, y, z, path=None, vmin=None, vmax=None,cmap=None,
    title="", xlabel="x", ylabel="y",show=False
):
    '''
    Plot heat map for a 3-dimension data
    '''
    plt.cla()
    # plt.figure()
    xx = np.linspace(np.min(x), np.max(x))
    yy = np.linspace(np.min(y), np.max(y))
    xx, yy = np.meshgrid(xx, yy)

    vals = interpolate.griddata(np.array([x, y]).T, np.array(z),
        (xx, yy), method='cubic')
    vals_0 = interpolate.griddata(np.array([x, y]).T, np.array(z),
        (xx, yy), method='nearest')
    vals[np.isnan(vals)] = vals_0[np.isnan(vals)]

    if vmin is not None and vmax is not None:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic",cmap=cmap,
                vmin=vmin, vmax=vmax,origin='lower')
    elif vmin is not None:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],
                aspect="equal", interpolation="bicubic",cmap=cmap,
                vmin=vmin,origin='lower')
    else:
        fig = plt.imshow(vals,
                extent=[np.min(x), np.max(x),np.min(y), np.max(y)],cmap=cmap,
                aspect="equal", interpolation="bicubic",origin='lower')
    fig.axes.set_autoscale_on(False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.colorbar()
    if path:
        plt.savefig(path)
    if show:
        plt.show()
    plt.close()

import contextlib

class Interp1d(torch.autograd.Function):
    def __call__(self, x, y, xnew, out=None):
        return self.forward(x, y, xnew, out)

    def forward(ctx, x, y, xnew, out=None):
        """
        Linear 1D interpolation on the GPU for Pytorch.
        This function returns interpolated values of a set of 1-D functions at
        the desired query points `xnew`.
        This function is working similarly to Matlab™ or scipy functions with
        the `linear` interpolation mode on, except that it parallelises over
        any number of desired interpolation problems.
        The code will run on GPU if all the tensors provided are on a cuda
        device.
        Parameters
        ----------
        x : (N, ) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values.
        y : (N,) or (D, N) Pytorch Tensor
            A 1-D or 2-D tensor of real values. The length of `y` along its
            last dimension must be the same as that of `x`
        xnew : (P,) or (D, P) Pytorch Tensor
            A 1-D or 2-D tensor of real values. `xnew` can only be 1-D if
            _both_ `x` and `y` are 1-D. Otherwise, its length along the first
            dimension must be the same as that of whichever `x` and `y` is 2-D.
        out : Pytorch Tensor, same shape as `xnew`
            Tensor for the output. If None: allocated automatically.
        """
        # making the vectors at least 2D
        is_flat = {}
        require_grad = {}
        v = {}
        device = []
        eps = torch.finfo(y.dtype).eps
        for name, vec in {'x': x, 'y': y, 'xnew': xnew}.items():
            assert len(vec.shape) <= 2, 'interp1d: all inputs must be '\
                                        'at most 2-D.'
            if len(vec.shape) == 1:
                v[name] = vec[None, :]
            else:
                v[name] = vec
            is_flat[name] = v[name].shape[0] == 1
            require_grad[name] = vec.requires_grad
            device = list(set(device + [str(vec.device)]))
        assert len(device) == 1, 'All parameters must be on the same device.'
        device = device[0]

        # Checking for the dimensions
        assert (v['x'].shape[1] == v['y'].shape[1]
                and (
                     v['x'].shape[0] == v['y'].shape[0]
                     or v['x'].shape[0] == 1
                     or v['y'].shape[0] == 1
                    )
                ), ("x and y must have the same number of columns, and either "
                    "the same number of row or one of them having only one "
                    "row.")

        reshaped_xnew = False
        if ((v['x'].shape[0] == 1) and (v['y'].shape[0] == 1)
           and (v['xnew'].shape[0] > 1)):
            # if there is only one row for both x and y, there is no need to
            # loop over the rows of xnew because they will all have to face the
            # same interpolation problem. We should just stack them together to
            # call interp1d and put them back in place afterwards.
            original_xnew_shape = v['xnew'].shape
            v['xnew'] = v['xnew'].contiguous().view(1, -1)
            reshaped_xnew = True

        # identify the dimensions of output and check if the one provided is ok
        D = max(v['x'].shape[0], v['xnew'].shape[0])
        shape_ynew = (D, v['xnew'].shape[-1])
        if out is not None:
            if out.numel() != shape_ynew[0]*shape_ynew[1]:
                # The output provided is of incorrect shape.
                # Going for a new one
                out = None
            else:
                ynew = out.reshape(shape_ynew)
        if out is None:
            ynew = torch.zeros(*shape_ynew, device=device)

        # moving everything to the desired device in case it was not there
        # already (not handling the case things do not fit entirely, user will
        # do it if required.)
        for name in v:
            v[name] = v[name].to(device)

        # calling searchsorted on the x values.
        ind = ynew.long()

        # expanding xnew to match the number of rows of x in case only one xnew is
        # provided
        if v['xnew'].shape[0] == 1:
            v['xnew'] = v['xnew'].expand(v['x'].shape[0], -1)

        torch.searchsorted(v['x'].contiguous(),
                           v['xnew'].contiguous(), out=ind)

        # the `-1` is because searchsorted looks for the index where the values
        # must be inserted to preserve order. And we want the index of the
        # preceeding value.
        ind -= 1
        # we clamp the index, because the number of intervals is x.shape-1,
        # and the left neighbour should hence be at most number of intervals
        # -1, i.e. number of columns in x -2
        ind = torch.clamp(ind, 0, v['x'].shape[1] - 1 - 1)

        # helper function to select stuff according to the found indices.
        def sel(name):
            if is_flat[name]:
                return v[name].contiguous().view(-1)[ind]
            return torch.gather(v[name], 1, ind)

        # activating gradient storing for everything now
        enable_grad = False
        saved_inputs = []
        for name in ['x', 'y', 'xnew']:
            if require_grad[name]:
                enable_grad = True
                saved_inputs += [v[name]]
            else:
                saved_inputs += [None, ]
        # assuming x are sorted in the dimension 1, computing the slopes for
        # the segments
        is_flat['slopes'] = is_flat['x']
        # now we have found the indices of the neighbors, we start building the
        # output. Hence, we start also activating gradient tracking
        with torch.enable_grad() if enable_grad else contextlib.suppress():
            v['slopes'] = (
                    (v['y'][:, 1:]-v['y'][:, :-1])
                    /
                    (eps + (v['x'][:, 1:]-v['x'][:, :-1]))
                )

            # now build the linear interpolation
            ynew = sel('y') + sel('slopes')*(
                                    v['xnew'] - sel('x'))

            if reshaped_xnew:
                ynew = ynew.view(original_xnew_shape)

        ctx.save_for_backward(ynew, *saved_inputs)
        return ynew

    @staticmethod
    def backward(ctx, grad_out):
        inputs = ctx.saved_tensors[1:]
        gradients = torch.autograd.grad(
                        ctx.saved_tensors[0],
                        [i for i in inputs if i is not None],
                        grad_out, retain_graph=True)
        result = [None, ] * 5
        pos = 0
        for index in range(len(inputs)):
            if inputs[index] is not None:
                result[index] = gradients[pos]
                pos += 1
        return (*result,)


class TorchQuantileTransformer():
    '''
    QuantileTransformer implemented by PyTorch
    '''

    def __init__(
            self,
            output_distribution,
            references_,
            quantiles_,
            device=torch.device('cpu')
    ) -> None:
        self.quantiles_ = torch.Tensor(quantiles_).to(device)
        self.output_distribution = output_distribution
        self._norm_pdf_C = np.sqrt(2 * np.pi)
        self.references_ = torch.Tensor(references_).to(device)
        BOUNDS_THRESHOLD = 1e-7
        self.clip_min = self.norm_ppf(torch.Tensor([BOUNDS_THRESHOLD - np.spacing(1)]))
        self.clip_max = self.norm_ppf(torch.Tensor([1 - (BOUNDS_THRESHOLD - np.spacing(1))]))

    def norm_pdf(self, x):
        return torch.exp(-x ** 2 / 2.0) / self._norm_pdf_C

    @staticmethod
    def norm_cdf(x):
        return ts.ndtr(x)

    @staticmethod
    def norm_ppf(x):
        return ts.ndtri(x)

    def transform_col(self, X_col, quantiles, inverse):
        BOUNDS_THRESHOLD = 1e-7
        output_distribution = self.output_distribution

        if not inverse:
            lower_bound_x = quantiles[0]
            upper_bound_x = quantiles[-1]
            lower_bound_y = 0
            upper_bound_y = 1
        else:
            lower_bound_x = 0
            upper_bound_x = 1
            lower_bound_y = quantiles[0]
            upper_bound_y = quantiles[-1]
            # for inverse transform, match a uniform distribution
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col = self.norm_cdf(X_col)
                # else output distribution is already a uniform distribution

        # find index for lower and higher bounds
        with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
            if output_distribution == "normal":
                lower_bounds_idx = X_col - BOUNDS_THRESHOLD < lower_bound_x
                upper_bounds_idx = X_col + BOUNDS_THRESHOLD > upper_bound_x
            if output_distribution == "uniform":
                lower_bounds_idx = X_col == lower_bound_x
                upper_bounds_idx = X_col == upper_bound_x

        isfinite_mask = ~torch.isnan(X_col)
        X_col_finite = X_col[isfinite_mask]
        torch_interp = Interp1d()
        X_col_out = X_col.clone()
        if not inverse:
            # Interpolate in one direction and in the other and take the
            # mean. This is in case of repeated values in the features
            # and hence repeated quantiles
            #
            # If we don't do this, only one extreme of the duplicated is
            # used (the upper when we do ascending, and the
            # lower for descending). We take the mean of these two
            X_col_out[isfinite_mask] = 0.5 * (
                    torch_interp(quantiles, self.references_, X_col_finite)
                    - torch_interp(-torch.flip(quantiles, [0]), -torch.flip(self.references_, [0]), -X_col_finite)
            )
        else:
            X_col_out[isfinite_mask] = torch_interp(self.references_, quantiles, X_col_finite)

        X_col_out[upper_bounds_idx] = upper_bound_y
        X_col_out[lower_bounds_idx] = lower_bound_y
        # for forward transform, match the output distribution
        if not inverse:
            with np.errstate(invalid="ignore"):  # hide NaN comparison warnings
                if output_distribution == "normal":
                    X_col_out = self.norm_ppf(X_col_out)
                    # find the value to clip the data to avoid mapping to
                    # infinity. Clip such that the inverse transform will be
                    # consistent
                    X_col_out = torch.clip(X_col_out, self.clip_min, self.clip_max)
                # else output distribution is uniform and the ppf is the
                # identity function so we let X_col unchanged

        return X_col_out

    def transform(self, X, inverse=True,component='all'):
        # print(self.quantiles_.shape, X.shape)
        if self.quantiles_.shape[-1]!=X.shape[-1]:            
            comp_x = [i for i in range(self.quantiles_.shape[-1])]                
            comp_ = [i for i in range(X.shape[-1]) if i not in comp_x]

            X_out = torch.zeros_like(X[:, comp_x], requires_grad=False)
            for feature_idx in comp_x:
                X_out[:, feature_idx] = self.transform_col(
                    X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
                )
            return torch.cat([X_out, X[:, comp_]], dim=-1)

        X_out = torch.zeros_like(X, requires_grad=False)
        for feature_idx in range(X.shape[1]):
            X_out[:, feature_idx] = self.transform_col(
                X[:, feature_idx], self.quantiles_[:, feature_idx], inverse
            )
        return X_out

    def to(self,device):
        self.quantiles_ = self.quantiles_.to(device)
        self.references_ = self.references_.to(device)
        return self


'''
    Simple normalization layer
'''
class UnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=True)
        self.std = X.std(dim=0, keepdim=True) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all', component_x=None):
        if component_x!=None or self.mean.shape[-1]!=X.shape[-1]:            
            if component_x==None: 
                component_x = [i for i in range(self.mean.shape[-1])]
                
            comp_ = [i for i in range(X.shape[-1]) if i not in component_x]
            if inverse:                
                orig_shape = X.shape
                X_ = (X[...,component_x] * (self.std[:,component_x] - 1e-8) + self.mean[:,component_x])
                return torch.cat([X_, X[..., comp_]], dim=-1).view(orig_shape)
            else:
                X_ = (X[...,component_x] - self.mean[:,component_x]) / self.std[:,component_x]
                return torch.cat([X_, X[:, comp_]], dim=-1)
            
        if component == 'all':
            if inverse:
                orig_shape = X.shape
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]


"""
    Normalizes input data between -1 and 1 using min-max normalization.
"""
class MinMaxTransformer():
    def __init__(self, X, component='all', log=False):
        self.log = log
        self.component = component
        self.min = X.min(dim=0, keepdim=True)[0]
        self.max = X.max(dim=0, keepdim=True)[0]
        # if X only has a single value, do no transform
        const_idx = (self.min == self.max)
        if const_idx.any():
            self.max[:, const_idx[0]], self.min[:, const_idx[0]] = 1, 0


    def to(self, device):
        self.min = self.min.to(device)
        self.max = self.max.to(device)
        return self
    
        
    def transform(self, X, inverse=True, component='all'):
        if self.min.shape[-1]!=X.shape[-1]:            
            component_x = [i for i in range(self.min.shape[-1])]                
            comp_ = [i for i in range(X.shape[-1]) if i not in component_x]

            if inverse:
                X_ = X[...,component_x]
                if self.log:
                    X_ = torch.exp(X_) - 1
                orig_shape = X.shape
                X_ = (X_ + 1) / 2 * (self.max - self.min) + self.min
                return torch.cat([X_, X[..., comp_]], dim=-1).view(orig_shape)
            else:
                X_ = (X[...,component_x] - self.min) / (self.max - self.min) * 2 - 1
                if self.log:
                    X_ = torch.log(X_ + 1 + 1e-8)
                return torch.cat([X_, X[:, comp_]], dim=-1)

        # print(X)
        if inverse:
            if self.log:
                X = torch.exp(X) - 1
            X = (X + 1) / 2  * (self.max - self.min) + self.min            
        else:
            X = (X - self.min) / (self.max - self.min) * 2 - 1
            if self.log:
                X = torch.log(X + 1 + 1e-8)

                
        return X

"""
    Normalizes input data by substracting min and taking log.
"""
class LogTransformer():
    def __init__(self, X, component='all'):
        self.component = component
        self.min = X.min(dim=0, keepdim=True)[0]


    def to(self, device):
        self.min = self.min.to(device)
        return self
    
        
    def transform(self, X, inverse=True, component='all'):
        if self.min.shape[-1]!=X.shape[-1]:            
            component_x = [i for i in range(self.min.shape[-1])]                
            comp_ = [i for i in range(X.shape[-1]) if i not in component_x]

            if inverse:
                X_ = X[...,component_x]
                X_ = torch.exp(X_) + self.min
                orig_shape = X.shape
                return torch.cat([X_, X[..., comp_]], dim=-1).view(orig_shape)
            else:
                X_ = X[...,component_x]
                X_ = torch.log(X_ - self.min + 1e-8)
                return torch.cat([X_, X[:, comp_]], dim=-1)

        # print(X)
        if inverse:
            X = torch.exp(X) + self.min  
        else:
            X = torch.log(X - self.min + 1e-8)


                
        return X

'''
    Simple pointwise normalization layer, all data must contain the same length, used only for FNO datasets
    X: B, N, C
'''
class PointWiseUnitTransformer():
    def __init__(self, X):
        self.mean = X.mean(dim=0, keepdim=False)
        self.std = X.std(dim=0, keepdim=False) + 1e-8


    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    def transform(self, X, inverse=True,component='all'):
        if component == 'all':
            if inverse:
                orig_shape = X.shape
                X = X.view(-1, self.mean.shape[0],self.mean.shape[1])   ### align shape for flat tensor
                return (X*(self.std - 1e-8) + self.mean).view(orig_shape)
            else:
                return (X-self.mean)/self.std
        else:
            if inverse:
                orig_shape = X.shape
                X = X.view(-1, self.mean.shape[0],self.mean.shape[1])
                return (X*(self.std[:,component] - 1e-8)+ self.mean[:,component]).view(orig_shape)
            else:
                return (X - self.mean[:,component])/self.std[:,component]


'''
    x: B, N (not necessary sorted)
    y: B, N, C (not necessary sorted)
    xnew: B, N (sorted)
'''
def binterp1d(x, y, xnew, eps=1e-9):
    x_, x_indice = torch.sort(x,dim=-1)
    y_ = y[torch.arange(x_.shape[0]).unsqueeze(1),x_indice]

    x_, y_, xnew = x_.contiguous(), y_.contiguous(), xnew.contiguous()

    ind = torch.searchsorted(x_, xnew)
    ind -= 1
    ind = torch.clamp(ind, 0, x_.shape[1] - 1 - 1)
    ind = ind.unsqueeze(-1).repeat([1, 1, y_.shape[-1]])
    x_ = x_.unsqueeze(-1).repeat([1, 1, y_.shape[-1]])

    slopes = ((y_[:, 1:]-y_[:, :-1])/(eps + (x_[:, 1:]-x_[:, :-1])))

    y_sel = torch.gather(y_, 1, ind)
    x_sel = torch.gather(x_,1, ind)
    slopes_sel = torch.gather(slopes, 1, ind)

    ynew =y_sel + slopes_sel * (xnew.unsqueeze(-1) - x_sel)

    return ynew

def plot_ref_query_error(g, g_u, y_interp, y_pred, y_interp_pred=None, save_path=None):
    # Plot the first data in a batch
    g_query = g
    ref, geo_ref, geo_query = g_u

    idx = 0
    # ref, geo_ref, geo_query = ref.squeeze(), geo_ref.squeeze(), geo_query.squeeze()
    ref, geo_ref, geo_query = ref[idx], geo_ref[idx], geo_query[idx]
    gs = dgl.unbatch(g_query)
    xs = [g_.ndata['x'] for g_ in gs]
    ys = [g_.ndata['y'] for g_ in gs]
    x_query, y = xs[idx], ys[idx]
    pred_error = y_pred[:len(x_query)].squeeze() - y.squeeze()
    ref_error = y_interp[:len(x_query)].squeeze() - y.squeeze()

    y_component = ['u', 'v', 'p']
    for y_idx in range(3):
        fig, ax = plt.subplots(2, 3, figsize=(17,10)) 

        cs_o = ax[0][0].tricontourf(x_query[:, 0], x_query[:, 1], y[:, y_idx].squeeze(), levels=50, cmap='jet')
        ax[0][0].scatter(geo_query[:, 0], geo_query[:, 1], s=0.7)
        fig.colorbar(cs_o, ax=ax[0][0])
        ax[0][0].set_title('Query GT Value')


        cs_interp = ax[0][1].tricontourf(x_query[:, 0], x_query[:, 1], y_interp[:len(x_query), y_idx].squeeze(), levels=50, cmap='jet')
        ax[0][1].scatter(geo_ref[:, 0], geo_ref[:, 1], s=0.7)
        fig.colorbar(cs_interp, ax=ax[0][1])
        ax[0][1].set_title('Triangular Interpolation from Ref')

        
        cs_r = ax[0][2].tricontourf(ref[:, 0], ref[:, 1], ref[:, y_idx + 2].squeeze(), levels=50, cmap='jet')
        ax[0][2].set_title('Reference')
        fig.colorbar(cs_r, ax=ax[0][2])

        cs_e = ax[1][0].tricontourf(x_query[:, 0], x_query[:, 1], y_pred[:len(x_query), y_idx], levels=50, cmap='jet')
        ax[1][0].scatter(geo_query[:, 0], geo_query[:, 1], s=0.7)
        fig.colorbar(cs_e, ax=ax[1][0])
        ax[1][0].set_title('Prediction')

        cs_e = ax[1][1].tricontourf(x_query[:, 0], x_query[:, 1], pred_error[:len(x_query), y_idx], levels=50, cmap='jet')
        # ax[1][1].scatter(geo_query[:, 0], geo_query[:, 1], s=0.7)
        fig.colorbar(cs_e, ax=ax[1][1])
        ax[1][1].set_title('Prediction Error')

        cs_e = ax[1][2].tricontourf(x_query[:, 0], x_query[:, 1], ref_error[:len(x_query), y_idx], levels=50, cmap='jet')
        ax[1][2].set_title('Ref Error')
        fig.colorbar(cs_e, ax=ax[1][2])

        plt.show()
        fig.savefig(save_path + f'_{y_component[y_idx]}.png')

    return fig

def plot_input_output(x_orig, y_pred_orig, x_r, y_r, opt_step, J, fig_path):
    y_comp = 3
    x_opt = x_orig.detach().cpu().numpy()
    y_opt = y_pred_orig.detach().cpu().numpy()
    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    cs = ax[0][0].scatter(x_opt[:, 0], x_opt[:, 1], c=x_opt[:, 2], s=3)
    ax[0][0].set_title('Query Input Theta')
    fig.colorbar(cs, ax=ax[0][0])
    cs = ax[1][0].scatter(x_opt[:, 0], x_opt[:, 1], c=y_opt[:, 1], s=3, cmap='RdBu_r')
    ax[1][0].set_title('Query Output u')
    fig.colorbar(cs, ax=ax[1][0])
    cs = ax[2][0].scatter(x_opt[:, 0], x_opt[:, 1], c=y_opt[:, y_comp], s=3, cmap='RdBu_r')
    ax[2][0].set_title('Query Output Concentration')
    fig.colorbar(cs, ax=ax[2][0])
    
    x_r = x_r.detach().cpu().numpy()
    y_r = y_r.detach().cpu().numpy()
    cs = ax[0][1].scatter(x_r[:, 0], x_r[:, 1], c=x_r[:, 2], s=3, cmap='viridis')
    ax[0][1].set_title(f'Ref Input Theta')
    fig.colorbar(cs, ax=ax[0][1])
    cs = ax[1][1].scatter(x_r[:, 0], x_r[:, 1], c=y_r[:, 1], s=3, cmap='RdBu_r')
    ax[1][1].set_title(f'Ref Output u')
    fig.colorbar(cs, ax=ax[1][1])
    cs = ax[2][1].scatter(x_r[:, 0], x_r[:, 1], c=y_r[:, y_comp], s=3, cmap='RdBu_r')
    ax[2][1].set_title(f'Ref Output Concentration')
    fig.colorbar(cs, ax=ax[2][1])

    if not os.path.exists(fig_path):
        os.makedirs(fig_path)
    fig.savefig(f'{fig_path}/Theta_c_{opt_step}_J={J}.png')
    plt.close()

def reg_theta(x_theta):
    mu, sig = 0.5, 0.1
    x_theta = (x_theta - mu) / sig
    x_theta = - x_theta.pow(2)
    reg = x_theta.exp().mean()
    # reg = x_theta.exp()
    
    return reg    
    
def micro_obj(x, y, i=0, N=100, spatial_dim=2):
    '''Input: x, y should be in original space
        Output: J in original space
    '''
    domain_dim = spatial_dim + 1
    if x.shape[-1] == domain_dim + 1:
        # print('Cropping effective domain.')
        domain = x[..., domain_dim].bool()
        x = x[domain]
        y = y[domain]
    
    theta = x[:, spatial_dim]
    c = y[:, 3]
    k_a = 0.25

    phi = k_a * (1 - theta) * c
    J = phi.mean() # maximize objective
    
    # # reg = reg_theta(theta)
    # # ratio = min(J.item() / reg.item(), 10)
    # reg = (((theta - 0.5).abs())**2).mean()
    # beta = J.detach().item() / (reg.detach().item() + 1e-8)
    # J_reg = (J + beta * reg) / 2
    # print(J_reg, J)
    J_reg = J
    
    
    # return J_reg, J, domain_idx
    return J_reg, J

def chunk_as(y, xs):
    ys, i = [], 0
    for x_ in xs:
        ys.append(y[i:i + len(x_)])
        i += len(x_)
    return ys

def micro_obj_batch(gs, y, x_norm, y_norm, spatial_dim=2, create_graph=True, retain_graph=True, original=True):
    '''Input:  xs and y are in transformed space.
        Output: dJdx in original space if original==True
    '''
    xs = [g_.ndata['x'] for g_ in gs]
    ys = chunk_as(y, xs)
    theta_std = x_norm.std[:, spatial_dim, None]
    
    dJdx_l = []
    J = 0
    for x_, y_ in zip(xs, ys):
        x_orig = x_norm.transform(x_, inverse=True, component_x=[0, 1, 2])
        y_orig = y_norm.transform(y_, inverse=True, component_x=[0, 1, 2, 3])
        J_, _ = micro_obj(x_orig, y_orig, spatial_dim)
        # J_ = y_orig[:, 3].mean() # Test autograd through model
        J = J + J_
    dJdx = torch.autograd.grad(J, xs, create_graph=create_graph, retain_graph=retain_graph)
    dJdx_l = [(dJ[:, spatial_dim, None] / theta_std) for dJ in dJdx] # rescale gradient to original space
    # print(dJdx_l)

    return dJdx_l

def drone_obj(x, y, i=0, N=100, spatial_dim=3):
    '''Input: x, y should be in original space
        Output: J in original space
    '''
    # print('Cropping effective domain.')
    # domain = x[..., -1].bool()
    # x = x[domain]
    # y = y[domain]
    
    # theta = x[:, spatial_dim]
    # e1, e2, e3, e12, e13, e23 = y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5] # strain
    # s1, s2, s3, s12, s13, s23 = y[:, 6], y[:, 7], y[:, 8], y[:, 9], y[:, 10], y[:, 11] # stress
    # # strain = torch.stack([torch.stack([e1, e12, e13], dim=-1), torch.stack([e12, e22, e23], dim=-1), torch.stack([e13, e23, e33], dim=-1)], dim=1) # nx3x3
    # # stress = torch.stack([torch.stack([s1, s12, s13], dim=-1), torch.stack([s12, s22, s23], dim=-1), torch.stack([s13, s23, s33], dim=-1)], dim=1) # nx3x3
    # # J = torch.einsum("nij, nij -> n", stress, strain).sum()

    # W = (e1*s1 + e2*s2 + e3*s3) + 2*(e12*s12 + e13*s13 + e23*s23)
    # J = W.mean() * 0.12 * 0.15 * 0.05
    # J = J + torch.max(theta.mean() - 0.125, 0)[0] # Volume constraint

    domain_idx = x[..., -1].bool() # material indices
    theta = x[:, spatial_dim, None]
    force = x[:, spatial_dim + 1 : spatial_dim + spatial_dim + 1] # x.shape[1] = spatial_dim + 1 (theta) + spatial_dim (force) + 3 (mask)
    disp = y[:, :spatial_dim]
    
    assert force.shape == disp.shape
    torque = (force[:, 0].abs().sum() > 0)
    # Compute potential energy - elasitic strain energy
    if not torque:
        J = (force * disp * theta)[domain_idx].sum(dim=1).mean() #* 0.12 * 0.15 * 0.05  # volume integral for gravity
        # print('Loadcase: gravity.')
    else:
        torque_idx = x[:, -3].bool() # torque surface indices
        J = (force * disp)[torque_idx].sum(dim=1).mean() * 0.01 * 0.01 # surface integral for torque
        # print('Loadcase: torque.')
    
    
    volfrac = torch.max(theta[domain_idx].mean() - 0.125, 0)[0]
    beta = J.item() / volfrac.item() if J.abs().item() < volfrac.item() else 1
    J_ = J + beta * volfrac # Volume constraint
    # print('theta: ', torch.max(theta.mean() - 0.125, 0)[0])
    # print('force: ', force.norm(p=2, dim=1).max())
    # print('displacement: ', disp.norm(p=2, dim=1).max())

    # J_ = J + 0.3 * theta.abs().mean() # L1 regularization
    
    return J_, J

def drone_obj_batch(gs, y, x_norm, y_norm, spatial_dim=3, create_graph=True, retain_graph=True, original=True):
    '''Input:  xs and y are in transformed space.
        Output: dJdx in original space if original==True
    '''
    xs = [g_.ndata['x'] for g_ in gs]
    ys = chunk_as(y, xs)
    theta_std = x_norm.std[:, spatial_dim, None]
    
    dJdx_l = []
    J = 0
    for x_, y_ in zip(xs, ys):
        x_orig = x_norm.transform(x_, inverse=True)
        y_orig = y_norm.transform(y_, inverse=True)
        J_, _ = drone_obj(x_orig, y_orig, spatial_dim)
        # J_ = y_orig[:, 3].mean() # Test autograd through model
        J = J + J_
    dJdx = torch.autograd.grad(J, xs, create_graph=create_graph, retain_graph=retain_graph)
    dJdx_l = [(dJ[:, spatial_dim, None] / theta_std) for dJ in dJdx] # rescale gradient to original space
    # print(dJdx_l[0].shape, xs[0].shape)
    # print('dJdx before screening: ', [dJ.view(-1).norm(p=2) for dJ in dJdx_l])
    dJdx_l = [dJ * x[:, -1, None]  for dJ, x in zip(dJdx_l, xs)] # screen dJ with domain mask
    # print('dJdx after screening: ', [dJ.view(-1).norm(p=2) for dJ in dJdx_l])
    dJdx_l = [dJ / dJ.view(-1).norm(p=2).detach().clone() for dJ in dJdx_l] # normalize dJdx
    # print([dJ.norm(p=2) for dJ in dJdx_l])

    # return dJdx_l
    return torch.cat(dJdx_l, dim=0).float()

def linear_interpolate(t, f, x):
    """
    Performs efficient differentiable linear interpolation in PyTorch for large input tensors.

    Args:
    t: 1D torch.Tensor representing the x-coordinates of the known function points.
    f: 1D torch.Tensor representing the corresponding y-coordinates of the known function points.
    x: 1D torch.Tensor representing the x-coordinates for interpolation.

    Returns:
    Interpolated values corresponding to the input x-coordinates.
    """

    # sign = torch.sign(x)
    x = x.abs()

    # Find the indices of the neighboring points
    idx_l = torch.searchsorted(t, x) - 1  # Index of the left neighbor
    idx_r = idx_l + 1  # Index of the right neighbor

    # Clamp indices to stay within the bounds of the data
    idx_l = torch.clamp(idx_l, 0, len(t) - 2)  # Clamp lower bound
    idx_r = torch.clamp(idx_r, 1, len(t) - 1)  # Clamp upper bound

    # Extract values of the neighboring points efficiently
    f_l = torch.index_select(f, 0, idx_l)
    f_r = torch.index_select(f, 0, idx_r)

    # Calculate the interpolation weights efficiently
    w_l = (t[idx_r] - x) / (t[idx_r] - t[idx_l] + 1e-8)
    w_r = 1 - w_l

    # Perform linear interpolation
    y_interp = w_l * f_l + w_r * f_r
    # y_interp = sign * y_interp

    return y_interp


def approximate_area_grid(mesh_points, grid_size, threshold):
    """
    Approximates the 2D area occupied by mesh points using a grid-based method.

    Args:
        mesh_points: A tensor of shape (N, 2) representing the x, y coordinates of the mesh points.
        grid_size: A tuple (grid_width, grid_height) defining the dimensions of the grid.
        threshold: The distance threshold for determining neighboring grid points.

    Returns:
        The estimated area occupied by the mesh points.
    """

    # Create a grid of points
    grid_x, grid_y = torch.meshgrid(
        torch.linspace(0, 0.2, grid_size[0]),
        torch.linspace(-0.25, 0.25, grid_size[1]),
        indexing='ij'
    )
    grid_points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1).to(mesh_points.device)

    # Calculate distances between mesh points and grid points
    distances = torch.cdist(mesh_points, grid_points)
    # print('distances.requires_grad:', distances.requires_grad)

    # Find grid points within the threshold distance
    relu = torch.nn.ReLU()
    neighboring_grid_points = relu(0.01 - distances).sum(dim=0)
    neighboring_grid_points = neighboring_grid_points / (neighboring_grid_points + 1e-8).detach().clone()
    # print('neighboring_grid_points.requires_grad:', neighboring_grid_points.requires_grad)

    # Calculate the estimated area
    grid_area = 0.2 * 0.5 / ((grid_size[0] -1) * (grid_size[1]-1))  # Area of each grid cell
    estimated_area = neighboring_grid_points.sum() * grid_area

    return estimated_area


def inductor_obj(x, y, i=0, N=100):
    '''Input: x, y should be in original space
        Output: J in original space
    '''
    device = x.device
    core_idx = x[:, 4].bool()
    alpha = x[0, 2]
    r = x[core_idx, 0]
    B = y[core_idx, :2]
    t = torch.tensor([0, 1.000000051691021, 1.4936495124126294, 1.9415328461315795, 2.257765669366018, 2.609980642431287, 2.8664452090837504, 3.1441438097176118, 3.448538051654125, 3.7816711973679054, 4.058345590113038, 4.420646552950275, 4.721274089545955, 4.972148140718701, 5.145510860855953, 5.245510861426532]).float().to(device)
    f = torch.tensor([0, 663.146, 1067.5, 1705.23, 2463.11, 3841.67, 5425.74, 7957.75, 12298.3, 20462.8, 32169.6, 61213.4, 111408, 188487.757, 267930.364, 347507.836]).to(device)
    H = linear_interpolate(t, f, B.view(-1)).reshape(-1, 2)
    
    Area = approximate_area_grid(x[core_idx, :2], (21, 51), 1e-2)
    W = (0.5 *(H.abs() * B.abs()).sum(dim=1) * r).mean() * 2 * torch.tensor(np.pi) * Area.detach().clone()
    # print(f'W={W}, Area={Area}')

    J =  (1- alpha) * W - alpha * Area # minimize objective
    J_reg = J # To be implemented
    
    return J_reg, J   


def shape_obj(x, y, i=0, N=100):
    '''Input: x, y should be in original space
        Output: J in original space
    '''
    arm_idx = x[:, 3].bool()
    inlet_idx = (x[:, 0] == x[:, 0].min())
    left_arm_idx = (x[:, 0] < x[:, 0].max() / 2) & arm_idx
    channel_probe_idx = (x[:, 0] == x[left_arm_idx, 0].max())

    pressure_drop = y[inlet_idx, 0].mean() - 0 # pressure at outlet is assumed zero
    u_var = y[channel_probe_idx, 1].var()
    J = pressure_drop + u_var # minimize objective
    
    J_reg = J # To be implemented
    
    return J_reg, J

def shape_obj_batch(gs, y, x_norm, y_norm, spatial_dim=2, create_graph=True, retain_graph=True, original=True, objective=None):
    '''Input:  xs and y are in transformed space.
        Output: dJdx in original space if original==True
    '''
    xs = [g_.ndata['x'] for g_ in gs]
    ys = chunk_as(y, xs)  
    
    if xs[0].shape[-1] == 4:
        objective = shape_obj
    elif xs[0].shape[-1] == 5:
        objective = inductor_obj
    else:
        raise NotImplementedError

    dJdx_l = []
    J = 0
    for x_, y_ in zip(xs, ys):
        x_orig = x_norm.transform(x_, inverse=True)
        y_orig = y_norm.transform(y_, inverse=True)
        J_, _ = objective(x_orig, y_orig)
        # J_ = y_orig[:, 3].mean() # Test autograd through model
        J = J + J_      

    dJdx = torch.autograd.grad(J, xs, create_graph=create_graph, retain_graph=retain_graph)
    dJdx = [dJ * x[:, -1, None]  for dJ, x in zip(dJdx, xs)] # screen dJ with domain mask
    dJdx = [(dJ[:, :spatial_dim] / x_norm.std) for dJ in dJdx] # rescale gradient to original space
    dJdx = [dJ / (dJ.norm(p=2, dim=1, keepdim=True).detach().clone() + 1e-8) for dJ in dJdx] # normalize dJdx


    return dJdx