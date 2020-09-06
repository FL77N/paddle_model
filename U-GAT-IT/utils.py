import numpy as np
from scipy import misc
import os, cv2
from PIL import Image
import os.path

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import SpectralNorm
from paddle.fluid.dygraph.base import to_variable
from paddle.fluid.dygraph import BCELoss
from paddle.fluid.layers import sigmoid, clip, relu, leaky_relu, unsqueeze, tanh, create_parameter, reduce_mean

class ReflectionPad2D(fluid.dygraph.Layer):
    
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):

        return fluid.layers.pad2d(x, self.padding, mode='reflect')

class Instancenorm(fluid.dygraph.Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):

        return fluid.layers.instance_norm(x)

class Resize_nearest(fluid.dygraph.Layer):

    def __init__(self, scale):
        super().__init__()
     
        self.scale = scale

    def forward(self, x):

        return fluid.layers.resize_nearest(x, scale=self.scale)

class Tanh(fluid.dygraph.Layer):

    def __init__(self):
        super().__init__()
    
    def forward(self, x):

        return tanh(x)

def my_var(input, dim=None, keep_dim=False, unbiased=True, name=None):
        
    rank = len(input.shape)
    dims = dim if dim != None and dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = reduce_mean((input - mean)**2, dim=dim, keep_dim=keep_dim, name=name)

    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor

    return tmp

class Spectralnorm(paddle.fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):

        super(Spectralnorm, self).__init__()

        self.spectral_norm = SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
 
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)

        return out

class ReLU(paddle.fluid.dygraph.Layer):

    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):

        if self.inplace:
            x.set_value(relu(x))
            return x
        else:
            y = relu(x)
            return y

class LeakyReLU(paddle.fluid.dygraph.Layer):

    def __init__(self, alpha, inplace=False):
        super(LeakyReLU, self).__init__()
        self.inplace = inplace
        self.alpha = alpha

    def forward(self, x):

        if self.inplace:
            x.set_value(leaky_relu(x, alpha=self.alpha))
            return x
        else:
            y = leaky_relu(x, alpha=self.alpha)
            return y

def load_test_data(image_path, size=256):
    img = misc.imread(image_path, mode='RGB')
    img = misc.imresize(img, [size, size])
    img = np.expand_dims(img, axis=0)
    img = preprocessing(img)

    return img

def preprocessing(x):

    x = x/127.5 - 1 

    return x

def save_images(images, size, image_path):

    return imsave(inverse_transform(images), size, image_path)

def inverse_transform(images):

    return (images+1.) / 2

def imsave(images, size, path):

    return misc.imsave(path, merge(images, size))

def merge(images, size):

    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[h*j:h*(j+1), w*i:w*(i+1), :] = image

    return img

def check_folder(log_dir):

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def str2bool(x):

    return x.lower() in ('true')

def cam(x, size = 256):

    x = x - np.min(x)
    cam_img = x / np.max(x)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, (size, size))
    cam_img = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)

    return cam_img / 255.0

def imagenet_norm(x):

    mean = fluid.Tensor(np.ndarray([0.485, 0.456, 0.406]), fluid.CUDAPlace())
    std = fluid.Tensor(np.ndarray([0.299, 0.224, 0.225]), fluid.CUDAPlace())
    mean = unsqueeze(mean, axes=0)
    mean = unsqueeze(mean, axes=2)
    mean = unsqueeze(mean, axes=3)
    std = unsqueeze(std, axes=0)
    std = unsqueeze(std, axes=2)
    std = unsqueeze(std, axes=3)

    return (x - mean) / std

def denorm(x):

    return x*0.5+0.5

def tensor2numpy(x):

    out = x.numpy()

    return out.transpose(1,2,0)

def RGB2BGR(x):
    
    return cv2.cvtColor(x, cv2.COLOR_RGB2BGR)

class BCEWithLogitsLoss():

    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):

        out = paddle.fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out

def read_img(reader):

    def r():
        
        for i in reader:
            yield i
        
    return r

def clip_rho(net, vmin=0, vmax=1):

    for name, param in net.named_parameters():
        if 'rho' in name:
            param.set_value(clip(param, vmin, vmax))
