
import paddle
from paddle.fluid.dygraph import Conv2D, Linear, Sequential
from paddle.fluid.layers import (adaptive_pool2d, reshape, unsqueeze, concat, transpose,
                                reduce_sum, create_parameter, reduce_mean, sqrt, expand,clip)

from utils import my_var, ReLU, LeakyReLU, Spectralnorm, ReflectionPad2D, Instancenorm, Resize_nearest, Tanh

class ResnetGenerator(paddle.fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2D(3),
                      Conv2D(input_nc, num_filters=ngf, filter_size=7, stride=1, padding=0),
                      Instancenorm(),
                      ReLU()]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2D(1),
                          Conv2D(ngf * mult, num_filters=ngf * mult * 2, filter_size=3, stride=2),
                          Instancenorm(),
                          ReLU()]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        
        self.gap_fc = Linear(ngf * mult, 1)
        self.gmp_fc = Linear(ngf * mult, 1)
        self.conv1x1 = Conv2D(ngf * mult * 2, num_filters=ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU()

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult),
                  ReLU(),
                  Linear(ngf * mult, ngf * mult),
                  ReLU()]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult),
                  ReLU(),
                  Linear(ngf * mult, ngf * mult),
                  ReLU()]

        self.gamma = Linear(ngf * mult, ngf * mult)
        self.beta = Linear(ngf * mult, ngf * mult)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Resize_nearest(scale=2),
                         ReflectionPad2D(1),
                         Conv2D(ngf * mult, num_filters=int(ngf * mult / 2), filter_size=3, stride=1),
                         ILN(int(ngf * mult / 2)),
                         ReLU()]

        UpBlock2 += [ReflectionPad2D(3),
                     Conv2D(ngf, num_filters=output_nc, filter_size=7, stride=1),
                     Tanh()]

        self.DownBlock = Sequential(*DownBlock)
        self.FC = Sequential(*FC)
        self.UpBlock2 = Sequential(*UpBlock2)

    def forward(self, input):

        x = self.DownBlock(input)

        gap = adaptive_pool2d(x, pool_size=[1, 1], pool_type='avg')

        gap_ = reshape(x=gap, shape=(x.shape[0], -1))

        gap_logit = self.gap_fc(gap_)  

        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = transpose(gap_weight, perm=[1, 0])
        gap_weight = unsqueeze(gap_weight, axes=2)
        gap_weight = unsqueeze(gap_weight, axes=3)

        gap = x * gap_weight

        gmp = adaptive_pool2d(x, pool_size=[1, 1], pool_type='max')

        gmp_ = reshape(x=gmp, shape=(x.shape[0], -1))

        gmp_logit = self.gmp_fc(gmp_)

        gmp_weight = self.gmp_fc.parameters()[0]
        gmp_weight = transpose(gmp_weight, perm=[1, 0])
        gmp_weight = unsqueeze(gmp_weight, axes=2)
        gmp_weight = unsqueeze(gmp_weight, axes=3)
      
        gmp = x * gmp_weight

        cam_logit = concat(input=[gap_logit, gmp_logit], axis=1)

        x = concat(input=[gap, gmp], axis=1)

        x = self.relu(self.conv1x1(x))

        heatmap = reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = adaptive_pool2d(x, pool_size=[1, 1], pool_type='avg')
            x_ = reshape(x=x_, shape=(x_.shape[0], -1))
            x_ = self.FC(x_)
        else:
            x_ = reshape(x, shape=(x.shape[0], -1))
            x_ = self.FC(x_)

        gamma, beta = self.gamma(x_), self.beta(x_)

        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(paddle.fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []

        conv_block += [ReflectionPad2D(1),
                      Conv2D(dim, num_filters=dim, filter_size=3, stride=1, bias_attr=use_bias),
                      Instancenorm(),
                      ReLU()]

        conv_block += [ReflectionPad2D(1),
                       Conv2D(dim, num_filters=dim, filter_size=3, stride=1, bias_attr=use_bias),
                       Instancenorm()]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(paddle.fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = Conv2D(dim, num_filters=dim, filter_size=3, stride=1, bias_attr=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU()

        self.pad2 = ReflectionPad2D(1)
        self.conv2 = Conv2D(dim, num_filters=dim, filter_size=3, stride=1, bias_attr=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):

        out = self.pad1(x)

        out = self.conv1(out)

        out = self.norm1(out, gamma, beta)

        out = self.relu1(out)

        out = self.pad2(out)

        out = self.conv2(out)

        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(paddle.fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=paddle.fluid.initializer.Constant(0.9))

    def forward(self, input, gamma, beta):

        in_mean, in_var = reduce_mean(input, dim=[2, 3], keep_dim=True), my_var(input, dim=[2, 3], keep_dim=True)

        out_in = (input - in_mean) / sqrt(in_var + self.eps)

        ln_mean, ln_var = reduce_mean(input, dim=[1, 2, 3], keep_dim=True), my_var(input, dim=[1, 2, 3], keep_dim=True)

        out_ln = (input - ln_mean) / sqrt(ln_var + self.eps)

        ex_rho = expand(self.rho, (input.shape[0], 1, 1, 1))

        out = ex_rho * out_in + (1-ex_rho) * out_ln

        gamma = unsqueeze(gamma, axes=2)
        gamma = unsqueeze(gamma, axes=3)
        beta = unsqueeze(beta, axes=2)
        beta = unsqueeze(beta, axes=3)
        out = out * gamma + beta

        return out


class ILN(paddle.fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=paddle.fluid.initializer.Constant(0.0))
        self.gamma = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=paddle.fluid.initializer.Constant(1.0))
        self.beta = self.create_parameter(shape=(1, num_features, 1, 1), dtype='float32', default_initializer=paddle.fluid.initializer.Constant(0.0))

    def forward(self, input):
        
        in_mean, in_var = reduce_mean(input, dim=[2, 3], keep_dim=True), my_var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / sqrt(in_var + self.eps)
        ln_mean, ln_var = reduce_mean(input, dim=[1, 2, 3], keep_dim=True), my_var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / sqrt(ln_var + self.eps)
        ex_rho = expand(self.rho, (input.shape[0], 1, 1, 1))
        out = ex_rho * out_in + (1-ex_rho) * out_ln
        ex_gamma = expand(self.gamma, (input.shape[0], 1, 1, 1))
        ex_beta = expand(self.beta, (input.shape[0], 1, 1, 1))
        out = out * ex_gamma + ex_beta

        return out


class Discriminator(paddle.fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2D(1),
                 Spectralnorm(
                 layer = Conv2D(input_nc, num_filters=ndf, filter_size=4, stride=2, bias_attr=True)),
                 LeakyReLU(alpha=0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2D(1),
                      Spectralnorm(
                      layer = Conv2D(ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=2, bias_attr=True)),
                      LeakyReLU(alpha=0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2D(1),
                  Spectralnorm(
                  layer = Conv2D(ndf * mult, num_filters=ndf * mult * 2, filter_size=4, stride=1, bias_attr=True)),
                  LeakyReLU(alpha=0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(layer = Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(layer = Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, num_filters=ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = LeakyReLU(alpha=0.2)

        self.pad = ReflectionPad2D(1)

        self.conv = Spectralnorm(
            layer = Conv2D(ndf * mult, num_filters= 1, filter_size=4, stride=1, bias_attr=False))

        self.model = Sequential(*model)

    def forward(self, input):

        x = self.model(input)

        gap = adaptive_pool2d(x, pool_size=[1, 1], pool_type='avg')

        gap_ = reshape(gap, shape=[x.shape[0], -1])
        
        gap_logit = self.gap_fc(gap_)

        gap_weight = self.gap_fc.parameters()[0]
        gap_weight = transpose(gap_weight, perm=[1, 0])
        gap_weight = unsqueeze(gap_weight, axes=2)
        gap_weight = unsqueeze(gap_weight, axes=3)

        gap = x * gap_weight

        gmp = adaptive_pool2d(x, pool_size=[1, 1], pool_type='max')

        gmp_ = reshape(gmp, shape=[x.shape[0], -1])

        gmp_logit = self.gmp_fc(gmp_)

        gmp_weight = self.gmp_fc.parameters()[0]
        gmp_weight = transpose(gmp_weight, perm=[1, 0])
        gmp_weight = unsqueeze(gmp_weight, axes=2)
        gmp_weight = unsqueeze(gmp_weight, axes=3)

        gmp = x * gmp_weight

        cam_logit = concat(input=[gap_logit, gmp_logit], axis=1)

        x = concat(input=[gap, gmp], axis=1)

        x = self.leaky_relu(self.conv1x1(x))

        heatmap = reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        
        out = self.conv(x)

        return out, cam_logit, heatmap


