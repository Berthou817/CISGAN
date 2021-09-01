import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out

class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1

class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 dlatent_size=512,
                 resolution=128,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents #True
        self.resolution_log2 = int(np.log2(resolution)) # 6
        self.num_layers = self.resolution_log2 * 2 - 2 # 10
        self.pixel_norm = PixelNorm()


    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers

class Conv3d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv3d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv3d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)

class Upscale3d(nn.Module):
    def __init__(self, factor=2, gain=1):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super().__init__()
        self.gain = gain
        self.factor = factor


    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1, shape[4], 1).expand(-1, -1, -1, self.factor, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3], self.factor * shape[4])

        return x

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.zeros(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1, 1) * noise.to(x.device)

class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):

        style = self.linear(latent) # style => [batch_size, n_channels*2]

        shape = [-1, 2, x.size(1), 1, 1 ,1]

        style = style.view(shape)    # [batch_size, 2, n_channels, ...]

        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x

class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()
        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):

        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        return x

class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,        # noise
                 dlatent_size=512,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=4096,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=32,       # Maximum number of feature maps in any layer.
                 ):
        super(GBlock, self).__init__()

        self.nf = [128,128,128,128,128,32]
        # res
        self.res = res

        # blur2d
        #self.blur = Blur2d(f)

        # noise
        self.noise_input = noise_input
        
        self.up_sample = Upscale3d(factor)
        # A Composition of LayerEpilogue and Conv2d.
        self.adaIn1 = LayerEpilogue(self.nf[res-2], dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv3d(input_channels=self.nf[res-2], output_channels=self.nf[res-2],
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf[res-2], dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):

        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x
class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=128,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=4096,                     # Overall multiplier for the number of feature maps.
                 num_channels=1,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=128,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 f=None,                        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_pixel_norm      = True,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,        # Enable instance normalization?
                 use_wscale = True,                  # Enable equalized learning rate?
                 use_noise = True,                   # Enable noise inputs?
                 use_style = True                    # Enable style inputs?
                 ):                             # batch size.
        """
            2019.3.31
        :param dlatent_size: 512 Disentangled latent(W) dimensionality.
        :param resolution: 1024 x 1024.
        :param fmap_base:
        :param num_channels:
        :param structure: only support 'fixed' mode.
        :param fmap_max:
        """
        super(G_synthesis, self).__init__()


        self.nf = [128,128,128,128,128,128,128,128,128,128,32,32,32,32,32]
        self.structure = structure
        self.resolution_log2 = int(np.log2(resolution))
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        num_layers = self.resolution_log2 * 2 - 2 # 12
        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res, 2 ** res]
            self.noise_inputs.append(torch.randn(*shape).to("cuda"))


        # Blur2d
        #self.blur = Blur2d(f)

        # torgb: fixed mode
        self.channel_shrinkage = Conv3d(input_channels=32,
                                        output_channels=32,
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv3d(32, num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        # Initial Input Block
        self.const_input = nn.Parameter(torch.ones(1, self.nf[1], 4, 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf[1]))
        self.adaIn1 = LayerEpilogue(self.nf[1], dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv3d(input_channels=self.nf[1], output_channels=self.nf[1], kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf[1], dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)
        self.transform= Conv3d(128,32,1,1,use_wscale=use_wscale)

        # Common Block
        # 4 x 4 -> 8 x 8
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 8 x 8 -> 16 x 16
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 16 x 16 -> 32 x 32
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 32 x 32 -> 64 x 64
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 64 x 64 -> 128 x 128
        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)
      

    def forward(self, dlatent):


        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:
            x = self.const_input.expand(dlatent.size(0), -1, -1, -1, -1)

            x = x + self.bias.view(1, -1, 1, 1, 1)

            #
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])

            #
            #
            x = self.conv1(x)

            #
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

            #
            # # block 1:
            # # 4 x 4 -> 8 x 8
            #
            x = self.GBlock1(x, dlatent)

            #
            #
            # # block 2:
            # # 8 x 8 -> 16 x 16
            x = self.GBlock2(x, dlatent)
            #
            # # block 3:
            # # 16 x 16 -> 32 x 32
            x = self.GBlock3(x, dlatent)
            #
            # # block 4:
            # # 32 x 32 -> 64 x 64
            x = self.GBlock4(x, dlatent)

            x = self.transform(x)



            # block 5:
            # # 64 x 64 -> 128 x 128
            x = self.GBlock5(x, dlatent)



            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out

class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps=512,
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.5,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=12,          # Number of layers for which to apply the truncation trick. None = disable.
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1,pore):

        latents1 = torch.cat((latents1, pore), 1)
        # latents1= latents1.data.cpu().numpy()
        # label = label.data.cpu().numpy().reshape(latents1.shape[0], 1)
        # latents1 = np.concatenate((latents1, label), axis=1)
        # latents1 = torch.tensor(latents1).type(torch.FloatTensor).cuda()


        dlatents1, num_layers = self.mapping(latents1)
        # let [N, O] -> [N, num_layers, O]

        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)


        # Apply truncation trick.
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            """Linear interpolation.
               a + (b - a) * t (a = 0)
               reduce to
               b * t
            """

            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)

        img = self.synthesis(dlatents1)
        img = F.sigmoid(img)
        return img
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1,128,4,2,1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128,128,4,2,1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 128, 4, 2, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),


            nn.Conv3d(128, 128, 3, 1, 1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 1, 3, 1, 1),




        )
        self.regression = nn.Linear(512,1)
    
        # )
        self.pore = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),

        )
    def forward(self,x):

        x = self.net(x)

        out=x.view(x.shape[0],-1)

        out = self.regression(out)
        pore = x.view(x.shape[0],512)
        pore = self.pore(pore)
        return out, pore
