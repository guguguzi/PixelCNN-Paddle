import paddle
from paddle import nn


class CausalConv2d(nn.Conv2D):
    """A Conv2d layer masked to respect the autoregressive property.

    Autoregressive masking means that the computation of the current pixel only
    depends on itself, pixels to the left, and pixels above. When mask_center=True, the
    computation of the current pixel does not depend on itself.

    E.g. for a 3x3 kernel, the following masks are generated for each channel:
                          [[1 1 1],                     [[1 1 1],
        mask_center=False  [1 1 0],    mask_center=True  [1 0 0],
                           [0 0 0]]                      [0 0 0]
    In [1], they refer to the left masks as 'type A' and right as 'type B'.

    NOTE: This layer does *not* implement autoregressive channel masking.
    """

    def __init__(self, mask_center, *args, **kwargs):
        """Initializes a new CausalConv2d instance.

        Args:
            mask_center: Whether to mask the center pixel of the convolution filters.
        """
        super().__init__(*args, **kwargs)
        i, o, h, w = self.weight.shape
        mask = paddle.zeros((i, o, h, w))
        mask[:, :, : h // 2, :] = 1
        mask[:, :, h // 2, : w // 2 + int(not mask_center)] = 1
        self.register_buffer("mask", mask)

    def forward(self, x):
        self.weight.set_value( self.weight.detach() * self.mask)
        return super().forward(x)


class GatedActivation(nn.Layer):
    """Gated activation function as introduced in [2].

    The function computes actiation_fn(f) * sigmoid(g). The f and g correspond to the
    top 1/2 and bottom 1/2 of the input channels.
    """

    def __init__(self, activation_fn=paddle.tanh):
        """Initializes a new GatedActivation instance.

        Args:
            activation_fn: Activation to use for the top 1/2 input channels.
        """
        super().__init__()
        self._activation_fn = activation_fn

    def forward(self, x):
        _, c, _, _ = x.shape
        assert c % 2 == 0, "x must have an even number of channels."
        x, gate = x[:, : c // 2, :, :], x[:, c // 2 :, :, :]
        return self._activation_fn(x) * nn.functional.sigmoid(gate)


class NCHWLayerNorm(nn.LayerNorm):
    """Applies LayerNorm to the channel dimension of NCHW tensors."""

    def forward(self, x):
        x = x.transpose([0, 2, 3, 1])
        x = super().forward(x)
        return x.transpose([0, 3, 1, 2])
