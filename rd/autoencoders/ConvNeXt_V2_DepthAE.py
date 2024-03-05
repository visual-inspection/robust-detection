"""
File: ConvNeXt_V2_DepthAE
Author: Sebastian Hönel

A "depth-wise" feature extractor for ConvNeXt V2. "Depth" relates to the fact
that features are learned along the channels, which is likely to be not optimal.
"""

from torch import nn, device, Tensor, reshape, swapaxes
from rd.tools.Split import Split
from rd.tools.MinPool2d import MinPool2d



class PrepDepthwiseBatchNorm1d(nn.Module):
    def __init__(self, inverse: bool=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inverse = inverse
    
    def forward(self, x: Tensor) -> Tensor:
        if self.inverse:
            x = reshape(x, (x.shape[0], 16, 16, 2816))
            x = swapaxes(swapaxes(x, 3, 2), 2, 1)
        else:
            x = swapaxes(swapaxes(x, 1, 2), 2, 3)
            x = reshape(x, (x.shape[0], 256, 2816))
        return x




class DepthAE(nn.Module):
    """
    This is the first type of AE on the ConvNeXT-V2 features that will work
    depth-wise. The data comes in 16x16x2816 and we will attempt to convolute
    along the first two axes, considering the many features along the last
    within each kernel's slice.
    The bottleneck of this AE should be a 1x1 convolution which we will later
    use for extracting features.
    """
    def __init__(self, dev: device, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ae = nn.Sequential(
            # Normalize each feature depth-wise along its deep dimension here.
            # The data comes as 2816x16x16, or, in other words, it comes as 256
            # 2816-valued features.
            PrepDepthwiseBatchNorm1d(),
            # N, C, L:
            nn.BatchNorm1d(num_features=256),
            PrepDepthwiseBatchNorm1d(inverse=True),

            # 2816x16x16  -->>  352x13x13
            #
            # Each filter has 4x4x2816+1=45,057 weights+bias and produces 13x13=169 values
            # We have 352 filters resulting in 15,860,064 weights/biases
            # We have 352x13x13=59,488 outputs
            nn.Conv2d(in_channels=2816, out_channels=44, kernel_size=(4,4), stride=(1,1), padding=0, bias=True),

            nn.SiLU(inplace=True),
            

            # 1x13x13 (Computes a max for each depth-wise feature). While this is what we want,
            # it reduces the amount of features perhaps too drastically.
            #nn.MaxPool3d(kernel_size=(352,1,1), stride=(1,1,1))
            Split(
                # 352x11x11 (Computes max of 352 13x13 patches, i.e., for each depth-slice/channel).
                # While this can work, this is not what we want, as we consider a feature along a
                # depth-axis.
                MinPool2d(kernel_size=(3,3), stride=(1,1), padding=0),
                nn.AvgPool2d(kernel_size=(3,3), stride=(1,1), padding=0),
                nn.MaxPool2d(kernel_size=(3,3), stride=(1,1), padding=0)
            ),

            # Here we learn 128 filters, each applied to a 352x1x1 slice. We will get
            # 128x11x11=15,488 features out of this.
            nn.Conv2d(in_channels=3*44, out_channels=132, kernel_size=(1,1), stride=1, bias=True, groups=3),


            #####  BOTTLENECK HERE  #####
            #####  BOTTLENECK HERE  #####
            #####  BOTTLENECK HERE  #####
            # Once trained, we shall cut of the AE between this last convolution and the next activation!

            nn.SiLU(inplace=True),

            ##### De-Convolution starts here  #####

            # Now we have to come back to 2816x16x16 from 128x11x11

            nn.ConvTranspose2d(in_channels=132, out_channels=352, kernel_size=(5,5), stride=1),
            nn.SiLU(inplace=True),

            # nn.Flatten(),
            nn.Dropout(inplace=True, p=1./3.),
            # nn.Linear(in_features=352*15*15, out_features=352*15*15)
            # # nn.LeakyReLU(inplace=True),
            # # Reshape(shape=(352,13,13)),

            nn.ConvTranspose2d(in_channels=352, out_channels=2816, kernel_size=(2,2), stride=1),
            nn.LeakyReLU(inplace=True)

        ).to(device=dev)
    
    def forward(self, x):
        return self.ae(x)



# TODO: Use Sequential::add_module to add and identify named module instead.

class InferenceDepthAE(nn.Module):
    """
    This class can perform inference using a DepthAE module. It slices the AE
    at the bottleneck.
    """
    def __init__(self, model: DepthAE, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.fe = nn.Sequential(*model.ae[0:7])
        assert isinstance(self.fe[-1], nn.Conv2d)
        assert isinstance(self.fe[-2], Split)
        self.fe.train = False
        self.train = False
    
    def forward(self, x: Tensor) -> Tensor:
        return self.fe.forward(x)
