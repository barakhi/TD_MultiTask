import torch
import torch.nn.functional as F
import math
import numbers
#import torch
from torch import nn
#from torch.nn import functional as F
from torchvision import transforms





class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups)


#smoothing = GaussianSmoothing(3, 5, 1)
#input = torch.rand(1, 3, 100, 100)
#input = F.pad(input, (2, 2, 2, 2), mode='reflect')
#output = smoothing(input)


def SpatialClassificationLoss(seg_log, xy_task_com):
    n, c, h, w = seg_log.size()
    #target = torch.zeros_like(seg_log)

    sm_p = F.log_softmax(seg_log.view(n,h*w), dim=1).view(n,c,h,w)

    target_list = []
    seg_list = []
    for i, xy_task in enumerate(xy_task_com):
        if (xy_task.size(0) ==2):
            target = torch.zeros(h, w)
            target[(xy_task[1]).long(), (xy_task[0]).long()] = 1
            target_list.append(target)
            seg_list.append(sm_p[i, 0, :, :])
        else:
            if (xy_task[0] and xy_task[1] >= 0 and xy_task[1] < w and xy_task[2] >= 0 and xy_task[2] < h):
                target = torch.zeros(h,w)
                target[torch.floor(xy_task[2]).long(),torch.floor(xy_task[1]).long()] = 1
                # blurring
                target_list.append(target)
                seg_list.append(sm_p[i,0,:,:])

    input_for_loss = torch.stack(seg_list).unsqueeze(1)

    with torch.no_grad():
        target_for_loss = torch.stack(target_list).unsqueeze(1)
        smoothing = GaussianSmoothing(1, 15, 3)
        target_for_loss = F.pad(target_for_loss, (7,7,7,7), mode='reflect')
        target_for_loss = smoothing(target_for_loss)
        target_for_loss = target_for_loss.cuda()

    loss = torch.nn.KLDivLoss(reduction='sum')(input_for_loss, target_for_loss)/n

    return loss


