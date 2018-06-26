from torch.autograd import Function, Variable
from .._ext import channelnorm


class ChannelNormFunction(Function):

    @staticmethod
    def forward(ctx, input1, norm_deg=2):
        assert input1.is_contiguous()
        b, _, h, w = input1.size()
        output = input1.new(b, 1, h, w).zero_()

        channelnorm.ChannelNorm_cuda_forward(input1, output, norm_deg)
        ctx.save_for_backward(input1, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input1, output = ctx.saved_tensors

        grad_input1 = Variable(input1.new(input1.size()).zero_())

        channelnorm.ChannelNorm_cuda_backward(input1, output, grad_output.data,
                                              grad_input1.data, ctx.norm_deg)

        return grad_input1, None
