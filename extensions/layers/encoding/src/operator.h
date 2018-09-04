#include <torch/torch.h>
#include <vector>


at::Tensor Aggregate_Forward_CUDA(
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_);

std::vector<at::Tensor> Aggregate_Backward_CUDA(
  const at::Tensor GE_,
  const at::Tensor A_,
  const at::Tensor X_,
  const at::Tensor C_);

at::Tensor ScaledL2_Forward_CUDA(
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor S_);

std::vector<at::Tensor> ScaledL2_Backward_CUDA(
  const at::Tensor GSL_,
  const at::Tensor X_,
  const at::Tensor C_,
  const at::Tensor S_,
  const at::Tensor SL_);
