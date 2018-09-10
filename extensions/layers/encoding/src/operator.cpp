#include "operator.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("aggregate_forward", &Aggregate_Forward_CUDA, "Aggregate forward (CUDA)");
  m.def("aggregate_backward", &Aggregate_Backward_CUDA, "Aggregate backward (CUDA)");
  m.def("scaled_l2_forward", &ScaledL2_Forward_CUDA, "ScaledL2 forward (CUDA)");
  m.def("scaled_l2_backward", &ScaledL2_Backward_CUDA, "ScaledL2 backward (CUDA)");
}
