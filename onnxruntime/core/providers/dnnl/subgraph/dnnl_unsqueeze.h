// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_subgraph.h"
#include "dnnl_subgraph_primitive.h"

namespace onnxruntime {
namespace ort_dnnl {

class DnnlUnsqueeze {
 public:
  enum InputTensors : int {
    IN_DATA = 0,
    IN_AXES = 1,
  };

  enum OutputTensors : int {
    OUT_EXPANDED = 0
  };

  DnnlUnsqueeze();
  void CreatePrimitive(DnnlSubgraphPrimitive& sp, DnnlNode& node);

 private:
  std::vector<int64_t> GetAxes(DnnlNode& node);
};

}  // namespace ort_dnnl
}  // namespace onnxruntime