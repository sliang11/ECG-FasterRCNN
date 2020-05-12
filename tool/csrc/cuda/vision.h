// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#pragma once
#include <torch/extension.h>



std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_width);

at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_width,
                                 const int channels,
                                 const int width);

at::Tensor nms_cuda(const at::Tensor boxes, const at::Tensor scores, float nms_overlap_thresh);


at::Tensor compute_flow_cuda(const at::Tensor& boxes,
                             const int height,
                             const int width);
