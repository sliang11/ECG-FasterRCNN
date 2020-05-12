// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>


// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__global__ void RoIPoolFForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int width, const int pooled_width,
    const T* bottom_rois, T* top_data, int* argmax_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int c = (index / pooled_width ) % channels;
    int n = index / pooled_width  / channels;

    const T* offset_bottom_rois = bottom_rois + n * 2;
    int roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[1] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);

    T bin_size_w = static_cast<T>(roi_width)
                       / static_cast<T>(pooled_width);

    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));

    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));

    // Add roi offsets and clip to input boundaries
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    //bool is_empty = (wend <= wstart);

    // Define an empty pooling region to be zero
    //T maxval = is_empty ? 0 : -FLT_MAX;
    T maxval=0;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    //int maxidx = -1;
    int maxidx = 0;
    const T* offset_bottom_data =bottom_data + c* width;

    if (wstart!=wend){
    for (int w = wstart; w < wend; ++w) {
        int bottom_index = w;
        maxval += offset_bottom_data[bottom_index];
        maxidx++;

    }
    }
    else{
        maxval +=offset_bottom_data[wstart];
        maxidx++;
    }
    top_data[index]=static_cast<T>(maxval/float(maxidx));
    argmax_data[index]=maxidx;

  }
}

template <typename T>
__global__ void RoIPoolFBackward(const int nthreads, const T* top_diff,
    const int* argmax_data, const int num_rois, const T spatial_scale,
    const int channels, const int width,
    const int pooled_width, T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int c = (index / pooled_width ) % channels;
    int n = index / pooled_width  / channels;


    const T* offset_bottom_rois = bottom_rois + n * 2;
    int roi_start_w = round(offset_bottom_rois[0] * spatial_scale);
    int roi_end_w = round(offset_bottom_rois[1] * spatial_scale);
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);

    T bin_size_w = static_cast<T>(roi_width)
                       / static_cast<T>(pooled_width);

    int wstart = static_cast<int>(floor(static_cast<T>(pw)
                                        * bin_size_w));

    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1)
                                     * bin_size_w));
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);

    int bottom_offset =  c * width;
    int top_offset    = (n * channels + c)  * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    T* offset_bottom_diff = bottom_diff + bottom_offset;

    const int* offset_argmax_data = argmax_data + top_offset;
    int argmax = offset_argmax_data[pw];

    if (wstart!=wend){
        for (int w = wstart; w < wend; ++w) {
            int bottom_index = w;
            offset_bottom_diff[bottom_index]=static_cast<T>((offset_top_diff[pw]/float(argmax)));
        }
    }
    else{
            offset_bottom_diff[wstart]=static_cast<T>((offset_top_diff[pw]/float(argmax)));
    }

  }
}

std::tuple<at::Tensor, at::Tensor> ROIPool_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_width) {
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(0);
  auto width = input.size(1);

  auto output = at::empty({num_rois, channels, pooled_width}, input.options());
  auto output_size = num_rois * pooled_width * channels;
  auto argmax = at::zeros({num_rois, channels,  pooled_width}, input.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(output_size, 512L), 4096L));
  dim3 block(512);

  if (output.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return std::make_tuple(output, argmax);
  }

  AT_DISPATCH_FLOATING_TYPES(input.type(), "ROIPool_forward", [&] {
    RoIPoolFForward<scalar_t><<<grid, block, 0, stream>>>(
         output_size,
         input.contiguous().data<scalar_t>(),
         spatial_scale,
         channels,
         width,
         pooled_width,
         rois.contiguous().data<scalar_t>(),
         output.data<scalar_t>(),
         argmax.data<int>());
  });
  THCudaCheck(cudaGetLastError());
  return std::make_tuple(output, argmax);
}

// TODO remove the dependency on input and use instead its sizes -> save memory
at::Tensor ROIPool_backward_cuda(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 const at::Tensor& rois,
                                 const at::Tensor& argmax,
                                 const float spatial_scale,
                                 const int pooled_width,
                                 const int channels,
                                 const int width) {
  AT_ASSERTM(grad.type().is_cuda(), "grad must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");
  // TODO add more checks

  auto num_rois = rois.size(0);
  auto grad_input = at::zeros({channels, width}, grad.options());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  dim3 grid(std::min(THCCeilDiv(grad.numel(), 512L), 4096L));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    THCudaCheck(cudaGetLastError());
    return grad_input;
  }

  AT_DISPATCH_FLOATING_TYPES(grad.type(), "ROIPool_backward", [&] {
    RoIPoolFBackward<scalar_t><<<grid, block, 0, stream>>>(
         grad.numel(),
         grad.contiguous().data<scalar_t>(),
         argmax.data<int>(),
         num_rois,
         spatial_scale,
         channels,
         width,
         pooled_width,
         grad_input.data<scalar_t>(),
         rois.contiguous().data<scalar_t>());
  });
  THCudaCheck(cudaGetLastError());
  return grad_input;
}
