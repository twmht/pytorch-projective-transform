#include <algorithm>
#include <vector>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>
#include <glog/logging.h>

///////////////////////////////////////////////////////////////////

const int CAFFE_CUDA_NUM_THREADS = 512;

#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}


template <typename Dtype>
__global__ void forward_projective(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* in, const Dtype* theta, Dtype* source_data, Dtype* out) {//, const Dtype* fill_value_) {

    const int map_size = output_H_ * output_W_;
    
    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

        int offset = 8 * n;
		Dtype z = 1 / (x_target * theta[offset + 6] + y_target * theta[offset + 7] + 1);
        Dtype x = x_target * theta[offset] + y_target * theta[offset + 1] + theta[offset + 2];
        Dtype y = x_target * theta[offset + 3] + y_target * theta[offset + 4] + theta[offset + 5];

		/*offset = (n * map_size + h * output_W_ + w) * 3;
		source_data[offset] = (x *= z);
		source_data[offset + 1] = (y *= z);
		source_data[offset + 2] = z;*/

		x = (x * z + (Dtype) 1.) * (width_ - 1) / 2;
        y = (y * z + (Dtype) 1.) * (height_ - 1) / 2;
		offset = (n * map_size + h * output_W_ + w) * 3;
		source_data[offset] = x;
		source_data[offset + 1] = y;
		source_data[offset + 2] = z;

		x = x > 0 ? x : 0; x = x < (width_ - 1) ? x : width_ - 1;
		y = y > 0 ? y : 0; y = y < (height_ - 1) ? y : height_ - 1;
		int w_min = (int)floor(x);
		int w_max = (int)ceil(x);
		int h_min = (int)floor(y);
		int h_max = (int)ceil(y);
		for (int c = 0; c < channels_; ++c) {
			Dtype r = 0;
			offset = (n * channels_ + c) * height_ * width_;
			for (int hh = h_min; hh <= h_max; ++hh) {
				const Dtype dy = (1 - fabs(y - hh));
				for (int ww = w_min; ww <= w_max; ++ww) {
					r += in[offset + hh * width_ + ww] * (1 - fabs(x - ww)) * dy;
				}
			}
			out[(n * channels_ + c) * map_size + h * output_W_ + w] = r;
		}
  }
}


std::vector<torch::Tensor> stn_cuda_forward(torch::Tensor input, torch::Tensor weights, int output_h, int output_w) {

  const float* bottom_data = input.data_ptr<float>();
  const auto batch_size = input.size(0);
  const auto channel = input.size(1);
  const auto height = input.size(2);
  const auto width = input.size(3);

  auto options = torch::TensorOptions()
    .dtype(torch::kFloat32);
  torch::Tensor output = torch::zeros({batch_size, channel, output_h, output_w}, options).to(input.device());
  torch::Tensor source = torch::zeros({batch_size, 3, output_h, output_w}, options).to(input.device());

  float* top_data = output.data_ptr<float>();
  float* source_data = source.data_ptr<float>();

  const float* theta_data = weights.data_ptr<float>();
  const int count = batch_size * output_h * output_w;

  forward_projective<float> << <CAFFE_GET_BLOCKS(count),
    CAFFE_CUDA_NUM_THREADS >> >(count, channel, height, width, output_h, output_w,
    bottom_data, theta_data, source_data, top_data);// , fill_value_.gpu_data());

  return std::vector<torch::Tensor>({output, source});

	
	// switch (t_type_) {
	// case 0:
		// // affine
		// forward_affine<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, theta_data, source_data, top_data);//, fill_value_.gpu_data());
		// break;
		
	// case 1:
		// // translation
		// forward_translation<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, theta_data, source_data, top_data,
			// this->layer_param_.st_param().theta_1_1(), this->layer_param_.st_param().theta_2_2());// , fill_value_.gpu_data());
		// break;

	// case 2:
		// // translation + scaling
		// forward_translation_scaling<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, theta_data, source_data, top_data);// , fill_value_.gpu_data());
		// break;

	// case 3:
		// // projective
		// forward_projective<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, theta_data, source_data, top_data);// , fill_value_.gpu_data());
		// break;

	// case 5:
		// // similarity
		// forward_similarity<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, theta_data, source_data, top_data);//, fill_value_.gpu_data());
		// break;

	// case 6:
		// // similarity+
		// forward_similarity_plus<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, theta_data, source_data, top_data);//, fill_value_.gpu_data());
		// break;
	// }
    
    // CUDA_POST_KERNEL_CHECK;
}


///////////////////////////////////////////////////////////////////

template <typename Dtype>
__global__ void backward_projective(const int count, const int channels_,
        const int height_, const int width_, const int output_H_, const int output_W_,
        const Dtype* data, const Dtype* source_data, const Dtype* top_diff,
		Dtype* data_diff, Dtype* theta_diff_cache) {
    
	const int map_size = output_H_ * output_W_;
	//const Dtype width_const = (Dtype)2 / (Dtype)(width_ - 1);
	//const Dtype height_const = (Dtype)2 / (Dtype)(height_ - 1);
	const Dtype width_const = (Dtype)(width_ - 1) / 2;
	const Dtype height_const = (Dtype)(height_ - 1) / 2;

    CUDA_KERNEL_LOOP(index, count) {
        int n = index / map_size;
        int n_rem = index % map_size;
        int h = n_rem / output_W_;
        int w = n_rem % output_W_;

        int offset = (n * map_size + h * output_W_ + w) * 3;
        Dtype x = source_data[offset];
        Dtype y = source_data[offset + 1];
		Dtype z = source_data[offset + 2];

		//Dtype x = (x0 + (Dtype) 1.) * (width_ - 1) / 2;
		//Dtype y = (y0 + (Dtype) 1.) * (height_ - 1) / 2;
		Dtype x0 = x - width_const, y0 = y - height_const;

		int w_min = (floor(x) > 0) ? floor(x) : 0;
        int w_max = (ceil(x) < width_ - 1) ? ceil(x) : (width_ - 1);
        int h_min = (floor(y) > 0) ? floor(y) : 0;
        int h_max = (ceil(y) < height_ - 1) ? ceil(y) : (height_ - 1);
        
		Dtype dv_dx = 0;
        Dtype dv_dy = 0;
		Dtype tmp_source_z = 0;

		for (int hh = h_min; hh <= h_max; ++hh) {
			int sign_y = (Dtype(0) <= Dtype(hh - y)) - (Dtype(hh - y) < Dtype(0));
			Dtype dy = 1 - fabs(y - hh);
			for (int ww = w_min; ww <= w_max; ++ww) {
				int sign_x = (Dtype(0) <= Dtype(ww - x)) - (Dtype(ww - x) < Dtype(0));
				Dtype dx = 1 - fabs(x - ww);

				for (int c = 0; c < channels_; ++c) {
					Dtype u = 
						top_diff[(n * channels_ + c) * map_size + h * output_W_ + w] *
						data[((n * channels_ + c) * height_ + hh) * width_ + ww];
					
					Dtype dv_dx_i = u * dy * sign_x;
					Dtype dv_dy_i = u * dx * sign_y;
					dv_dx += dv_dx_i;
					dv_dy += dv_dy_i;
					tmp_source_z -= dv_dx_i * x0 + dv_dy_i * y0;
				}
			}
		}
		
		dv_dx *= width_const * z;
		dv_dy *= height_const * z;
		tmp_source_z *= z;

		Dtype x_target = (Dtype) w / (output_W_-1) * 2 - (Dtype)1.;
        Dtype y_target = (Dtype) h / (output_H_-1) * 2 - (Dtype)1.;

		n = n * 8 * map_size + h * output_W_ + w;
		theta_diff_cache[n] = dv_dx * x_target;
		theta_diff_cache[n + map_size] = dv_dx * y_target;
		theta_diff_cache[n + map_size*2] = dv_dx;
		theta_diff_cache[n + map_size*3] = dv_dy * x_target;
		theta_diff_cache[n + map_size*4] = dv_dy * y_target;
		theta_diff_cache[n + map_size*5] = dv_dy;
		theta_diff_cache[n + map_size*6] = tmp_source_z * x_target;
		theta_diff_cache[n + map_size*7] = tmp_source_z * y_target;
    }
}


std::vector<torch::Tensor> stn_cuda_backward(torch::Tensor input, torch::Tensor source, torch::Tensor grad_input) {
    const float* bottom_data = input.data_ptr<float>();
    const float* top_diff = grad_input.data_ptr<float>();

    // float* theta_diff = grad_weight.data_ptr<float>();

    int batch_size = grad_input.size(0);
    int channel = grad_input.size(1);
    int output_h = grad_input.size(2);
    int output_w = grad_input.size(3);

    int height = input.size(2);
    int width = input.size(3);

    int count = batch_size * output_h * output_w;

    auto options = torch::TensorOptions()
      .dtype(torch::kFloat32);
    torch::Tensor theta_diff_cache = torch::zeros({batch_size, 8, output_h, output_w}, options).to(input.device());
    float *theta_diff_cache_data = theta_diff_cache.data_ptr<float>();

    torch::Tensor theta_diff_op = torch::ones({batch_size, output_h * output_w, 1}, options).to(input.device());

    const float* source_data = source.data_ptr<float>();

    torch::Tensor grad_old_input = torch::zeros_like(input);
    float* data_diff = grad_old_input.data_ptr<float>();

    backward_projective<float> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
      count, channel, height, width, output_h, output_w,
      bottom_data, source_data, top_diff, data_diff, theta_diff_cache_data);

    torch::Tensor tmp_old_theta_diff = theta_diff_cache.view({batch_size, 8, output_h * output_w});
    torch::Tensor grad_old_theta = tmp_old_theta_diff.bmm(theta_diff_op).squeeze(2);
    return std::vector<torch::Tensor>({grad_old_input, grad_old_theta});




    // caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 8, 1, map_size_,
      // Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);


	// switch (t_type_) {
	// case 0:
		// // affine
		
		// // compute gradient with respect to theta
		// backward_affine<Dtype> << <CAFFE_GET_BLOCKS(count),
			// CAFFE_CUDA_NUM_THREADS >> >(count, channels_,
			// height_, width_, output_H_, output_W_,
			// bottom_data, source_data, top_diff,		// input
			// data_diff, theta_diff_cache				// output
			// );

		// // aggregate gradient for theta 
		// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 6, 1, map_size_,
			// Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		// break;
		
	// case 1:
		// // translation
		// backward_translation<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			// count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, source_data, top_diff, data_diff, theta_diff_cache);

		// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 2, 1, map_size_,
			// Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		// break;

	// case 2:
		// // translation + scaling
		// backward_translation_scaling<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			// count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, source_data, top_diff, data_diff, theta_diff_cache);

		// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 4, 1, map_size_,
			// Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		// break;

	// case 3:
		// // projective
		// backward_projective<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			// count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, source_data, top_diff, data_diff, theta_diff_cache);

		// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 8, 1, map_size_,
			// Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		// break;

	// case 5:
		// // similarity
		// backward_similarity<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			// count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, source_data, top_diff, bottom[1]->gpu_data(),
			// data_diff, theta_diff_cache);

		// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 4, 1, map_size_,
			// Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		// break;

	// case 6:
		// // similarity+
		// backward_similarity_plus<Dtype> << <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS >> >(
			// count, channels_, height_, width_, output_H_, output_W_,
			// bottom_data, source_data, top_diff, bottom[1]->gpu_data(),
			// data_diff, theta_diff_cache);

		// caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, num_ * 4, 1, map_size_,
			// Dtype(1), theta_diff_cache, theta_diff_op_.gpu_data(), Dtype(0), theta_diff);
		// break;
	// }
	
    // CUDA_POST_KERNEL_CHECK;
}
