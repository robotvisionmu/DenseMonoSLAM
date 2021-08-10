#ifndef DEPTHPREDICTION_H_
#define DEPTHPREDICTION_H_

#include <Utils/Resolution.h>
#include <Utils/Stopwatch.h>
#include <Utils/Options.h>
#include <memory>
#include <onnxruntime/core/session/onnxruntime_cxx_api.h>
// /#include <onnxruntime/core/framework/data_types.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <Eigen/Core>

class DepthPrediction {
public:
  DepthPrediction(bool half_float = false);
  ~DepthPrediction();

  void predict(const std::shared_ptr<unsigned char> & rgb);

  std::shared_ptr<unsigned short> depth() { return m_depth; }

private:
  std::shared_ptr<unsigned short> m_depth;
  std::string m_onnx_model_path;

  int device_id = 0;
  size_t  cuda_mem_limit = 65535;
  bool do_copy_in_default_stream = true;
  bool has_user_compute_stream = false;
  void * user_compute_stream = NULL;
  int arena_extend_strategy = 0;

  Ort::Env m_env;
  Ort::SessionOptions m_session_options;
  OrtCUDAProviderOptions m_cuda_provider_options{device_id, OrtCudnnConvAlgoSearch::DEFAULT, cuda_mem_limit, arena_extend_strategy, do_copy_in_default_stream, has_user_compute_stream,user_compute_stream};
  Ort::Session *m_session;

  size_t m_num_input_nodes;
  std::vector<const char *> m_input_node_names;
  std::vector<int64_t> m_input_node_dims;

  size_t m_num_output_nodes;
  std::vector<const char *> m_output_node_names;
  std::vector<int64_t> m_output_node_dims;

  Ort::Value m_input_tensor{nullptr};
  Ort::Value m_output_tensor{nullptr};

  Ort::Value m_input_tensor_f16{nullptr};
  Ort::Value m_output_tensor_f16{nullptr};

  size_t m_input_tensor_size;
  std::vector<float> m_input_tensor_values;
  std::vector<float> m_output_tensor_values;

  bool m_half_float;
  size_t m_input_tensor_size_f16;
  std::vector<Ort::Float16_t> m_input_tensor_values_f16;
  std::vector<Ort::Float16_t> m_output_tensor_values_f16;
};

#endif /*DEPTHPREDICTION_H_*/