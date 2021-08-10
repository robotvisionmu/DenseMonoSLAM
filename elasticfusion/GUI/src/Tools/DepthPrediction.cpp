#include "DepthPrediction.h"

DepthPrediction::DepthPrediction(bool half_float)
    : m_half_float(half_float),
      m_env(ORT_LOGGING_LEVEL_ERROR, "verbose") {
  
  m_onnx_model_path = m_half_float ? "/home/louis/Development/elasticfusion/weights/normnet_float16_opset12.onnx" : "/home/louis/Development/elasticfusion/weights/normnet_float32_opset12.onnx";
  std::cout << "onnx model: " << m_onnx_model_path << std::endl;
  m_session_options.SetIntraOpNumThreads(4);
  m_session_options.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
m_session_options.AppendExecutionProvider_CUDA(m_cuda_provider_options);
  m_session =
      new Ort::Session(m_env, m_onnx_model_path.c_str(), m_session_options);
  m_depth = std::shared_ptr<unsigned short>(
      new unsigned short[Resolution::getInstance().numPixels()]);

  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  m_num_input_nodes = m_session->GetInputCount();
  m_input_node_names = std::vector<const char *>(m_num_input_nodes);

  printf("Number of inputs = %zu\n", m_num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < m_num_input_nodes; i++) {
    // print input node names
    char *input_name = m_session->GetInputName(i, allocator);
    printf("Input %d : name=%s\n", i, input_name);
    m_input_node_names[i] = input_name;

    // print input node types
    Ort::TypeInfo type_info = m_session->GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    m_input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, m_input_node_dims.size());
    for (int j = 0; j < m_input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, m_input_node_dims[j]);
  }

  m_num_output_nodes = m_session->GetOutputCount();
  m_output_node_names = std::vector<const char *>(m_num_output_nodes);

  printf("Number of outputs = %zu\n", m_num_output_nodes);

  // iterate over all input nodes
  for (int i = 0; i < m_num_output_nodes; i++) {
    // print input node names
    char *output_name = m_session->GetOutputName(i, allocator);
    printf("Output %d : name=%s\n", i, output_name);
    m_output_node_names[i] = output_name;

    // print input node types
    Ort::TypeInfo type_info = m_session->GetOutputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Output %d : type=%d\n", i, type);

    // print input shapes/dims
    m_output_node_dims = tensor_info.GetShape();
    printf("Output %d : num_dims=%zu\n", i, m_output_node_dims.size());
    for (int j = 0; j < m_output_node_dims.size(); j++)
      printf("Output %d : dim %d=%jd\n", i, j, m_output_node_dims[j]);
  }

  Ort::MemoryInfo m_memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  m_input_tensor_size = Resolution::getInstance().numPixels() * 3;
  m_input_tensor_values = std::vector<float>(m_input_tensor_size);
  m_output_tensor_values = std::vector<float>(Resolution::getInstance().numPixels());

  m_input_tensor_size_f16 = Resolution::getInstance().numPixels() * 3;
  m_input_tensor_values_f16 = std::vector<Ort::Float16_t>(m_input_tensor_size_f16);
  m_output_tensor_values_f16 = std::vector<Ort::Float16_t>(Resolution::getInstance().numPixels());

  m_input_tensor = Ort::Value::CreateTensor<float>(
      m_memory_info, m_input_tensor_values.data(), m_input_tensor_size,
      m_input_node_dims.data(), m_input_node_dims.size());
  
  m_input_tensor_f16 = Ort::Value::CreateTensor<Ort::Float16_t>(
      m_memory_info, m_input_tensor_values_f16.data(), m_input_tensor_size_f16,
      m_input_node_dims.data(), m_input_node_dims.size());
  
  printf("\n");
  m_output_tensor = Ort::Value::CreateTensor<float>(
      m_memory_info, m_output_tensor_values.data(),Resolution::getInstance().numPixels(), m_output_node_dims.data(),
      m_output_node_dims.size());
  
  m_output_tensor_f16 = Ort::Value::CreateTensor<Ort::Float16_t>(
      m_memory_info, m_output_tensor_values_f16.data(),Resolution::getInstance().numPixels(), m_output_node_dims.data(),
      m_output_node_dims.size());

    unsigned char * img = new unsigned char[Resolution::getInstance().numPixels() * 3];
    std::shared_ptr<unsigned char> rgb(img);
    predict(rgb);
}
DepthPrediction::~DepthPrediction() { delete m_session; }

void DepthPrediction::predict(const std::shared_ptr<unsigned char> &rgb) {
  TICK("DepthPredict::Load");
  cv::Mat im = cv::Mat(Resolution::getInstance().rows(),
                       Resolution::getInstance().cols(), CV_8UC3,
                       rgb.get());
  cv::Mat im_f;
  im.convertTo(im_f, CV_32FC3, 1.0/255.0);
  
  cv::Mat split_rgb[3];
  cv::split(im_f, split_rgb);

  memcpy(m_input_tensor_values.data(), split_rgb[0].data, Resolution::getInstance().numPixels()  *sizeof(float));
  
  memcpy(m_input_tensor_values.data() + (1 * (Resolution::getInstance().numPixels())), split_rgb[1].data, Resolution::getInstance().numPixels()  *sizeof(float));
  
  memcpy(m_input_tensor_values.data() + (2 * (Resolution::getInstance().numPixels())), split_rgb[2].data, Resolution::getInstance().numPixels()  *sizeof(float));
  TOCK("DepthPredict::Load");
  
  if(m_half_float)
  {
    for(int i = 0; i < m_input_tensor_values.size(); i++)
    {
      m_input_tensor_values_f16[i] = Eigen::half_impl::float_to_half_rtne(m_input_tensor_values[i]).x;
    }

  }

  
  TICK("DepthPredict::inference");
  if(!m_half_float)
  {
    m_session->Run(
    Ort::RunOptions{nullptr}, m_input_node_names.data(), &m_input_tensor, 1,
    m_output_node_names.data(), &m_output_tensor, 1);
  }
  else
  {
      m_session->Run(
      Ort::RunOptions{nullptr}, m_input_node_names.data(), &m_input_tensor_f16, 1,
      m_output_node_names.data(), &m_output_tensor_f16, 1);
  }
  TOCK("DepthPredict::inference");
  
  TICK("DepthPredict::Unload");

  if(m_half_float)
  {
    memcpy(m_output_tensor_values_f16.data(), m_output_tensor_f16.GetTensorMutableData<Ort::Float16_t>(), 
    m_output_tensor_values_f16.size() * 2);
    for(int i = 0; i < m_output_tensor_values_f16.size(); i++)
    {
        Ort::Float16_t hf = m_output_tensor_f16.GetTensorMutableData<Ort::Float16_t>()[i];//m_output_tensor_values_f16[i];
        m_output_tensor_values[i] = Eigen::half_impl::half_to_float(Eigen::half_impl::raw_uint16_to_half(hf));
   }
  }

  cv::Mat im_d = cv::Mat(Resolution::getInstance().rows(),
                    Resolution::getInstance().cols(), CV_32F,
                    m_half_float ? m_output_tensor_values.data() : m_output_tensor.GetTensorMutableData<float>());
  cv::Mat im_d_s;
  im_d.convertTo(im_d_s, CV_16UC1, 1000.0);
  memcpy(m_depth.get(), im_d_s.data, Resolution::getInstance().numPixels() * 2);
  TOCK("DepthPredict::Unload");
}
