#include "main.h"
#include "onnxruntime_cxx_api.h"
#include "cpu_provider_factory.h"  // cpu provider
#include <fstream>
#include <assert.h>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <algorithm>  // std::generate, std::max_element



// pretty prints a shape dimension vector
std::string print_shape(const std::vector<std::int64_t>& v) {
    std::stringstream ss("");
    for (std::size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
    ss << v[v.size() - 1];
    return ss.str();
}


// 用于计算 vector 元素个数
int calculate_product(const std::vector<int64_t>& v) {
    int total = 1;
    for (auto& i : v) total *= i;
    return total;
}

std::vector<std::string> readTextLinesFromFile(const std::string& filePath) {
    std::vector<std::string> lines;
    std::ifstream inputFile(filePath);

    if (!inputFile.is_open()) {
        std::cerr << "Error: Could not open file " << filePath << std::endl;
        return lines; // Empty vector if the file cannot be opened
    }

    std::string line;
    while (std::getline(inputFile, line)) {
        lines.push_back(line);
    }

    inputFile.close();
    return lines;
}


std::vector<cv::Mat> preprocess_img_batch(const std::vector<cv::Mat>& input_imgs_bgr, const std::vector<std::int64_t>& input_node_dims) {

    int height = input_node_dims[2];
    int width = input_node_dims[3];

    std::vector<cv::Mat> processed_imgs;
    for (size_t i = 0; i < input_imgs_bgr.size(); ++i) {
        cv::Mat img_bgr = input_imgs_bgr[i];
        cv::Mat img_rgb, resized_img_rgb, processed_img;
        // Convert image format if needed (e.g., from BGR to RGB)
        cv::cvtColor(img_bgr, img_rgb, cv::COLOR_BGR2RGB);
        cv::resize(img_rgb, resized_img_rgb, cv::Size(width, height));
        // 转成 float32, 归一化到 0-1
        resized_img_rgb.convertTo(resized_img_rgb, CV_32F, 1.0 / 255);
        // 减均值除以方差
        cv::subtract(resized_img_rgb, cv::Scalar(0.485, 0.456, 0.406), resized_img_rgb);
        cv::divide(resized_img_rgb, cv::Scalar(0.229, 0.224, 0.225), resized_img_rgb);
        // HWC -> CHW
        processed_img = cv::dnn::blobFromImage(resized_img_rgb);
        processed_imgs.emplace_back(processed_img);
    }

    return processed_imgs;
}

template <typename T>
Ort::Value vec_to_tensor(std::vector<T>& data, const std::vector<std::int64_t>& shape) {
    Ort::MemoryInfo mem_info =
        Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    auto tensor = Ort::Value::CreateTensor<T>(mem_info, data.data(), data.size(), shape.data(), shape.size());
    return tensor;
}


// https://github.com/microsoft/onnxruntime-inference-examples/blob/main/c_cxx/model-explorer/batch-model-explorer.cpp
int main()
{
    int device_id = 0;
    int batch_size = 2;
    
    // onnxruntime 模型路径需要是宽字符 wstring
    std::string model_path = "model/resnet50.onnx";
    std::wstring w_model_path = std::wstring(model_path.begin(), model_path.end());

    // 读取 ImageNet 标签
    std::string label_path = "input/imagenet_classes.txt";
    std::vector<std::string> labels = readTextLinesFromFile(label_path);
    
    // Read multiple input images (e.g., for a batch of size 4)
    std::vector<cv::Mat> img_batch;
    for (int i = 0; i < batch_size; ++i) {
        std::string img_path = "input/n01491361_tiger_shark.JPEG";  // Change this to your image path
        cv::Mat img_bgr = cv::imread(img_path, cv::ImreadModes::IMREAD_COLOR);
        img_batch.push_back(img_bgr);
    }

    // ----------------------------------------------------------- 
    // 2. onnxruntime 初始化
    
    Ort::SessionOptions session_options;
    // 设置 logging level 为 ERROR
    Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "torch-onnx");
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    std::cout << "onnxruntime inference try to use GPU Device: " << device_id << std::endl;
    auto status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, device_id);
    // auto status = OrtSessionOptionsAppendExecutionProvider_CPU(session_options, device_id);
    Ort::Session session(env, w_model_path.c_str(), session_options);

    // ----------------------------------------------------------- 
    // 3. 从模型中读取输入和输入信息
    
    // print (name/shape) of inputs
    Ort::AllocatorWithDefaultOptions allocator;
    std::vector<std::string> input_names;
    std::vector<std::int64_t> input_shapes;
    std::cout << "Input Node Name/Shape (" << input_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetInputCount(); i++) {
        input_names.emplace_back(session.GetInputNameAllocated(i, allocator).get());
        input_shapes = session.GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << input_names.at(i) << " : " << print_shape(input_shapes) << std::endl;
    }

    // print (name/shape) of outputs
    std::vector<std::string> output_names;
    std::vector<std::int64_t> output_shapes;
    std::cout << "Output Node Name/Shape (" << output_names.size() << "):" << std::endl;
    for (std::size_t i = 0; i < session.GetOutputCount(); i++) {
        output_names.emplace_back(session.GetOutputNameAllocated(i, allocator).get());
        output_shapes = session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "\t" << output_names.at(i) << " : " << print_shape(output_shapes) << std::endl;
    }

    // Assume model has 1 input node and 1 output node.
    assert(input_names.size() == 1 && output_names.size() == 1);

    // ----------------------------------------------------------- 
    // 4. 构造输入 tensor

    auto input_node_dims = input_shapes;
    assert(input_node_dims[0] == -1);  // symbolic dimensions are represented by a -1 value
    input_node_dims[0] = batch_size;
    auto total_number_elements = calculate_product(input_node_dims);
    auto processed_imgs = preprocess_img_batch(img_batch, input_node_dims);

    std::vector<float> input_tensor_values(total_number_elements);

    // 将 processed_img 复制到 input_tensor_values
    for (int64_t i = 0; i < batch_size; ++i)
    {
        std::copy(processed_imgs.at(i).begin<float>(),
            processed_imgs.at(i).end<float>(),
            input_tensor_values.begin() + i * total_number_elements / batch_size);
    }

    std::vector<Ort::Value> input_tensors;
    input_tensors.emplace_back(vec_to_tensor<float>(input_tensor_values, input_node_dims));

    // double-check the dimensions of the input tensor
    assert(input_tensors[0].IsTensor() &&
        input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_node_dims);
    std::cout << "input_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << std::endl;

    std::vector<const char*> input_names_char(input_names.size(), nullptr);
    std::transform(std::begin(input_names), std::end(input_names), std::begin(input_names_char),
        [&](const std::string& str) { return str.c_str(); });

    std::vector<const char*> output_names_char(output_names.size(), nullptr);
    std::transform(std::begin(output_names), std::end(output_names), std::begin(output_names_char),
        [&](const std::string& str) { return str.c_str(); });


    // ----------------------------------------------------------- 
    // 5. 推理

    std::cout << "Running model..." << std::endl;

    try {

        auto output_tensors = session.Run(Ort::RunOptions{ nullptr },
            input_names_char.data(),
            input_tensors.data(),
            input_names_char.size(),
            output_names_char.data(),
            output_names_char.size());

        std::cout << "Done inference!" << std::endl;

        // double-check the dimensions of the output tensors
        // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
        assert(output_tensors.size() == output_names.size() && output_tensors[0].IsTensor());

        // 遍历读取每一个输出节点, 可以处理 tensor 和 sequence 的情况
        for (std::size_t i = 0; i < output_names.size(); i++) {
            // GetONNXType(): [UNKNOWN, TENSOR, SEQUENCE, MAP, OPAQUE, SPARSETENSOR, OPTIONAL]
            if (output_tensors[i].GetTypeInfo().GetONNXType() == 1) {
                // 输出是 tensor 的情况
                const float* output_data_ptr = output_tensors[i].GetTensorMutableData<float>();
                int num_classes = output_shapes[1];

                // Process output for each image in the batch
                for (int j = 0; j < batch_size; ++j) {

                    // Extract output data for a particular image in the batch
                    // static_cast<std::ptrdiff_t>(j) 防止算术溢出
                    const float* img_output = output_data_ptr + static_cast<std::ptrdiff_t>(j) * num_classes;

                    // Find the index of the class with the highest probability for the j-th image
                    int max_class_index = std::distance(img_output, std::max_element(img_output, img_output + num_classes));
                    float max_class_prob = img_output[max_class_index];

                    // Print the class ID and corresponding class name for the j-th image in the batch
                    std::cout << "Image " << j + 1 << " - Predicted class ID: " << max_class_index
                        << " | Predicted class name: " << labels[max_class_index]
                        << " | Probability: " << max_class_prob << std::endl;
                }
            }
            else {
                std::cout << "Encountered new ONNXType: " << output_tensors[i].GetTypeInfo().GetONNXType() << std::endl;
            }
        }

    }
    catch (const Ort::Exception& exception) {
        std::cout << "ERROR running model inference: " << exception.what() << std::endl;
        exit(-1);
    }
}
