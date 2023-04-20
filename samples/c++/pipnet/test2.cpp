#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/dnn.hpp>

#include <assert.h>
#include <vector>
#include <iostream>
#include <ctime>
#include "face_keypoints.hpp"


// 图像处理  标准化处理
void PreProcess(const cv::Mat& image, cv::Mat& image_blob)
{
	cv::Mat input;
	image.copyTo(input);


	//数据处理 标准化
	std::vector<cv::Mat> channels, channel_p;
	split(input, channels);
	cv::Mat R, G, B;
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

	B = (B / 255. - 0.406) / 0.225;
	G = (G / 255. - 0.456) / 0.224;
	R = (R / 255. - 0.485) / 0.229;

	channel_p.push_back(R);
	channel_p.push_back(G);
	channel_p.push_back(B);

	cv::Mat outt;
	merge(channel_p, outt);
	image_blob = outt;
}


int main(int argc, char* argv[])
{
    //记录程序运行时间
    auto start_time = clock();

    //初始化环境，每个进程一个环境
    //环境保留了线程池和其他状态信息
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    //初始化 Session options 选项
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

#ifdef _WIN32
	const wchar_t* model_path = L"/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/weights_onnx/pipnet_mv3_wflw98_256x256_sim.onnx";
    // const char* model_path = "/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/weights_onnx/scrfd_768_432.onnx";
#else
	const char* model_path = "/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/weights_onnx/pipnet_mv3_wflw98_256x256_sim.onnx";
    // const char* model_path = "/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/weights_onnx/scrfd_768_432.onnx";
#endif

    // 创建Session并把模型加载到内存中
    Ort::Session session(env, model_path, session_options);

    //打印模型的输入层(node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    //输出模型 输入/输出 节点的数量
    size_t num_input_nodes = session.GetInputCount();
    printf("num_input_nodes: %d\n", num_input_nodes);
    size_t num_output_nodes = session.GetOutputCount();
    printf("num_output_nodes: %d\n", num_output_nodes);

    // std::vector<const char*> input_node_names(num_input_nodes);
    // std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<const char*> input_node_names = {"input_img_3x256x256"};
    std::vector<const char*> output_node_names = {"outputs_cls_98x8x8", \
                                                  "outputs_x_98x8x8", \
                                                  "outputs_y_98x8x8", \
                                                  "outputs_nb_x_980x8x8", \
                                                  "outputs_nb_y_980x8x8"};
    std::vector<int64_t> input_node_dims;

    bool dynamic_flag = false;

    //迭代所有的输入节点
    for (int i = 0; i < num_input_nodes; i++) {
         //输出输入节点的名称
        Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(i, allocator);
        char* input_name = input_name_Ptr.get();
        printf("Input %d : name=%s\n", i, input_name);
        // input_node_names[i] = input_name;
        // 输出输入节点的类型
        Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Input %d : type=%d\n", i, type);
        input_node_dims = tensor_info.GetShape();
        //输入节点的打印维度
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
            if (input_node_dims[j] < 1)
            {
                dynamic_flag  = true;
            }
        }

        // input_node_dims[0] = 1;
    }
    //打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_name_Ptr = session.GetOutputNameAllocated(i, allocator);
        char* output_name = output_name_Ptr.get();
        printf("Output: %d name=%s\n", i, output_name);
        // output_node_names[i] = output_name;
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output: %d type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output: %d num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
        {
            printf("Output: %d dim %d=%jd\n", i, j, output_node_dims[j]);
        }
    }

    // 使用样本数据对模型进行评分，并检验出入值的合法性
    size_t input_tensor_size = 3 * 256 * 256;
    std::vector<float> input_tensor_values(input_tensor_size);

    // 初始化一个数据（演示用,这里实际应该传入归一化的数据）
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    // OrtValue * input_val = nullptr;

    // 为输入数据创建一个Tensor对象
    std::vector<int64_t> input_shape = {1, 3, 256, 256};
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), 4);
    assert(input_tensor.IsTensor());

    try
    {
        // 推理得到结果
        // std::cout << "input_node_names: " << input_node_names.data() << std::endl;
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, num_input_nodes, output_node_names.data(), num_output_nodes);
        assert(output_tensors.size() == num_output_nodes && output_tensors.front().IsTensor());

        // Get pointer to output tensor float values
        for (int i = 0; i < num_output_nodes; i++)
        {
            float* floatarr = output_tensors[i].GetTensorMutableData<float>();
            printf("Output_tensors value = %f\n", floatarr);
        }

        // // 另一种形式
        // Ort::IoBinding io_binding{session};
        // io_binding.BindInput("input_img_3x256x256", input_tensor);
        // Ort::MemoryInfo output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        // // std::vector<const char*> outputnames = {"outputs_cls_98x8x8", "outputs_x_98x8x8", "outputs_y_98x8x8", "outputs_nb_x_980x8x8", "outputs_nb_y_980x8x8"};
        // io_binding.BindOutput("outputs_cls_98x8x8", output_mem_info);
        // session.Run(Ort::RunOptions{ nullptr }, io_binding);

        printf("Number of outputs = %d\n", output_tensors.size());
    }

    catch (Ort::Exception& e)
    {
        printf(e.what());
    }

    auto end_time = clock();
    printf("Proceed exit after %.2f seconds\n", static_cast<float>(end_time - start_time) / CLOCKS_PER_SEC);
    printf("Done!\n");
    return 0;
}
