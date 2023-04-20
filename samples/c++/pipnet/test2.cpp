#include <assert.h>
#include <vector>
#include <ctime>
#include "face_keypoints.hpp"

int main(int argc, char* argv[])
{
    //记录程序运行时间
    auto start_time = clock();
    //初始化环境，每个进程一个环境
    //环境保留了线程池和其他状态信息
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    //初始化Session选项
    Ort::SessionOptions session_options;

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

    // 创建Session并把模型加载到内存中
    const char* model_path = "test.onnx";

    Ort::Session session(env, model_path, session_options);

    //打印模型的输入层(node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    //输出模型输入节点的数量
    size_t num_input_nodes = session.GetInputCount();
    size_t num_output_nodes = session.GetOutputCount();
    std::vector<const char*> input_node_names(num_input_nodes);
    std::vector<const char*> output_node_names(num_output_nodes);
    std::vector<int64_t> input_node_dims;

    bool dynamic_flag = false;





    std::vector<char*> input_name;
    std::vector<char*> output_name;
    //迭代所有的输入节点
    for (int i = 0; i < num_input_nodes; i++) {
         //输出输入节点的名称
        Ort::AllocatedStringPtr input_name_Ptr = session.GetInputNameAllocated(i, allocator);
        input_name.push_back(input_name_Ptr.get());
        printf("Input %d : name=%s\n", i, input_name);
        input_node_names[i] = input_name_Ptr.get();

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

        input_node_dims[0] = 1;
    }
    //打印输出节点信息，方法类似
    for (int i = 0; i < num_output_nodes; i++)
    {
        Ort::AllocatedStringPtr output_name_Ptr= session.GetInputNameAllocated(i, allocator);
        output_name.push_back(output_name_Ptr.get());
        printf("Output: %d name=%s\n", i, output_name);
        output_node_names[i] = output_name_Ptr.get();
        Ort::TypeInfo type_info = session.GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        printf("Output %d : type=%d\n", i, type);
        auto output_node_dims = tensor_info.GetShape();
        printf("Output %d : num_dims=%zu\n", i, output_node_dims.size());
        for (int j = 0; j < input_node_dims.size(); j++)
            printf("Output %d : dim %d=%jd\n", i, j, output_node_dims[j]);
    }

    // 使用样本数据对模型进行评分，并检验出入值的合法性
    size_t input_tensor_size = 3 * 320 * 320;

    std::vector<float> input_tensor_values(input_tensor_size);

    // 初始化一个数据（演示用,这里实际应该传入归一化的数据）
    for (unsigned int i = 0; i < input_tensor_size; i++)
        input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    // 为输入数据创建一个Tensor对象
    try
    {
        OrtValue * input_val = nullptr;

        std::vector<int64_t> input_shape = {1, 3, 320, 320};

        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), 4);
        //assert(input_tensor.IsTensor());

        // 推理得到结果
        auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
        assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

        // Get pointer to output tensor float values
        float* floatarr = output_tensors.front().GetTensorMutableData<float>();


        // 另一种形式
        Ort::IoBinding io_binding{session};
        io_binding.BindInput("img", input_tensor);
        Ort::MemoryInfo output_mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        io_binding.BindOutput("mask", output_mem_info);
        session.Run(Ort::RunOptions{ nullptr }, io_binding);

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
