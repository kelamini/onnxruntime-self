#include "test2.hpp"


// 图像处理  标准化处理
void PreProcess(const Mat& image, Mat& image_blob)
{
	Mat input;
	image.copyTo(input);

	//数据处理 标准化
	std::vector<Mat> channels, channel_p;
	split(input, channels);
	Mat R, G, B;
	B = channels.at(0);
	G = channels.at(1);
	R = channels.at(2);

    // (tensor - mean) / std
    // normalize = (mean=[0.485, 0.456, 0.406],
    //               std=[0.229, 0.224, 0.225])
	B = (B / 255. - 0.406) / 0.225;
	G = (G / 255. - 0.456) / 0.224;
	R = (R / 255. - 0.485) / 0.229;

	channel_p.push_back(R);
	channel_p.push_back(G);
	channel_p.push_back(B);

	Mat outt;
	merge(channel_p, outt);
	image_blob = outt;
}

int main(int argc, char* argv[])
{
    // 模型输入图像的宽高
    int rewidth;
    int reheight;

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
	// const char* model_path = "/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/weights_onnx/pipnet_mv3_wflw98_256x256_sim.onnx";
    const char* model_path = "/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/weights_onnx/scrfd_768_432.onnx";
#endif

    // 创建Session并把模型加载到内存中
    Ort::Session session(env, model_path, session_options);

    //打印模型的输入层(node names, types, shape etc.)
    Ort::AllocatorWithDefaultOptions allocator;

    //输出模型 输入/输出 节点的数量
    size_t num_input_nodes = session.GetInputCount();
    printf("num_input_nodes: %ld\n", num_input_nodes);
    size_t num_output_nodes = session.GetOutputCount();
    printf("num_output_nodes: %ld\n", num_output_nodes);

    // std::vector<const char*> input_node_names = {"input_img_3x256x256"};
    // std::vector<const char*> output_node_names = {"outputs_cls_98x8x8", \
    //                                               "outputs_x_98x8x8", \
    //                                               "outputs_y_98x8x8", \
    //                                               "outputs_nb_x_980x8x8", \
    //                                               "outputs_nb_y_980x8x8"};
    std::vector<const char*> input_node_names = {"input.1"};
    std::vector<const char*> output_node_names = {"score_8", \
                                                  "score_16", \
                                                  "score_32", \
                                                  "bbox_8", \
                                                  "bbox_16", \
                                                  "bbox_32", \
                                                  "kps_8", \
                                                  "kps_16", \
                                                  "kps_32"};
    std::vector<int64_t> input_node_dims;

    bool dynamic_flag = false;

    //打印输入节点信息
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
        //打印输入节点的维度
        input_node_dims = tensor_info.GetShape();
        printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
        //打印各个维度的大小
        for (int j = 0; j < input_node_dims.size(); j++)
        {
            printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
            if (j == 2)
            {
                rewidth = input_node_dims[j];
            }
            if (j == 3)
            {
                reheight = input_node_dims[j];
            }
            if (input_node_dims[j] < 1)
            {
                dynamic_flag  = true;
            }
        }
    }
    //打印输出节点信息
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
        for (int j = 0; j < output_node_dims.size(); j++)
        {
            printf("Output: %d dim %d=%jd\n", i, j, output_node_dims[j]);
        }
    }

    // // 使用样本数据对模型进行评分，并检验出入值的合法性
    // size_t input_tensor_size = 3 * 256 * 256;
    // std::vector<float> input_tensor_values(input_tensor_size);

    // // 初始化一个数据（演示用,这里实际应该传入归一化的数据）
    // for (unsigned int i = 0; i < input_tensor_size; i++)
    //     input_tensor_values[i] = (float)i / (input_tensor_size + 1);

    // // OrtValue * input_val = nullptr;

    // // 为输入数据创建一个Tensor对象
    // std::vector<int64_t> input_shape = {1, 3, 256, 256};
    // auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_shape.data(), 4);
    // assert(input_tensor.IsTensor());

    VideoCapture capture;
    //-- 2. Read the video stream
    capture.open( 0 );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat img;
    while ( capture.read(img) )
    {
        if( img.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        if( waitKey(1) == 27 )
        {
            break; // escape
        }

        //加载图片
        // Mat img = imread("/home/kelamini/workspace/onnxruntime/samples/c++/pipnet/data/demo.jpeg");
        Mat det1, det2;
        resize(img, det1, Size(rewidth, reheight), INTER_AREA);
        det1.convertTo(det1, CV_32FC3);
        PreProcess(det1, det2);         //标准化处理
        Mat blob = dnn::blobFromImage(det2, 1., Size(rewidth, reheight), Scalar(0, 0, 0), true, CV_8U);
        printf("Preprocess success!\n");

        //创建输入tensor
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::vector<Ort::Value> input_tensors;
        input_tensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info, blob.ptr<float>(), blob.total(), input_node_dims.data(), input_node_dims.size()));
        std::cout << "imgs_total(B x C x W x H): " << blob.total() << std::endl;

        clock_t startTime, endTime;
        try
        {
            // 推理得到结果
            startTime = clock();
            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensors.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
            assert(output_tensors.size() == num_output_nodes && output_tensors.front().IsTensor());
            printf("Number of outputs = %ld\n", output_tensors.size());  // 输出的节点数
            endTime = clock();

            // Get pointer to output tensor float values
            for (int i = 0; i < num_output_nodes; i++)
            {
                // float* floatarr = output_tensors[i].GetTensorMutableData<float>();
                std::cout << "output_tensor_size: " << output_tensors[i].GetTensorTypeAndShapeInfo().GetShape().size() << std::endl;   // 输出当前节点的维度大小
                // std::cout << "Output_tensors lens: " << output_tensors[i] << std::endl;
            }
        }

        catch (Ort::Exception& e)
        {
            std::cout << "Error: " << e.what() << std::endl;
        }

        // fps
        double fps = capture.get(CAP_PROP_FPS);
        string fps_str = to_string((int)fps);
        putText(img, "FPS: "+fps_str, Point(50, 50), 2, 1, (255, 0, 0), 1);

        // img_width, img_height
        int height = img.rows;
        string height_str = to_string(height);
        int width = img.cols;
        string width_str = to_string(width);
        putText(img, "(width, height): ("+width_str+", "+height_str+")", Point(50, 80), 2, 1, (255, 0, 0), 1);

        imshow( "DMS", img );

        printf("Proceed exit after %.2f seconds\n", static_cast<float>(endTime - startTime) / CLOCKS_PER_SEC);
        printf("Done!\n");
    }
    return 0;
}
