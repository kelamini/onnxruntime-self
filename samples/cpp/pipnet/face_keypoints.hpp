#include <assert.h>
#include <vector>
#include <iostream>
#include <ctime>

// opencv
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <opencv2/dnn.hpp>

// onnxruntime
// #include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
// #include <cuda_provider_factory.h>

// torch
#include <torch/torch.h>
#include <torch/script.h>

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;


// 自定义配置结构
struct Configuration
{
	public:
	// float confThreshold; // Confidence threshold
	// float nmsThreshold;  // Non-maximum suppression threshold
	// float objThreshold;  //Object Confidence threshold
	string modelpath;
};

// 定义 KeyPointsInfo 结构类型
typedef struct KeyPointsInfo
{
	float x;
	float y;
} KeyPointsInfo;


class PiPNet
{
public:
	PiPNet(Configuration config);
	void detect(Mat& frame);
private:
	// float confThreshold;
	// float nmsThreshold;
	// float objThreshold;
	int num_nb;
	int inpWidth;
	int inpHeight;
	// int nout;
	// int num_proposal;
	// int num_classes;

	const bool keep_ratio = true;
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	vector<torch::Tensor> nms(vector<Value>& ort_outputs, int neww, int newh);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);
	void get_meanface(int num_nb);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "pipnet-98"); // 初始化环境
	Session *ort_session = nullptr;    // 初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象
	//SessionOptions sessionOptions;
	vector<const char*> input_names = {"input_img_3x256x256"};  // 定义一个字符指针vector
	vector<const char*> output_names = {"outputs_cls_98x8x8",
								  "outputs_x_98x8x8",
								  "outputs_y_98x8x8",
								  "outputs_nb_x_980x8x8",
								  "outputs_nb_y_980x8x8"}; // 定义一个字符指针vector
	vector<vector<int64_t>> input_node_dims; // >=1 outputs，二维vector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs，int64_t C/C++标准
};
