#include <fstream>
#include <sstream>
#include <iostream>

// #include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// #include <cuda_provider_factory.h>   // 提供cuda加速
#include <onnxruntime_cxx_api.h>	 // C或c++的api

// 命名空间
using namespace std;
using namespace cv;
using namespace Ort;

// 自定义配置结构
struct Configuration
{
	public:
	float confThreshold; // Confidence threshold
	float nmsThreshold;  // Non-maximum suppression threshold
	// float objThreshold;  //Object Confidence threshold
	string modelpath;
};

// 定义BoxInfo结构类型
typedef struct BoxInfo
{
	float x1;
	float y1;
	float x2;
	float y2;
	float score;
	// int label;
} BoxInfo;

// int endsWith(string s, string sub) {
// 	return s.rfind(sub) == (s.length() - sub.length()) ? 1 : 0;
// }

// const float anchors_640[3][6] = { {10.0,  13.0, 16.0,  30.0,  33.0,  23.0},
// 								 {30.0,  61.0, 62.0,  45.0,  59.0,  119.0},
// 								 {116.0, 90.0, 156.0, 198.0, 373.0, 326.0} };

// const float anchors_1280[4][6] = { {19, 27, 44, 40, 38, 94},{96, 68, 86, 152, 180, 137},{140, 301, 303, 264, 238, 542},
// 					   {436, 615, 739, 380, 925, 792} };

class YOLOv5
{
public:
	YOLOv5(Configuration config);
	void detect(Mat& frame);
private:
	float confThreshold;
	float nmsThreshold;
	// float objThreshold;
	int inpWidth;
	int inpHeight;
	// vector<int64_t> nout;
	// vector<int64_t> num_proposal;
	// int num_classes;
	// string classes[1] = {"face"};

	const bool keep_ratio = true;
	vector<float> input_image_;		// 输入图片
	void normalize_(Mat img);		// 归一化函数
	void nms(vector<BoxInfo>& input_boxes);
	Mat resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left);

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "scrfd"); // 初始化环境
	Session *ort_session = nullptr;    // 初始化Session指针选项
	SessionOptions sessionOptions = SessionOptions();  //初始化Session对象
	//SessionOptions sessionOptions;
	vector<const char*> input_names = {"input.1"};  // 定义一个字符指针vector
	vector<const char*> output_names = {"score_8",
										"score_16",
										"score_32",
										"bbox_8",
										"bbox_16",
										"bbox_32",
										"kps_8",
										"kps_16",
										"kps_32",}; // 定义一个字符指针vector
	size_t numInputNodes;
	size_t numOutputNodes;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs，二维vector
	vector<vector<int64_t>> output_node_dims; // >=1 outputs，int64_t C/C++标准
};
