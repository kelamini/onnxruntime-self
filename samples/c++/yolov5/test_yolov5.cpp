#include "yolov5.hpp"


int main(int argc,char *argv[])
{
	string onnxpath = "/home/kelamini/workspace/onnxruntime/samples/c++/yolov5/weights_onnx/yolov5s.onnx";
	string imgpath = "/home/kelamini/workspace/onnxruntime/samples/c++/yolov5/data/demo.jpeg";
	string outputpath = "/home/kelamini/workspace/onnxruntime/samples/c++/yolov5/rusults/demo_results.jpg";

	clock_t startTime, endTime;	//计算时间
	Configuration yolo_nets = { 0.3, 0.5, 0.3, onnxpath };
	YOLOv5 yolo_model(yolo_nets);
	Mat srcimg = imread(imgpath);

	double timeStart = (double)getTickCount();
	startTime = clock();	//计时开始
	yolo_model.detect(srcimg);
	endTime = clock();		//计时结束
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "clock_running time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "The run time is:" << (double)clock() /CLOCKS_PER_SEC<< "s" << endl;
	cout << "getTickCount_running time :" << nTime << "sec" << endl;
	static const string kWinName = "Deep learning object detection in ONNXRuntime";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, srcimg);
	waitKey(0);
	destroyAllWindows();
	imwrite(outputpath, srcimg);

	return 0;
}
