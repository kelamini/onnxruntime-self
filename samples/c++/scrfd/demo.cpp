#include "scrfd.hpp"

int main(int argc,char *argv[])
{
	clock_t startTime,endTime;	//计算时间
	Configuration yolo_nets = { 0.3, 0.5, 0.3, "yolov5s.onnx" };
	string imgpath = "bus.jpg";
	YOLOv5 yolo_model(yolo_nets);
	Mat srcimg = imread(imgpath);

	double timeStart = (double)getTickCount();
	startTime = clock();	//计时开始
	yolo_model.detect(srcimg);
	endTime = clock();		//计时结束
	double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
	cout << "clock_running time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    cout << "The run time is:" << (double)clock() /CLOCKS_PER_SEC<< "s" << endl;
	cout << "getTickCount_running time :" << nTime << "sec\n" << endl;
	// static const string kWinName = "Deep learning object detection in ONNXRuntime";
	// namedWindow(kWinName, WINDOW_NORMAL);
	// imshow(kWinName, srcimg);
	// waitKey(0);
	// destroyAllWindows();
	imwrite("restult_ort.jpg",srcimg);

	return 0;
}
