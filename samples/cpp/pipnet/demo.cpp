#include "face_keypoints.hpp"


int main(int argc,char *argv[])
{
	string modelpath = "/home/kelamini/workspace/onnxruntime/samples/cpp/pipnet/weights_onnx/pipnet_mv3_wflw98_256x256_sim.onnx";

	clock_t startTime, endTime;	//计算时间
	Configuration yolo_nets = { modelpath };
	PiPNet yolo_model(yolo_nets);

    VideoCapture capture;
    capture.open( 0 );
    if ( ! capture.isOpened() )
    {
        cout << "--(!)Error opening video capture\n";
        return -1;
    }

    Mat srcimg;
    while ( capture.read(srcimg) )
    {
        if( srcimg.empty() )
        {
            cout << "--(!) No captured frame -- Break!\n";
            break;
        }

        if( waitKey(1) == 27 )
        {
            break; // escape
        }

		double timeStart = (double)getTickCount();
		startTime = clock();	//计时开始
		yolo_model.detect(srcimg);
		endTime = clock();		//计时结束

		// fps
        double fps = capture.get(CAP_PROP_FPS);
        string fps_str = to_string((int)fps);
        putText(srcimg, "FPS: "+fps_str, Point(50, 50), 2, 1, Scalar(255, 0, 0), 1);

		double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
		// cout << "clock_running time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << " s" << endl;
		// cout << "The run time is:" << (double)clock() /CLOCKS_PER_SEC<< " s" << endl;
		// cout << "getTickCount_running time :" << nTime << " sec" << endl;

		static const string kWinName = "Deep learning object detection in ONNXRuntime";
		imshow(kWinName, srcimg);
	}

	return 0;
}
