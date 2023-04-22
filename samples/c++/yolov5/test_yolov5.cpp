#include "yolov5.hpp"


static void help(const char** argv)
{
    cout << "\nThis program demonstrates the use of cv::CascadeClassifier class to detect objects (Face + eyes). You can use Haar or LBP features.\n"
            "This classifier can recognize many kinds of rigid objects, once the appropriate classifier is trained.\n"
            "It's most known use is for faces.\n"
            "Usage:\n"
        <<  argv[0]
        <<  "   [--conftd=<confThreshold>] "
            "   [--nmstd=<nmsThreshold>] "
            "   [--objtd=<objThreshold>] "
            "   [--modelpath=<modelpath>]\n\n"
            "example:\n"
        <<  argv[0]
        <<  " --conftd=0.3 --nmstd=0.5 --objtd=0.3 --modelpath=scrfd_768_432.onnx\n\n"
            "During execution:\n"
            "\tHit any key to quit.\n"
            "\tUsing OpenCV version " << CV_VERSION << "\n" << endl;
}

int main(int argc, const char *argv[])
{
    float confThreshold, nmsThreshold, objThreshold;
    string modelpath;

    CommandLineParser parser(argc, argv,
        "{help h||}"
        "{conftd||}"
        "{nmstd||}"
        "{objtd||}"
        "{modelpath||}");
    if (parser.has("help"))
    {
        help(argv);
        return 0;
    }

    confThreshold = parser.get<float>("conftd");
    nmsThreshold = parser.get<float>("nmstd");
    objThreshold = parser.get<float>("objtd");
	modelpath = parser.get<string>("modelpath");

	clock_t startTime, endTime;	//计算时间
	Configuration yolo_nets = { confThreshold, nmsThreshold, objThreshold, modelpath };  // confThreshold, nmsThreshold, objThreshold, modelpath
	YOLOv5 yolo_model(yolo_nets);

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
        putText(srcimg, "FPS: "+fps_str, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);

		double nTime = ((double)getTickCount() - timeStart) / getTickFrequency();
		cout << "clock_running time is:" <<(double)(endTime - startTime) / CLOCKS_PER_SEC << " s" << endl;
		cout << "The run time is:" << (double)clock() /CLOCKS_PER_SEC<< " s" << endl;
		cout << "getTickCount_running time :" << nTime << " sec" << endl;

		static const string kWinName = "Deep learning object detection in ONNXRuntime";
		imshow(kWinName, srcimg);
	}

	return 0;
}
