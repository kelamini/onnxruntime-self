#include "scrfd.hpp"


YOLOv5::YOLOv5(Configuration config)
{
	this->confThreshold = config.confThreshold;
	this->nmsThreshold = config.nmsThreshold;
	// this->objThreshold = config.objThreshold;
	// this->num_classes = sizeof(this->classes)/sizeof(this->classes[0]);  // 类别数量
	this->inpHeight = 768;
	this->inpWidth = 432;

	string model_path = config.modelpath;
	//std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  //用于UTF-16编码的字符

	//gpu, https://blog.csdn.net/weixin_44684139/article/details/123504222
	//CUDA加速开启
    // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

	sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);  //设置图优化类型

	// 创建会话，把模型加载到内存中
	//ort_session = new Session(env, widestr.c_str(), sessionOptions);
	//ort_session = new Session(env, (const ORTCHAR_T*)model_path.c_str(), sessionOptions);
	ort_session = new Session(env, (const char*)model_path.c_str(), sessionOptions);

	//输入输出节点数量
	this->numInputNodes = ort_session->GetInputCount();
	this->numOutputNodes = ort_session->GetOutputCount();

	// 配置输入输出节点内存
	// AllocatorWithDefaultOptions allocator;

	// 输入节点
	for (int i = 0; i < this->numInputNodes; i++)
	{
		// input_names.push_back(ort_session->GetInputNameAllocated(i, allocator).get());	// 内存
		Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);   // 类型
		auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();  //
		auto input_dims = input_tensor_info.GetShape();    // 输入shape
		this->input_node_dims.push_back(input_dims);	// 保存
	}
	this->inpHeight = this->input_node_dims[0][2];
	this->inpWidth = this->input_node_dims[0][3];
	// std::cout << "inpHeight, inpWidth" << this->inpHeight << ", " << this->inpWidth << std::endl;

	// 输出节点
	for (int i = 0; i < this->numOutputNodes; i++)
	{
		// output_names.push_back(ort_session->GetOutputNameAllocated(i, allocator).get());
		Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
		auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
		auto output_dims = output_tensor_info.GetShape();
		this->output_node_dims.push_back(output_dims);

		// if (i%3 == 0)
		// {
		// 	this->nout.push_back(this->output_node_dims[i][1]);	// score + (x, y, w, h) + kps
		// }
		// if (i/3 < 1)
		// {
		// 	this->num_proposal.push_back(this->output_node_dims[i][2]*output_node_dims[i][3]);  // pre_box
		// }
	}
	// std::cout << " this->nout: " << this->nout[0] << std::endl;
	// std::cout << " this->num_proposal: " << this->num_proposal[0] << std::endl;
}

Mat YOLOv5::resize_image(Mat srcimg, int *newh, int *neww, int *top, int *left)
{
	int srch = srcimg.rows, srcw = srcimg.cols;
	*newh = this->inpHeight;
	*neww = this->inpWidth;
	Mat dstimg;
	if (this->keep_ratio && srch != srcw) {
		float hw_scale = (float)srch / srcw;
		if (hw_scale > 1) {
			*newh = this->inpHeight;
			*neww = int(this->inpWidth / hw_scale);
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*left = int((this->inpWidth - *neww) * 0.5);
			copyMakeBorder(dstimg, dstimg, 0, 0, *left, this->inpWidth - *neww - *left, BORDER_CONSTANT, Scalar(0, 0, 0));
		}
		else {
			*newh = (int)this->inpHeight * hw_scale;
			*neww = this->inpWidth;
			std::cout << "*newh, *neww, *top, *left: " << *newh << ", " << *neww << ", " << *top << ", " << *left << std::endl;
			resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
			*top = (int)(this->inpHeight - *newh) * 0.5;
			copyMakeBorder(dstimg, dstimg, *top, this->inpHeight - *newh - *top, 0, 0, BORDER_CONSTANT, Scalar(0, 0, 0));
		}
	}
	else {
		resize(srcimg, dstimg, Size(*neww, *newh), INTER_AREA);
	}
	// std::cout << "*newh, *neww, *top, *left: " << *newh << ", " << *neww << ", " << *top << ", " << *left << std::endl;
	return dstimg;
}

void YOLOv5::normalize_(Mat img)
{
	img.convertTo(img, CV_32FC3);
	int row = img.rows;
	int col = img.cols;
	vector<float> mean = {0.485, 0.456, 0.406};
    vector<float> std = {0.229, 0.224, 0.225};
	this->input_image_.resize(row * col * img.channels());  // vector大小
	for (int c = 0; c < 3; c++)  // bgr
	{
		for (int i = 0; i < row; i++)  // 行
		{
			for (int j = 0; j < col; j++)  // 列
			{
				float pix = img.ptr<char32_t>(i)[j * 3 + 2 - c];  // Mat里的ptr函数访问任意一行像素的首地址,2-c:表示rgb
				this->input_image_[c * row * col + i * col + j] = (pix / 255.0 - mean[2-c]) / std[2-c];
			}
		}
	}
}

void YOLOv5::nms(vector<BoxInfo>& input_boxes)
{
	sort(input_boxes.begin(), input_boxes.end(), [](BoxInfo a, BoxInfo b) { return a.score > b.score; }); // 降序排列
	vector<float> vArea(input_boxes.size());
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		vArea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1)
			* (input_boxes[i].y2 - input_boxes[i].y1 + 1);
	}
	// 全初始化为false，用来作为记录是否保留相应索引下pre_box的标志vector
	vector<bool> isSuppressed(input_boxes.size(), false);
	for (int i = 0; i < input_boxes.size(); ++i)
	{
		if (isSuppressed[i]) { continue; }
		for (int j = i + 1; j < input_boxes.size(); ++j)
		{
			if (isSuppressed[j]) { continue; }
			float xx1 = max(input_boxes[i].x1, input_boxes[j].x1);
			float yy1 = max(input_boxes[i].y1, input_boxes[j].y1);
			float xx2 = min(input_boxes[i].x2, input_boxes[j].x2);
			float yy2 = min(input_boxes[i].y2, input_boxes[j].y2);

			float w = max(0.0f, xx2 - xx1 + 1);
			float h = max(0.0f, yy2 - yy1 + 1);
			float inter = w * h;	// 交集
			// if(input_boxes[i].label == input_boxes[j].label)
			// {
			float ovr = inter / (vArea[i] + vArea[j] - inter);  // 计算iou
			// std::cout << "===> IoU: " << ovr << std::endl;
			if (ovr > this->nmsThreshold)
			{
				std::cout << "===> IoU: " << ovr << std::endl;
				isSuppressed[j] = true;
			}
			// }
		}
	}
	// return post_nms;
	int idx_t = 0;
       // remove_if()函数 remove_if(beg, end, op) //移除区间[beg,end)中每一个“令判断式:op(elem)获得true”的元素
	input_boxes.erase(remove_if(input_boxes.begin(), input_boxes.end(), [&idx_t, &isSuppressed](const BoxInfo& f) { return isSuppressed[idx_t++]; }), input_boxes.end());
}

void YOLOv5::detect(Mat& frame)
{
	// 图像初始化
	int newh = 0, neww = 0, padh = 0, padw = 0;
	Mat dstimg = this->resize_image(frame, &newh, &neww, &padh, &padw);
	std::cout << "===> dstimg (h, w): " << dstimg.rows << ", " << dstimg.cols << std::endl;


	this->normalize_(dstimg);
	std::cout << "===> inputs_images_ (h, w): " << this->input_image_.size() << std::endl;
	imshow("kWinName", dstimg);
	// std::cout << "&newh, &neww, &padh, &padw: " << newh << ", " << neww << ", " << padh << ", " << padw << std::endl;

	// 定义一个输入矩阵，int64_t 是下面作为输入参数时的类型
	vector<int64_t> input_shape_{ 1, 3, this->inpHeight, this->inpWidth };

    //创建输入tensor
	auto allocator_info = MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Value input_tensor_ = Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
	// std::cout << "====> input_image_.size(): " << input_image_.size() << std::endl;
	// 开始推理
	vector<Value> ort_outputs = ort_session->Run(RunOptions{ nullptr }, &input_names[0], &input_tensor_, input_names.size(), output_names.data(), output_names.size());   // 开始推理

	// generate proposals
	vector<BoxInfo> generate_boxes;  // BoxInfo 自定义的结构体

	// float ratioh = (float)frame.rows / newh, ratiow = (float)frame.cols / neww;
	float ratioh = (float)frame.rows / this->inpHeight, ratiow = (float)frame.cols / this->inpWidth;
	// float ratioh = (float)frame.rows, ratiow = (float)frame.cols;
	// std::cout << "===> frame_rows, newh, frame_cols, neww: " << (float)frame.rows << ", " << newh << ", " << (float)frame.cols << ", " << neww << std::endl;

	for (int n = 0; n < 3; n++)
	{
		Value &score = ort_outputs.at(n);
		Value &bbox = ort_outputs.at(n+3);
		// std::cout << "this->output_node_dims[0][2]: " << this->output_node_dims[n][2] << std::endl;
		// std::cout << "this->output_node_dims[0][3]: " << this->output_node_dims[n][3] << std::endl;

		for (int i = 0; i < this->output_node_dims[n][2]; ++i)
		{
			for (int j = 0; j < this->output_node_dims[n][3]; ++j)
			{
				float confs = score.At<float>({0, 0, i, j});
				// std::cout << "confs: " << confs << std::endl;
				if (confs > this->confThreshold) // 再次筛选
				{
					std::cout << "===> confs: " << confs << std::endl;
					float cor_cx = bbox.At<float>({0, 0, i, j});
					std::cout << "cor_x: " << cor_cx << std::endl;
					float cor_cy = bbox.At<float>({0, 1, i, j});
					std::cout << "cor_y: " << cor_cy << std::endl;
					float cor_w = bbox.At<float>({0, 2, i, j});
					std::cout << "cor_w: " << cor_w << std::endl;
					float cor_h = bbox.At<float>({0, 3, i, j});
					std::cout << "cor_h: " << cor_w << std::endl;

					float xmin = std::max(0.f, (float)((cor_cx)*ratiow));
					float ymin = std::max(0.f, (float)((cor_cy)*ratioh));
					float xmax = std::min(frame.rows - 1.f, (float)((cor_w)*ratiow));
					float ymax = std::min(frame.cols - 1.f, (float)((cor_h)*ratioh));

					generate_boxes.push_back(BoxInfo{ xmin, ymin, xmax, ymax, confs });

				}
				float confs2 = score.At<float>({0, 1, i, j});
				// std::cout << "confs2: " << confs2 << std::endl;
				if (confs2 > this->confThreshold) // 再次筛选
				{
					std::cout << "===> confs2: " << confs2 << std::endl;
					float cor_cx2 = bbox.At<float>({0, 4, i, j});
					std::cout << "cor_cx2: " << cor_cx2 << std::endl;
					float cor_cy2 = bbox.At<float>({0, 5, i, j});
					std::cout << "cor_cy2: " << cor_cy2 << std::endl;
					float cor_w2 = bbox.At<float>({0, 6, i, j});
					std::cout << "cor_w2: " << cor_w2 << std::endl;
					float cor_h2 = bbox.At<float>({0, 7, i, j});
					std::cout << "cor_h2: " << cor_h2 << std::endl;


					float xmin2 = std::max(0.f, (float)((cor_cx2)*ratiow));
					float ymin2 = std::max(0.f, float((cor_cy2)*ratioh));
					float xmax2 = std::min(frame.rows - 1.f, (float)((cor_w2)*ratiow));
					float ymax2 = std::min(frame.cols - 1.f, (float)((cor_h2)*ratioh));

					generate_boxes.push_back(BoxInfo{ xmin2, ymin2, xmax2, ymax2, confs2 });
				}
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	nms(generate_boxes);
	for (size_t i = 0; i < generate_boxes.size(); ++i)
	{
		int xmin = (int)generate_boxes[i].x1;
		int ymin = (int)generate_boxes[i].y1;
		// std::cout << "(x, y): " << xmin << ", " << ymin << std::endl;
		rectangle(frame, Point(xmin, ymin), Point((int)generate_boxes[i].x2, (int)generate_boxes[i].y2), Scalar(0, 0, 255), 2);
		// string label = format("%.2f", generate_boxes[i].score);
		// label = this->classes[generate_boxes[i].label] + ":" + label;
		// putText(frame, label, Point(xmin, ymin - 5), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 255), 2);
	}
}
