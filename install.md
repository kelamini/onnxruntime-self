下载源码：
git clone https://github.com/microsoft/onnxruntime.git

拉取子项目：
cd onnxruntime
git submodule update --init --recursive
若不成功可以手动下载然后放到指定路径下

开始编译：
./build.sh --skip_tests --config Release --build_shared_lib --parallel --use_cuda --cuda_home /usr/local/cuda-11.3 --cudnn_home /usr/local/cuda-11.3

