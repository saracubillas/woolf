apt-get update
apt-get -y install curl
apt-get -y install libopencv-dev
apt-get -y install build-essential checkinstall cmake pkg-config yasm
apt-get -y install libtiff4-dev libjpeg-dev libjasper-dev
apt-get -y install libavcodec-dev libavformat-dev libswscale-dev libdc1394-22-dev libxine-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev libv4l-dev

apt-get -y install python-dev python-numpy
apt-get -y install libtbb-dev

apt-get -y install libqt4-dev libgtk2.0-dev


curl -L -O http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.6.1/opencv-2.4.6.1.tar.gz

tar -xvf opencv-2.4.6.1.tar.gz
cd opencv-2.4.6.1/
mkdir build
cd build

sudo cmake -D WITH_TBB=ON -D WITH_EIGEN=ON ..
sudo cmake -D BUILD_DOCS=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF ..
sudo cmake -D WITH_OPENCL=OFF -D WITH_CUDA=OFF -D BUILD_opencv_gpu=OFF -D BUILD_opencv_gpuarithm=OFF ..
sudo cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..


make

sudo make install


