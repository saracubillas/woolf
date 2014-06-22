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

 wget --no-check-certificate --no-cookies --header "Cookie: oraclelicense=accept-securebackup-cookie" http://download.oracle.com/otn-pub/java/jdk/8u5-b13/jdk-8u5-linux-x64.tar.gz
sudo su
mkdir /opt/jdk
tar -zxf jdk-8u5-linux-x64.tar.gz -C /opt/jdk
update-alternatives --install /usr/bin/java java /opt/jdk/jdk1.8.0_05/bin/java 100
update-alternatives --install /usr/bin/javac javac /opt/jdk/jdk1.8.0_05/bin/javac 100
