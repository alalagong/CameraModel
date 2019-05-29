# CameraModel

Here is two camera models commonly used in **SLAM/VO**, it consists of MEI and Pinhole. It can rectify image or points, lift from pixel to unit plane and project from world to pixel. It gives both implementation based on **OpenCV** and **no-OpenCV**, so that we can explore all of the algorithm.

## 1. Dependencies

It's tested in **Ubuntu 16.04**

### 1.1 OpenCV

​	[OpenCV 3.2.0](https://docs.opencv.org/3.2.0/d2/d75/namespacecv.html) is used in the code.

### 1.2 Eigen

​	we use Eigen 3.2.29.

### 1.3 GLOG

​	This is used to record log in `log` folder.

> sudo apt-get install libgoogle-glog-dev

## 2. Build

​	After clone, run:

> mkdir build  
> cd build  
> cmake ..  
> make 


## 3. Usage
The calibration files are stored in `calib` folder, and the format of *yaml* can refer to the examples.
The images are stored in `data` folder.

### 3.1 Run test
​	We can test the code using commands:

> ./bin/test ./calib/cam0_tumvio_mei.yaml ./data/1.png

### 3.2 Output

​	After finish running, corrected image will be put to `data` folder, and name format is:

> original name + \_ undist \_  + model(pinhole or mei) + original extension

​	Examples: 

> hello.jpg   ==> hello_undist_mei.jpg

