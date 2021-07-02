# Object_grasping_bs

An application to grasp object using Background Subtraction technique.

# Requirements
* OpenCV 4.2+
* PCL 1.8+
* Libfranka 0.7 [more](https://github.com/frankaemika/libfranka)
* Intel® RealSense™ librealsense 2.31+ [more](https://github.com/IntelRealSense/librealsense)
* Visp [more](https://github.com/lagadic/visp)

# Installation
In order to install object_grasping_bs, download this repository and run

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```
# Reference
* [Franka Emika Panda](https://frankaemika.github.io/docs/)
* [Intel Realsense D435 camera](https://www.intelrealsense.com/depth-camera-d435/)
* [Visp with Franka](https://visp-doc.inria.fr/doxygen/visp-daily/tutorial-franka-pbvs.html)
