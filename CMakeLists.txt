cmake_minimum_required(VERSION 3.0)

project(bs-detection CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_CURRENT_LIST_DIR}/../cmake)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Franka REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(OpenCV 4.2.0 REQUIRED )
find_package(realsense2 REQUIRED)
find_package(VISP REQUIRED)
find_package(PCL 1.8 REQUIRED )

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(common STATIC
  common.cpp
)

include_directories(${VISP_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(include/imbs/)

add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

target_link_libraries(common PUBLIC Franka::Franka Eigen3::Eigen3)

set(OpenCV_LIBS /usr/local/lib/libopencv_dnn.so /usr/local/lib/libopencv_core.so /usr/local/lib/libopencv_core.so.4.2 /usr/local/lib/libopencv_imgproc.so.4.2 opencv_core opencv_highgui)

add_executable(bs-main bs_detection.cpp pc-utils.hpp include/imbs/imbs.hpp include/imbs/imagemanager.h include/imbs/natural_less.h)
target_link_libraries(bs-main Franka::Franka common movep2p Eigen3::Eigen3)
target_link_libraries(bs-main ${OpenCV_LIBS} )
target_link_libraries(bs-main ${realsense2_LIBRARY})
target_link_libraries(bs-main ${VISP_LIBRARIES})
target_link_libraries(bs-main ${PCL_LIBRARIES} )
target_link_libraries(bs-main Threads::Threads)

include(GNUInstallDirs)
install(TARGETS ${UNIBAS_SINGLE_ROBOT}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)