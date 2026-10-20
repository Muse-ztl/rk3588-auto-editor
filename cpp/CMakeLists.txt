cmake_minimum_required(VERSION 3.16)
project(auto_editor CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn features2d calib3d)

# rknn_api.h 随仓库提供；运行库用板上的 librknnrt.so（2.3.2）
set(RKNN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/rknn/include" CACHE PATH "rknn headers")
find_library(RKNN_LIB NAMES rknnrt PATHS /usr/lib /usr/local/lib REQUIRED)

add_executable(auto_editor
  src/main.cpp
  src/detector.cpp
  src/tracker.cpp
  src/aesthetics.cpp
  src/highlight.cpp
  src/video_io.cpp
)

target_include_directories(auto_editor PRIVATE
  include
  ${OpenCV_INCLUDE_DIRS}
  ${RKNN_INCLUDE_DIR}
)

target_link_libraries(auto_editor PRIVATE ${OpenCV_LIBS} ${RKNN_LIB} pthread)
