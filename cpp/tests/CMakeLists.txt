# 只链接与硬件无关的算法文件，x86 上直接跑，不需要 RKNN / ONNX Runtime。
add_executable(ed_tests
  test_main.cpp
  test_core.cpp
  test_highlight.cpp
  test_tracker.cpp
  test_aesthetics.cpp
  ${CMAKE_SOURCE_DIR}/src/highlight.cpp
  ${CMAKE_SOURCE_DIR}/src/tracker.cpp
  ${CMAKE_SOURCE_DIR}/src/aesthetics.cpp
)

target_include_directories(ed_tests PRIVATE
  ${CMAKE_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(ed_tests PRIVATE ${OpenCV_LIBS})

add_test(NAME ed_core COMMAND ed_tests)
