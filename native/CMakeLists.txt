cmake_minimum_required(VERSION 3.16)
project(silero_vad_bridge C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN YES)

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime root directory")

find_path(
  ONNXRUNTIME_INCLUDE_DIR
  NAMES onnxruntime_c_api.h
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES include)

find_library(
  ONNXRUNTIME_LIBRARY
  NAMES onnxruntime
  HINTS ${ONNXRUNTIME_ROOT}
  PATH_SUFFIXES lib)

if(NOT ONNXRUNTIME_INCLUDE_DIR)
  message(FATAL_ERROR "onnxruntime_c_api.h not found. Set ONNXRUNTIME_ROOT or CMAKE_INCLUDE_PATH.")
endif()

if(NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "onnxruntime library not found. Set ONNXRUNTIME_ROOT or CMAKE_LIBRARY_PATH.")
endif()

add_library(silero_vad_bridge SHARED src/vad_bridge.c)

target_include_directories(
  silero_vad_bridge
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})

target_link_libraries(silero_vad_bridge PRIVATE ${ONNXRUNTIME_LIBRARY})

if(APPLE)
  set_target_properties(
    silero_vad_bridge
    PROPERTIES
      OUTPUT_NAME "silero_vad_bridge"
      INSTALL_RPATH "@loader_path")
else()
  set_target_properties(
    silero_vad_bridge
    PROPERTIES
      OUTPUT_NAME "silero_vad_bridge"
      INSTALL_RPATH "$ORIGIN")
endif()
