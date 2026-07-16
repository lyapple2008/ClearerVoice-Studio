cmake_minimum_required(VERSION 3.20)
project(clearvoice_se_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  kaldi_native_fbank
  GIT_REPOSITORY https://github.com/csukuangfj/kaldi-native-fbank.git
  GIT_TAG b09e686fe2084732ddd30d1ef80acfc0f13eaf01
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(kaldi_native_fbank)

set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime library")
if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  add_library(onnxruntime_external UNKNOWN IMPORTED)
  set_target_properties(onnxruntime_external PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
  set(ONNXRUNTIME_TARGET onnxruntime_external)
else()
  find_package(onnxruntime CONFIG REQUIRED)
  set(ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
endif()

add_executable(clearvoice_se
  src/audio_io.cpp
  src/enhancer.cpp
  src/main.cpp
)
target_link_libraries(clearvoice_se PRIVATE
  kaldi-native-fbank-core
  kissfft
  ${ONNXRUNTIME_TARGET}
)
target_include_directories(clearvoice_se PRIVATE ${kaldi_native_fbank_SOURCE_DIR})

if(MSVC)
  target_compile_options(clearvoice_se PRIVATE /W4)
else()
  target_compile_options(clearvoice_se PRIVATE -Wall -Wextra -Wpedantic)
endif()
