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

set(ONNXRUNTIME_ROOT "" CACHE PATH "Root of an official ONNX Runtime package")
set(ONNXRUNTIME_INCLUDE_DIR "" CACHE PATH "ONNX Runtime include directory")
set(ONNXRUNTIME_LIBRARY "" CACHE FILEPATH "ONNX Runtime library")
if(ONNXRUNTIME_ROOT)
  unset(ONNXRUNTIME_ROOT_LIBRARY CACHE)
  find_library(ONNXRUNTIME_ROOT_LIBRARY
    NAMES onnxruntime
    PATHS "${ONNXRUNTIME_ROOT}/lib"
    NO_DEFAULT_PATH
    REQUIRED)
  set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include"
    CACHE PATH "ONNX Runtime include directory" FORCE)
  set(ONNXRUNTIME_LIBRARY "${ONNXRUNTIME_ROOT_LIBRARY}"
    CACHE FILEPATH "ONNX Runtime library" FORCE)
endif()
if((ONNXRUNTIME_INCLUDE_DIR AND NOT ONNXRUNTIME_LIBRARY) OR
   (ONNXRUNTIME_LIBRARY AND NOT ONNXRUNTIME_INCLUDE_DIR))
  message(FATAL_ERROR
    "ONNXRUNTIME_INCLUDE_DIR and ONNXRUNTIME_LIBRARY must be set together")
endif()

if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
  add_library(onnxruntime_external UNKNOWN IMPORTED)
  set_target_properties(onnxruntime_external PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
  )
  set(ONNXRUNTIME_TARGET onnxruntime_external)
  set(ONNXRUNTIME_CHECK_INCLUDE_DIR "${ONNXRUNTIME_INCLUDE_DIR}")
  set(ONNXRUNTIME_CHECK_LIBRARY "${ONNXRUNTIME_LIBRARY}")
else()
  find_package(onnxruntime CONFIG REQUIRED)
  set(ONNXRUNTIME_TARGET onnxruntime::onnxruntime)
  get_target_property(ONNXRUNTIME_CHECK_INCLUDE_DIR
    onnxruntime::onnxruntime INTERFACE_INCLUDE_DIRECTORIES)
  get_target_property(ONNXRUNTIME_CHECK_LIBRARY
    onnxruntime::onnxruntime IMPORTED_LOCATION_RELEASE)
  if(NOT ONNXRUNTIME_CHECK_LIBRARY)
    get_target_property(ONNXRUNTIME_CHECK_LIBRARY
      onnxruntime::onnxruntime IMPORTED_LOCATION)
  endif()
endif()

if(NOT CMAKE_CROSSCOMPILING)
  get_filename_component(ONNXRUNTIME_CHECK_LIBRARY_DIR
    "${ONNXRUNTIME_CHECK_LIBRARY}" DIRECTORY)
  try_run(
    ONNXRUNTIME_API_RUN_RESULT
    ONNXRUNTIME_API_COMPILE_RESULT
    "${CMAKE_BINARY_DIR}/onnxruntime_api_check"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_ort_api.cpp"
    CMAKE_FLAGS
      "-DINCLUDE_DIRECTORIES=${ONNXRUNTIME_CHECK_INCLUDE_DIR}"
      "-DCMAKE_BUILD_RPATH=${ONNXRUNTIME_CHECK_LIBRARY_DIR}"
    LINK_LIBRARIES "${ONNXRUNTIME_CHECK_LIBRARY}"
    RUN_OUTPUT_VARIABLE ONNXRUNTIME_API_CHECK_OUTPUT
  )
  if(NOT ONNXRUNTIME_API_COMPILE_RESULT OR
     NOT ONNXRUNTIME_API_RUN_RESULT EQUAL 0)
    message(FATAL_ERROR
      "ONNX Runtime headers and library are incompatible.\n"
      "Include: ${ONNXRUNTIME_CHECK_INCLUDE_DIR}\n"
      "Library: ${ONNXRUNTIME_CHECK_LIBRARY}\n"
      "Check output: ${ONNXRUNTIME_API_CHECK_OUTPUT}\n"
      "Use headers and library from the same ONNX Runtime release, then clear "
      "the CMake cache or configure a new build directory.")
  endif()
endif()

add_library(clearvoice_runtime STATIC
  src/audio_io.cpp
  src/bandwidth_sub.cpp
  src/clearvoice.cpp
  src/enhancer.cpp
  src/gan_model.cpp
  src/onnx_runner.cpp
  src/resample.cpp
)
target_link_libraries(clearvoice_runtime PUBLIC
  kaldi-native-fbank-core
  kissfft
  ${ONNXRUNTIME_TARGET}
)
target_include_directories(clearvoice_runtime PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${kaldi_native_fbank_SOURCE_DIR}
)

add_executable(clearvoice src/clearvoice_main.cpp)
target_link_libraries(clearvoice PRIVATE clearvoice_runtime)

add_executable(clearvoice_se src/main.cpp)
target_link_libraries(clearvoice_se PRIVATE clearvoice_runtime)

if(MSVC)
  target_compile_options(clearvoice_runtime PRIVATE /W4)
else()
  target_compile_options(clearvoice_runtime PRIVATE -Wall -Wextra -Wpedantic)
endif()
