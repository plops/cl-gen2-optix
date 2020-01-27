#ifndef GLOBALS_H

#define GLOBALS_H

#include <GLFW/glfw3.h>

#include <cassert>
#include <complex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <mutex>
#include <optix.h>
#include <optix_stubs.h>
#include <queue>
#include <string>
#include <thread>
struct LaunchParams {
  int frameID;
  uint32_t *colorBuffer;
  int fbSize_x;
  int fbSize_y;
};
typedef struct LaunchParams LaunchParams;
class CUDABuffer {
  void *_d_ptr;
  size_t _size_in_bytes;
  CUdeviceptr d_pointer() { return reinterpret_cast<CUdeviceptr>(_d_ptr); }
  void resize(size_t size) {
    if (_d_ptr) {
      free();
    };
    alloc(size);
  }
  void alloc(size_t size) {
    assert((nullptr) == (_d_ptr));
    this->_size_in_bytes = size;
    {
      auto res = cudaMalloc(static_cast<void **>(&_d_ptr), _size_in_bytes);
      if (!((CUDA_SUCCESS) == (res))) {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ")
                    << ("FAIL: cuda cudaMalloc(static_cast<void**>(&_d_ptr), "
                        "_size_in_bytes)")
                    << (" ") << (std::setw(8)) << (" res=") << (res)
                    << (std::endl) << (std::flush);
      };
    };
  }
  void free() {
    {
      auto res = cudaFree(_d_ptr);
      if (!((CUDA_SUCCESS) == (res))) {

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("FAIL: cuda cudaFree(_d_ptr)") << (" ")
                    << (std::setw(8)) << (" res=") << (res) << (std::endl)
                    << (std::flush);
      };
    };
    _d_ptr = nullptr;
    _size_in_bytes = 0;
  }
  template <typename T> void alloc_and_upload(const std::vector<T> &vt) {
    alloc(((vt.size()) * (sizeof(T))));
    upload(static_cast<const T *>(vt.data()), vt.size());
  }
  template <typename T> void upload(const T *dat, size_t count) {
    assert((nullptr) != (_d_ptr));
    assert((_size_in_bytes) == (((count) * (sizeof(T)))));
    {
      auto res = cudaMemcpy(_d_ptr, static_cast<void *>(dat),
                            ((count) * (sizeof(T))), cudaMemcpyHostToDevice);
      if (!((CUDA_SUCCESS) == (res))) {

        (std::cout)
            << (std::setw(10))
            << (std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count())
            << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
            << (":") << (__LINE__) << (" ") << (__func__) << (" ")
            << ("FAIL: cuda cudaMemcpy(_d_ptr, static_cast<void*>(dat), "
                "((count)*(sizeof(T))), cudaMemcpyHostToDevice)")
            << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
            << (std::flush);
      };
    };
  }
  template <typename T> void download(T *dat, size_t count) {
    assert((nullptr) != (_d_ptr));
    assert((_size_in_bytes) == (((count) * (sizeof(T)))));
    {
      auto res = cudaMemcpy(static_cast<void *>(dat), _d_ptr,
                            ((count) * (sizeof(T))), cudaMemcpyDeviceToHost);
      if (!((CUDA_SUCCESS) == (res))) {

        (std::cout)
            << (std::setw(10))
            << (std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count())
            << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
            << (":") << (__LINE__) << (" ") << (__func__) << (" ")
            << ("FAIL: cuda cudaMemcpy(static_cast<void*>(dat), _d_ptr, "
                "((count)*(sizeof(T))), cudaMemcpyDeviceToHost)")
            << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
            << (std::flush);
      };
    };
  }
};
template <typename T, int MaxLen> class FixedDequeTM : public std::deque<T> {
  // https://stackoverflow.com/questions/56334492/c-create-fixed-size-queue
public:
  std::mutex mutex;
  void push_back(const T &val) {
    if ((MaxLen) == (this->size())) {
      this->pop_front();
    };
    std::deque<T>::push_back(val);
  }
};

#include <chrono>
struct State {
  typeof(std::chrono::high_resolution_clock::now().time_since_epoch().count())
      _start_time;
  CUDABuffer color_buffer;
  CUDABuffer launch_params_buffer;
  LaunchParams launch_params;
  OptixShaderBindingTable shader_bindings_table;
  CUDABuffer hitgroup_records_buffer;
  std::vector<OptixProgramGroup> hit_programs;
  std::vector<OptixProgramGroup> miss_programs;
  std::vector<OptixProgramGroup> ray_programs;
  OptixModule module;
  OptixPipeline pipeline;
  OptixPipelineLinkOptions pipeline_link_options;
  OptixPipelineCompileOptions pipeline_compile_options;
  OptixModuleCompileOptions module_compile_options;
  OptixDeviceContext oxctx;
  CUcontext cuctx;
  cudaDeviceProp dev_prop;
  CUstream stream;
  const int dev_id;
  GLuint _fontTex;
  bool _framebufferResized;
  GLFWwindow *_window;
  char const *_filename;
};
typedef struct State State;

#endif
