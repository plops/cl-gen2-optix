#ifndef GLOBALS_H

#define GLOBALS_H

#include <GLFW/glfw3.h>

#include <cassert>
#include <complex>
#include <condition_variable>
#include <cuda_runtime.h>
#include <deque>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>
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
  glm::vec3 camera_position;
  glm::vec3 camera_direction;
  glm::vec3 camera_horizontal;
  glm::vec3 camera_vertical;
  OptixTraversableHandle traversable;
};
typedef struct LaunchParams LaunchParams;
struct camera_t {
  glm::vec3 from;
  glm::vec3 at;
  glm::vec3 up;
};
typedef struct camera_t camera_t;
class linear_space_t {
public:
  glm::vec3 vx;
  glm::vec3 vy;
  glm::vec3 vz;
};
class affine_space_t {
public:
  linear_space_t l;
  glm::vec3 p;
};
inline glm::vec3 xfm_point(const affine_space_t &m, const glm::vec3 &p) {
  return glm::fma(glm::vec3(p[0]), m.l.vx,
                  glm::fma(glm::vec3(p[1]), m.l.vy,
                           glm::fma(glm::vec3(p[2]), m.l.vz, m.p)));
}
class triangle_mesh_t {
public:
  std::vector<glm::vec3> _vertex;
  std::vector<glm::ivec3> _index;
  void add_unit_cube(const affine_space_t &m) {
    auto first_vertex_id = static_cast<int>(_vertex.size());
    _vertex.push_back(xfm_point(m, glm::vec3((0.0e+0f), (0.0e+0f), (0.0e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((1.e+0f), (0.0e+0f), (0.0e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((0.0e+0f), (1.e+0f), (0.0e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((1.e+0f), (1.e+0f), (0.0e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((0.0e+0f), (0.0e+0f), (1.e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((1.e+0f), (0.0e+0f), (1.e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((0.0e+0f), (1.e+0f), (1.e+0f))));
    _vertex.push_back(xfm_point(m, glm::vec3((1.e+0f), (1.e+0f), (1.e+0f))));
    int indices[] = {0, 1, 3, 2, 3, 0, 5, 7, 6, 5, 6, 4, 0, 4, 5, 0, 5, 1,
                     2, 3, 7, 2, 7, 6, 1, 5, 7, 1, 7, 3, 4, 0, 2, 4, 2, 6};
    for (int i = 0; i < 12; (i) += (1)) {
      _index.push_back(((glm::ivec3(indices[((0) + (((3) * (i))))],
                                    indices[((1) + (((3) * (i))))],
                                    indices[((2) + (((3) * (i))))])) +
                        (first_vertex_id)));
    };
  }
  void add_cube(const glm::vec3 &center, const glm::vec3 &size) {
    const affine_space_t m = {{glm::vec3(size.x, 0, 0), glm::vec3(0, size.y, 0),
                               glm::vec3(0, 0, size.z)},
                              ((center) - ((((5.e-1f)) * (size))))};
    add_unit_cube(m);
  }
};
class CUDABuffer {
public:
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
      auto res = cudaMemcpy(_d_ptr, static_cast<const void *>(dat),
                            ((count) * (sizeof(T))), cudaMemcpyHostToDevice);
      if (!((CUDA_SUCCESS) == (res))) {

        (std::cout)
            << (std::setw(10))
            << (std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count())
            << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
            << (":") << (__LINE__) << (" ") << (__func__) << (" ")
            << ("FAIL: cuda cudaMemcpy(_d_ptr, static_cast<const void*>(dat), "
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
  camera_t last_set_camera;
  CUDABuffer color_buffer;
  CUDABuffer launch_params_buffer;
  LaunchParams launch_params;
  OptixShaderBindingTable shader_bindings_table;
  CUDABuffer hitgroup_records_buffer;
  std::vector<OptixProgramGroup> hitgroup_programs;
  CUDABuffer miss_records_buffer;
  std::vector<OptixProgramGroup> miss_programs;
  CUDABuffer raygen_records_buffer;
  std::vector<OptixProgramGroup> raygen_programs;
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
  std::vector<uint32_t> _pixels;
  char const *_filename;
};
typedef struct State State;

#endif
