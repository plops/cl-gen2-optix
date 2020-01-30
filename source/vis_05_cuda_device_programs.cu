
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <optix_device.h>

extern "C" __constant__ LaunchParams optixLaunchParams;
enum { SURFACE_RAY_TYPE = 0, RAY_TYPE_COUNT };
static __forceinline__ __device__ void *unpack_pointer(uint32_t i0,
                                                       uint32_t i1) {
  auto uptr = (((static_cast<uint64_t>(i0)) << (32)) | (i1));
  return reinterpret_cast<void *>(uptr);
}
static __forceinline__ __device__ void pack_pointer(void *ptr, uint32_t &i0,
                                                    uint32_t &i1) {
  auto uptr = reinterpret_cast<uint64_t>(ptr);
  i0 = (uptr) >> (32);
  i1 = ((uptr) & (0xFFFFFFFF));
}
template <typename T> static __forceinline__ __device__ T *get_prd() {
  auto u0 = optixGetPayload_0();
  auto u1 = optixGetPayload_1();
  return reinterpret_cast<T *>(unpack_pointer(u0, u1));
}
inline __device__ glm::vec3 random_color(int i) {
  auto r = static_cast<int>(
      ((0x234235) + (((13) * (17) * (static_cast<unsigned>(i))))));
  auto g = static_cast<int>(
      ((0x773477) + (((7) * (3) * (5) * (static_cast<unsigned>(i))))));
  auto b = static_cast<int>(
      ((0x223766) + (((11) * (19) * (static_cast<unsigned>(i))))));
  return glm::vec3(((((r) & (255))) / ((2.55e+2f))),
                   ((((g) & (255))) / ((2.55e+2f))),
                   ((((b) & (255))) / ((2.55e+2f))));
}
extern "C" __global__ void __closesthit__radiance() {
  auto id = optixGetPrimitiveIndex();
  glm::vec3 &prd = *(get_prd<glm::vec3>());
  prd = random_color(id);
}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __raygen__renderFrame() {
  const int frameID = optixLaunchParams.frameID;
  auto ix = optixGetLaunchIndex().x;
  auto iy = optixGetLaunchIndex().y;
  auto fbIndex = ((ix) + (((iy) * (optixLaunchParams.fbSize_x))));
  optixLaunchParams.colorBuffer[fbIndex] = 0xFF123456;
};