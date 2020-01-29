
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
extern "C" __global__ void __closesthit__radiance() {}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {}
extern "C" __global__ void __raygen__renderFrame() {
  const int frameID = optixLaunchParams.frameID;
  auto ix = optixGetLaunchIndex().x;
  auto iy = optixGetLaunchIndex().y;
  auto fbIndex = ((ix) + (((iy) * (optixLaunchParams.fbSize_x))));
  optixLaunchParams.colorBuffer[fbIndex] = 0xFF123456;
};