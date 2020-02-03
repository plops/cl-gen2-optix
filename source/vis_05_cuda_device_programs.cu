
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
  const uint64_t uptr = (((static_cast<uint64_t>(i0)) << (32)) | (i1));
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
inline __device__ float3 random_color(int i) {
  auto r = static_cast<int>(
      ((0x234235) + (((13) * (17) * (static_cast<unsigned>(i))))));
  auto g = static_cast<int>(
      ((0x773477) + (((7) * (3) * (5) * (static_cast<unsigned>(i))))));
  auto b = static_cast<int>(
      ((0x223766) + (((11) * (19) * (static_cast<unsigned>(i))))));
  float3 res;
  res.x = ((((r) & (255))) / ((2.55e+2f)));
  res.y = ((((g) & (255))) / ((2.55e+2f)));
  res.z = ((((b) & (255))) / ((2.55e+2f)));
  return res;
}
extern "C" __global__ void __closesthit__radiance() {
  auto id = optixGetPrimitiveIndex();
  float3 *prd = get_prd<float3>();
  auto c = random_color(id);
  prd->x = (0.0e+0f);
  prd->y = (1.e+0f);
  prd->z = (0.0e+0f);
}
extern "C" __global__ void __anyhit__radiance() {}
extern "C" __global__ void __miss__radiance() {
  float3 *prd = get_prd<float3>();
  prd->x = (1.e-1f);
  prd->y = (0.0e+0f);
  prd->z = (0.0e+0f);
}
extern "C" __global__ void __exception__all() {
  printf("optix exception: %d\n", optixGetExceptionCode());
}
extern "C" __global__ void __raygen__renderFrame() {
  const int frameID = optixLaunchParams.frameID;
  auto ix = optixGetLaunchIndex().x;
  auto iy = optixGetLaunchIndex().y;
  auto camera_position = optixLaunchParams.camera_position;
  auto camera_direction = optixLaunchParams.camera_direction;
  auto camera_horizontal = optixLaunchParams.camera_horizontal;
  auto camera_vertical = optixLaunchParams.camera_vertical;
  float3 pixel_color_prd;
  auto u0 = uint32_t(0);
  auto u1 = uint32_t(0);
  auto screen =
      ((glm::vec2(((.5f) + (ix)), ((.5f) + (iy)))) /
       (glm::vec2(optixLaunchParams.fbSize_x, optixLaunchParams.fbSize_y)));
  auto ray_dir =
      glm::normalize(((camera_direction) +
                      (((camera_horizontal) * (((screen[0]) - ((5.e-1f)))))) +
                      (((camera_vertical) * (((screen[1]) - ((5.e-1f))))))));
  auto fbIndex = ((ix) + (((iy) * (optixLaunchParams.fbSize_x))));
  pack_pointer(&pixel_color_prd, u0, u1);
  auto pos = reinterpret_cast<float3 *>(&camera_position);
  auto dir = reinterpret_cast<float3 *>(&ray_dir);
  optixTrace(optixLaunchParams.traversable, *pos, *dir, (0.0e+0f), (1.e+20f),
             (0.0e+0f), OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE,
             SURFACE_RAY_TYPE, RAY_TYPE_COUNT, SURFACE_RAY_TYPE, u0, u1);
  auto r = static_cast<int>((((2.5599e+2f)) * (pixel_color_prd.x)));
  auto g = static_cast<int>((((2.5599e+2f)) * (pixel_color_prd.y)));
  auto b = static_cast<int>((((2.5599e+2f)) * (pixel_color_prd.z)));
  auto rgba = ((4278190080) | ((r) << (0)) | ((g) << (8)) | ((b) << (16)));
  optixLaunchParams.colorBuffer[fbIndex] = rgba;
};