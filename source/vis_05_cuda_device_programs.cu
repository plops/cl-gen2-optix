
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <optix_device.h>

__constant__ LaunchParams optixLaunchParams;
__global__ void __closesthit_radiance() {}
__global__ void __anythit_radiance() {}
__global__ void __miss_radiance() {}
__global__ void __raygen__renderFrame() {
  const int frameID = optixLaunchParams.frameID;
  auto ix = optixGetLaunchIndex().x;
  auto iy = optixGetLaunchIndex().y;
  auto fbIndex = ((ix) + (((iy) * (optixLaunchParams.fbSize_x))));
  optixLaunchParams.colorBuffer[fbIndex] = 0xFF123456;
};