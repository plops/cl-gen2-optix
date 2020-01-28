
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
#include <optix_device.h>

extern "C" __constant__ LaunchParams optixLaunchParams;
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