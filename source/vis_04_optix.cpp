
#include "utils.h"

#include "globals.h"

#include "proto2.h"
;
extern State state;
// derived from Ingo Wald's optix7course example03_inGLFWindow
// SampleRenderer.cpp
#include <cuda_runtime.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

extern "C" const char ptx_code[];
void createContext() {
  {
    auto res = cudaSetDevice(state.dev_id);
    if (!((CUDA_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda cudaSetDevice(state.dev_id)")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  {
    auto res = cudaStreamCreate(&(state.stream));
    if (!((CUDA_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda cudaStreamCreate(&(state.stream))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  cudaGetDeviceProperties(&(state.dev_prop), state.dev_id);

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("running on device:")
      << (" ") << (std::setw(8)) << (" state.dev_prop.name=")
      << (state.dev_prop.name) << (std::endl) << (std::flush);
  {
    auto res = cuCtxGetCurrent(&(state.cuctx));
    if (!((CUDA_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda cuCtxGetCurrent(&(state.cuctx))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  {
    OptixResult res = optixDeviceContextCreate(state.cuctx, 0, &(state.oxctx));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("FAIL: optix optixDeviceContextCreate(state.cuctx, 0, "
                      "&(state.oxctx))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  auto log_cb = [](unsigned int level, const char *tag, const char *msg,
                   void *data) {
    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("context_log") << (" ") << (std::setw(8)) << (" level=")
                << (level) << (std::setw(8)) << (" tag=") << (tag)
                << (std::setw(8)) << (" msg=") << (msg) << (std::endl)
                << (std::flush);
  };
  {
    OptixResult res =
        optixDeviceContextSetLogCallback(state.oxctx, log_cb, nullptr, 4);
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixDeviceContextSetLogCallback(state.oxctx, "
              "log_cb, nullptr, 4)")
          << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
          << (std::flush);
    };
  };
}
void createModule() {
  state.module_compile_options = {};
  state.module_compile_options.maxRegisterCount = 50;
  state.module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
  state.module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
  state.pipeline_compile_options = {};
  state.pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.numPayloadValues = 2;
  state.pipeline_compile_options.numAttributeValues = 2;
  state.pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
  state.pipeline_compile_options.pipelineLaunchParamsVariableName =
      "optixLaunchParams";
  state.pipeline_link_options = {};
  state.pipeline_link_options.overrideUsesMotionBlur = false;
  state.pipeline_link_options.maxTraceDepth = 2;
  const std::string ptx = ptx_code;
  char log[2048];
  auto size_log = sizeof(log);
  {
    OptixResult res =
        optixModuleCreateFromPTX(state.oxctx, &(state.module_compile_options),
                                 &(state.pipeline_compile_options), ptx.c_str(),
                                 ptx.size(), log, &size_log, &(state.module));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("FAIL: optix optixModuleCreateFromPTX(state.oxctx, "
                      "&(state.module_compile_options), "
                      "&(state.pipeline_compile_options), ptx.c_str(), "
                      "ptx.size(), log, &size_log, &(state.module))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  if (1 < size_log) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" size_log=") << (size_log)
                << (std::setw(8)) << (" log=") << (log) << (std::endl)
                << (std::flush);
  };
}
void createRayGenPrograms() {
  state.ray_programs.resize(1);
  OptixProgramGroupOptions pg_options;
  OptixProgramGroupDesc pg_desc;
  pg_options = {};
  ;
  pg_desc = {};
  pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
  pg_desc.raygen.module = state.module;
  pg_desc.raygen.entryFunctionName = "__raygen__renderFrame";
  char log[2048];
  auto size_log = sizeof(log);
  {
    OptixResult res =
        optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, &(pg_options), log,
                                &size_log, &(state.ray_programs[0]));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, "
              "&(pg_options), log, &size_log, &(state.ray_programs[0]))")
          << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
          << (std::flush);
    };
  };
  if (1 < size_log) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" size_log=") << (size_log)
                << (std::setw(8)) << (" log=") << (log) << (std::endl)
                << (std::flush);
  };
}
void createMissPrograms() {
  state.miss_programs.resize(1);
  OptixProgramGroupOptions pg_options;
  OptixProgramGroupDesc pg_desc;
  pg_options = {};
  ;
  pg_desc = {};
  pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
  pg_desc.miss.module = state.module;
  pg_desc.miss.entryFunctionName = "__miss__radiance";
  char log[2048];
  auto size_log = sizeof(log);
  {
    OptixResult res =
        optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, &(pg_options), log,
                                &size_log, &(state.miss_programs[0]));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, "
              "&(pg_options), log, &size_log, &(state.miss_programs[0]))")
          << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
          << (std::flush);
    };
  };
  if (1 < size_log) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" size_log=") << (size_log)
                << (std::setw(8)) << (" log=") << (log) << (std::endl)
                << (std::flush);
  };
}
void createHitGroupPrograms() {
  state.hit_programs.resize(1);
  OptixProgramGroupOptions pg_options;
  OptixProgramGroupDesc pg_desc;
  pg_options = {};
  ;
  pg_desc = {};
  pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  pg_desc.hitgroup.moduleCH = state.module;
  pg_desc.hitgroup.moduleAH = state.module;
  pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
  pg_desc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";
  char log[2048];
  auto size_log = sizeof(log);
  {
    OptixResult res =
        optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, &(pg_options), log,
                                &size_log, &(state.hit_programs[0]));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, "
              "&(pg_options), log, &size_log, &(state.hit_programs[0]))")
          << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
          << (std::flush);
    };
  };
  if (1 < size_log) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" size_log=") << (size_log)
                << (std::setw(8)) << (" log=") << (log) << (std::endl)
                << (std::flush);
  };
}
void createPipeline() {
  std::vector<OptixProgramGroup> program_groups;
  for (auto &p : state.ray_programs) {
    program_groups.push_back(p);
  };
  for (auto &p : state.miss_programs) {
    program_groups.push_back(p);
  };
  for (auto &p : state.hit_programs) {
    program_groups.push_back(p);
  };
  char log[2048];
  auto size_log = sizeof(log);
  {
    OptixResult res = optixPipelineCreate(
        state.oxctx, &(state.pipeline_compile_options),
        &(state.pipeline_link_options), program_groups.data(),
        static_cast<int>(program_groups.size()), log, &size_log,
        &(state.pipeline));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("FAIL: optix optixPipelineCreate(state.oxctx, "
                      "&(state.pipeline_compile_options), "
                      "&(state.pipeline_link_options), program_groups.data(), "
                      "static_cast<int>(program_groups.size()), log, "
                      "&size_log, &(state.pipeline))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  if (1 < size_log) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" size_log=") << (size_log)
                << (std::setw(8)) << (" log=") << (log) << (std::endl)
                << (std::flush);
  };
  {
    OptixResult res = optixPipelineSetStackSize(
        state.pipeline, ((2) * (1024)), ((2) * (1024)), ((2) * (1024)), 1);
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("FAIL: optix optixPipelineSetStackSize(state.pipeline, "
                      "((2)*(1024)), ((2)*(1024)), ((2)*(1024)), 1)")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  if (1 < size_log) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ") << ("")
                << (" ") << (std::setw(8)) << (" size_log=") << (size_log)
                << (std::setw(8)) << (" log=") << (log) << (std::endl)
                << (std::flush);
  };
}
void initOptix() {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("initOptix") << (" ")
      << (std::endl) << (std::flush);
  cudaFree(0);
  int num_devices;
  cudaGetDeviceCount(&num_devices);
  if ((0) == (num_devices)) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("FAIL: no cuda device") << (" ") << (std::endl)
                << (std::flush);
  };
  {
    OptixResult res = optixInit();
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: optix optixInit()") << (" ")
                  << (std::setw(8)) << (" res=") << (res) << (std::endl)
                  << (std::flush);
    };
  };
  createContext();
  createModule();
  createRayGenPrograms();
  createMissPrograms();
  createHitGroupPrograms();
  createPipeline();
}
void cleanupOptix(){};