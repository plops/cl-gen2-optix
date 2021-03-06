
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

#include <glm/geometric.hpp>
extern "C" const char ptx_code[];
void createContext() {
  int count = 0;
  {
    auto res = cudaGetDeviceCount(&count);
    if (!((cudaSuccess) == (res))) {
      auto err_ = cudaGetLastError();
      auto err_name = cudaGetErrorName(err_);
      auto err_str = cudaGetErrorString(err_);

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda  cudaGetDeviceCount(&count)")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                  << (" err_name=") << (err_name) << (std::setw(8))
                  << (" err_str=") << (err_str) << (std::endl) << (std::flush);
      throw std::runtime_error("cudaGetDeviceCount(&count)");
    };
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("get device count")
      << (" ") << (std::setw(8)) << (" count=") << (count) << (std::endl)
      << (std::flush);
  state.dev_id = 0;
  {
    auto res = cudaSetDevice(state.dev_id);
    if (!((cudaSuccess) == (res))) {
      auto err_ = cudaGetLastError();
      auto err_name = cudaGetErrorName(err_);
      auto err_str = cudaGetErrorString(err_);

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda  cudaSetDevice(state.dev_id)")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                  << (" err_name=") << (err_name) << (std::setw(8))
                  << (" err_str=") << (err_str) << (std::endl) << (std::flush);
      throw std::runtime_error("cudaSetDevice(state.dev_id)");
    };
  };
  {
    auto res = cudaStreamCreate(&(state.stream));
    if (!((cudaSuccess) == (res))) {
      auto err_ = cudaGetLastError();
      auto err_name = cudaGetErrorName(err_);
      auto err_str = cudaGetErrorString(err_);

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda  cudaStreamCreate(&(state.stream))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                  << (" err_name=") << (err_name) << (std::setw(8))
                  << (" err_str=") << (err_str) << (std::endl) << (std::flush);
      throw std::runtime_error("cudaStreamCreate(&(state.stream))");
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
    if (!((cudaSuccess) == (res))) {
      auto err_ = cudaGetLastError();
      auto err_name = cudaGetErrorName(err_);
      auto err_str = cudaGetErrorString(err_);

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ") << ("FAIL: cuda  cuCtxGetCurrent(&(state.cuctx))")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                  << (" err_name=") << (err_name) << (std::setw(8))
                  << (" err_str=") << (err_str) << (std::endl) << (std::flush);
      throw std::runtime_error("cuCtxGetCurrent(&(state.cuctx))");
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
  state.module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
  state.module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
  state.pipeline_compile_options = {};
  state.pipeline_compile_options.traversableGraphFlags =
      OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
  state.pipeline_compile_options.usesMotionBlur = false;
  state.pipeline_compile_options.numPayloadValues = 2;
  state.pipeline_compile_options.numAttributeValues = 2;
  state.pipeline_compile_options.exceptionFlags =
      ((OPTIX_EXCEPTION_FLAG_DEBUG) | (OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW) |
       (OPTIX_EXCEPTION_FLAG_TRACE_DEPTH));
  state.pipeline_compile_options.pipelineLaunchParamsVariableName =
      "optixLaunchParams";
  state.pipeline_link_options = {};
  state.pipeline_link_options.overrideUsesMotionBlur = false;
  state.pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
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
  state.raygen_programs.resize(1);
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
                                &size_log, &(state.raygen_programs[0]));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, "
              "&(pg_options), log, &size_log, &(state.raygen_programs[0]))")
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
void createExceptionPrograms() {
  state.exception_programs.resize(1);
  OptixProgramGroupOptions pg_options;
  OptixProgramGroupDesc pg_desc;
  pg_options = {};
  ;
  pg_desc = {};
  pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_EXCEPTION;
  pg_desc.exception.module = state.module;
  pg_desc.exception.entryFunctionName = "__exception__all";
  char log[2048];
  auto size_log = sizeof(log);
  {
    OptixResult res =
        optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, &(pg_options), log,
                                &size_log, &(state.exception_programs[0]));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, "
              "&(pg_options), log, &size_log, &(state.exception_programs[0]))")
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
  state.hitgroup_programs.resize(1);
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
                                &size_log, &(state.hitgroup_programs[0]));
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixProgramGroupCreate(state.oxctx, &(pg_desc), 1, "
              "&(pg_options), log, &size_log, &(state.hitgroup_programs[0]))")
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
  for (auto &p : state.raygen_programs) {
    program_groups.push_back(p);
  };
  for (auto &p : state.miss_programs) {
    program_groups.push_back(p);
  };
  for (auto &p : state.hitgroup_programs) {
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
}
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) raygen_record_t {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};
typedef struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) raygen_record_t
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) raygen_record_t;
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) miss_record_t {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  void *data;
};
typedef struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) miss_record_t
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) miss_record_t;
struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) hitgroup_record_t {
  __align__(
      OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
  int object_id;
};
typedef struct __align__(OPTIX_SBT_RECORD_ALIGNMENT) hitgroup_record_t
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) hitgroup_record_t;
void buildSBT() {
  std::vector<raygen_record_t> raygen_records;
  for (int i = 0; i < state.raygen_programs.size(); (i) += (1)) {
    raygen_record_t rec;
    {
      OptixResult res =
          optixSbtRecordPackHeader(state.raygen_programs[i], &(rec));
      if (!((OPTIX_SUCCESS) == (res))) {

        (std::cout)
            << (std::setw(10))
            << (std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count())
            << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
            << (":") << (__LINE__) << (" ") << (__func__) << (" ")
            << ("FAIL: optix "
                "optixSbtRecordPackHeader(state.raygen_programs[i], &(rec))")
            << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
            << (std::flush);
      };
    };
    raygen_records.push_back(rec);
  }
  state.raygen_records_buffer.alloc_and_upload(raygen_records);
  state.shader_bindings_table.raygenRecord =
      state.raygen_records_buffer.d_pointer();
  std::vector<miss_record_t> miss_records;
  for (int i = 0; i < state.miss_programs.size(); (i) += (1)) {
    miss_record_t rec;
    {
      OptixResult res =
          optixSbtRecordPackHeader(state.miss_programs[i], &(rec));
      if (!((OPTIX_SUCCESS) == (res))) {

        (std::cout)
            << (std::setw(10))
            << (std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count())
            << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
            << (":") << (__LINE__) << (" ") << (__func__) << (" ")
            << ("FAIL: optix optixSbtRecordPackHeader(state.miss_programs[i], "
                "&(rec))")
            << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
            << (std::flush);
      };
    };
    miss_records.push_back(rec);
  }
  state.miss_records_buffer.alloc_and_upload(miss_records);
  state.shader_bindings_table.missRecordBase =
      state.miss_records_buffer.d_pointer();
  state.shader_bindings_table.missRecordStrideInBytes = sizeof(miss_record_t);
  state.shader_bindings_table.missRecordCount =
      static_cast<int>(miss_records.size());
  std::vector<hitgroup_record_t> hitgroup_records;
  for (int i = 0; i < state.hitgroup_programs.size(); (i) += (1)) {
    hitgroup_record_t rec;
    {
      OptixResult res =
          optixSbtRecordPackHeader(state.hitgroup_programs[i], &(rec));
      if (!((OPTIX_SUCCESS) == (res))) {

        (std::cout)
            << (std::setw(10))
            << (std::chrono::high_resolution_clock::now()
                    .time_since_epoch()
                    .count())
            << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
            << (":") << (__LINE__) << (" ") << (__func__) << (" ")
            << ("FAIL: optix "
                "optixSbtRecordPackHeader(state.hitgroup_programs[i], &(rec))")
            << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
            << (std::flush);
      };
    };
    hitgroup_records.push_back(rec);
  }
  state.hitgroup_records_buffer.alloc_and_upload(hitgroup_records);
  state.shader_bindings_table.hitgroupRecordBase =
      state.hitgroup_records_buffer.d_pointer();
  state.shader_bindings_table.hitgroupRecordStrideInBytes =
      sizeof(hitgroup_record_t);
  state.shader_bindings_table.hitgroupRecordCount =
      static_cast<int>(hitgroup_records.size());
}
void render() {
  if ((0) == (state.launch_params.fbSize_x)) {

    (std::cout) << (std::setw(10))
                << (std::chrono::high_resolution_clock::now()
                        .time_since_epoch()
                        .count())
                << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
                << (":") << (__LINE__) << (" ") << (__func__) << (" ")
                << ("can't launch because first resize hasn't happened")
                << (" ") << (std::endl) << (std::flush);
    return;
  };
  state.launch_params_buffer.upload(&(state.launch_params), 1);
  (state.launch_params.frameID)++;
  camera_t camera = {glm::vec3((-1.e+1f), (2.e+0f), (-1.2e+1f)),
                     glm::vec3((0.0e+0f), (0.0e+0f), (0.0e+0f)),
                     glm::vec3((0.0e+0f), (1.e+0f), (0.0e+0f))};
  set_camera(camera);
  {
    OptixResult res = optixLaunch(
        state.pipeline, state.stream, state.launch_params_buffer.d_pointer(),
        state.launch_params_buffer._size_in_bytes,
        &(state.shader_bindings_table), state.launch_params.fbSize_x,
        state.launch_params.fbSize_y, 1);
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixLaunch(state.pipeline, state.stream, "
              "state.launch_params_buffer.d_pointer(), "
              "state.launch_params_buffer._size_in_bytes, "
              "&(state.shader_bindings_table), state.launch_params.fbSize_x, "
              "state.launch_params.fbSize_y, 1)")
          << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
          << (std::flush);
    };
  };
  {
    {
      auto res = cudaDeviceSynchronize();
      if (!((cudaSuccess) == (res))) {
        auto err_ = cudaGetLastError();
        auto err_name = cudaGetErrorName(err_);
        auto err_str = cudaGetErrorString(err_);

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ")
                    << ("FAIL: cuda after launch cudaDeviceSynchronize()")
                    << (" ") << (std::setw(8)) << (" res=") << (res)
                    << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                    << (" err_name=") << (err_name) << (std::setw(8))
                    << (" err_str=") << (err_str) << (std::endl)
                    << (std::flush);
        throw std::runtime_error("cudaDeviceSynchronize()");
      };
    };
  };
}
void resize(int x, int y) {
  state.color_buffer.resize(((x) * (y) * (sizeof(uint32_t))));
  state.launch_params.fbSize_x = x;
  state.launch_params.fbSize_y = y;
  state.launch_params.colorBuffer =
      static_cast<uint32_t *>(state.color_buffer._d_ptr);
  set_camera(state.last_set_camera);
}
void download_pixels(uint32_t *h_pixels) {
  state.color_buffer.download(h_pixels, ((state.launch_params.fbSize_x) *
                                         (state.launch_params.fbSize_y)));
}
void set_camera(const camera_t &camera) {
  state.last_set_camera = camera;
  auto cos_fov_y = (6.6e-1f);
  auto aspect = ((static_cast<float>(state.launch_params.fbSize_x)) /
                 (state.launch_params.fbSize_y));
  state.launch_params.camera_position = camera.from;
  state.launch_params.camera_direction =
      glm::normalize(((camera.at) - (camera.from)));
  state.launch_params.camera_horizontal =
      ((cos_fov_y) * (aspect) *
       (glm::normalize(
           glm::cross(state.launch_params.camera_direction, camera.up))));
  state.launch_params.camera_vertical =
      ((cos_fov_y) *
       (glm::normalize(glm::cross(state.launch_params.camera_horizontal,
                                  state.launch_params.camera_direction))));
}
OptixTraversableHandle buildAccel(const triangle_mesh_t &model) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ")
      << ("start building acceleration structure") << (" ") << (std::endl)
      << (std::flush);
  state.vertex_buffer.alloc_and_upload(model._vertex);
  state.index_buffer.alloc_and_upload(model._index);
  OptixTraversableHandle handle = {0};
  OptixBuildInput triangle_input = {};
  auto d_vertices = state.vertex_buffer.d_pointer();
  auto d_indices = state.index_buffer.d_pointer();
  uint32_t triangle_input_flags[] = {0};
  triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
  triangle_input.triangleArray.vertexStrideInBytes = sizeof(glm::vec3);
  triangle_input.triangleArray.numVertices =
      static_cast<int>(model._vertex.size());
  triangle_input.triangleArray.vertexBuffers = &d_vertices;
  triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
  triangle_input.triangleArray.indexStrideInBytes = sizeof(glm::ivec3);
  triangle_input.triangleArray.numIndexTriplets =
      static_cast<int>(model._index.size());
  triangle_input.triangleArray.indexBuffer = d_indices;
  triangle_input.triangleArray.flags = triangle_input_flags;
  triangle_input.triangleArray.numSbtRecords = 1;
  triangle_input.triangleArray.sbtIndexOffsetBuffer = 0;
  triangle_input.triangleArray.sbtIndexOffsetSizeInBytes = 0;
  triangle_input.triangleArray.sbtIndexOffsetStrideInBytes = 0;
  OptixAccelBuildOptions accel_options = {};
  accel_options.buildFlags =
      ((OPTIX_BUILD_FLAG_NONE) | (OPTIX_BUILD_FLAG_ALLOW_COMPACTION));
  accel_options.motionOptions.numKeys = 1;
  accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
  OptixAccelBufferSizes blas_buffer_sizes;
  {
    OptixResult res = optixAccelComputeMemoryUsage(
        state.oxctx, &accel_options, &triangle_input, 1, &blas_buffer_sizes);
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("FAIL: optix optixAccelComputeMemoryUsage(state.oxctx, "
                      "&accel_options, &triangle_input, 1, &blas_buffer_sizes)")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("prepare compaction")
      << (" ") << (std::endl) << (std::flush);
  CUDABuffer compacted_size_buffer;
  OptixAccelEmitDesc emit_desc;
  compacted_size_buffer.alloc(sizeof(uint64_t));
  emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
  emit_desc.result = compacted_size_buffer.d_pointer();

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("execute build")
      << (" ") << (std::endl) << (std::flush);
  CUDABuffer temp_buffer;
  CUDABuffer output_buffer;
  temp_buffer.alloc(blas_buffer_sizes.tempSizeInBytes);
  output_buffer.alloc(blas_buffer_sizes.outputSizeInBytes);
  {
    OptixResult res =
        optixAccelBuild(state.oxctx, 0, &accel_options, &triangle_input, 1,
                        temp_buffer.d_pointer(), temp_buffer._size_in_bytes,
                        output_buffer.d_pointer(), output_buffer._size_in_bytes,
                        &handle, &emit_desc, 1);
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout)
          << (std::setw(10))
          << (std::chrono::high_resolution_clock::now()
                  .time_since_epoch()
                  .count())
          << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__)
          << (":") << (__LINE__) << (" ") << (__func__) << (" ")
          << ("FAIL: optix optixAccelBuild(state.oxctx, 0, &accel_options, "
              "&triangle_input, 1, temp_buffer.d_pointer(), "
              "temp_buffer._size_in_bytes, output_buffer.d_pointer(), "
              "output_buffer._size_in_bytes, &handle, &emit_desc, 1)")
          << (" ") << (std::setw(8)) << (" res=") << (res) << (std::endl)
          << (std::flush);
    };
  };
  {
    {
      auto res = cudaDeviceSynchronize();
      if (!((cudaSuccess) == (res))) {
        auto err_ = cudaGetLastError();
        auto err_name = cudaGetErrorName(err_);
        auto err_str = cudaGetErrorString(err_);

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ") << ("FAIL: cuda build cudaDeviceSynchronize()")
                    << (" ") << (std::setw(8)) << (" res=") << (res)
                    << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                    << (" err_name=") << (err_name) << (std::setw(8))
                    << (" err_str=") << (err_str) << (std::endl)
                    << (std::flush);
        throw std::runtime_error("cudaDeviceSynchronize()");
      };
    };
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("perform compaction")
      << (" ") << (std::endl) << (std::flush);
  uint64_t compacted_size;
  compacted_size_buffer.download(&compacted_size, 1);
  state.accel_buffer.alloc(compacted_size);
  {
    OptixResult res = optixAccelCompact(
        state.oxctx, 0, handle, (state.accel_buffer.d_pointer)(),
        state.accel_buffer._size_in_bytes, &handle);
    if (!((OPTIX_SUCCESS) == (res))) {

      (std::cout) << (std::setw(10))
                  << (std::chrono::high_resolution_clock::now()
                          .time_since_epoch()
                          .count())
                  << (" ") << (std::this_thread::get_id()) << (" ")
                  << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                  << (" ")
                  << ("FAIL: optix optixAccelCompact(state.oxctx, 0, handle, "
                      "(state.accel_buffer.d_pointer)(), "
                      "state.accel_buffer._size_in_bytes, &handle)")
                  << (" ") << (std::setw(8)) << (" res=") << (res)
                  << (std::endl) << (std::flush);
    };
  };
  {
    {
      auto res = cudaDeviceSynchronize();
      if (!((cudaSuccess) == (res))) {
        auto err_ = cudaGetLastError();
        auto err_name = cudaGetErrorName(err_);
        auto err_str = cudaGetErrorString(err_);

        (std::cout) << (std::setw(10))
                    << (std::chrono::high_resolution_clock::now()
                            .time_since_epoch()
                            .count())
                    << (" ") << (std::this_thread::get_id()) << (" ")
                    << (__FILE__) << (":") << (__LINE__) << (" ") << (__func__)
                    << (" ")
                    << ("FAIL: cuda compaction cudaDeviceSynchronize()")
                    << (" ") << (std::setw(8)) << (" res=") << (res)
                    << (std::setw(8)) << (" err_=") << (err_) << (std::setw(8))
                    << (" err_name=") << (err_name) << (std::setw(8))
                    << (" err_str=") << (err_str) << (std::endl)
                    << (std::flush);
        throw std::runtime_error("cudaDeviceSynchronize()");
      };
    };
  };

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("clean up") << (" ")
      << (std::endl) << (std::flush);
  output_buffer.free();
  temp_buffer.free();
  compacted_size_buffer.free();
  return handle;
}
void initOptix(const triangle_mesh_t &model) {

  (std::cout)
      << (std::setw(10))
      << (std::chrono::high_resolution_clock::now().time_since_epoch().count())
      << (" ") << (std::this_thread::get_id()) << (" ") << (__FILE__) << (":")
      << (__LINE__) << (" ") << (__func__) << (" ") << ("initOptix") << (" ")
      << (std::endl) << (std::flush);
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
  state.launch_params.traversable = buildAccel(model);
  createPipeline();
  buildSBT();
  state.launch_params_buffer.alloc(sizeof(LaunchParams));
}
void cleanupOptix(){};