#ifndef SILERO_VAD_BRIDGE_H
#define SILERO_VAD_BRIDGE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define VAD_EXPORT __declspec(dllexport)
#else
#define VAD_EXPORT __attribute__((visibility("default")))
#endif

typedef struct VadHandle VadHandle;

VAD_EXPORT int VadCreate(
    const char* model_path,
    int sample_rate,
    int log_level,
    VadHandle** out_handle,
    char* err_buf,
    size_t err_buf_len);

VAD_EXPORT int VadInfer(
    VadHandle* handle,
    const float* pcm,
    size_t pcm_len,
    float* state_inout,
    float* prob_out,
    char* err_buf,
    size_t err_buf_len);

VAD_EXPORT void VadDestroy(VadHandle* handle);

#ifdef __cplusplus
}
#endif

#endif
