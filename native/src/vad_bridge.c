#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "onnxruntime_c_api.h"
#include "vad_bridge.h"

enum {
  kStateLen = 2 * 1 * 128,
};

struct VadHandle {
  const OrtApi* api;
  OrtEnv* env;
  OrtSessionOptions* session_opts;
  OrtSession* session;
  OrtMemoryInfo* memory_info;
  int64_t sample_rate;
};

static void set_error(char* err_buf, size_t err_buf_len, const char* msg) {
  if (err_buf == NULL || err_buf_len == 0) {
    return;
  }
  if (msg == NULL) {
    err_buf[0] = '\0';
    return;
  }
  snprintf(err_buf, err_buf_len, "%s", msg);
}

static int fail_status(const OrtApi* api, OrtStatus* status, const char* prefix, char* err_buf, size_t err_buf_len) {
  const char* msg = api->GetErrorMessage(status);
  if (prefix != NULL && prefix[0] != '\0') {
    if (err_buf != NULL && err_buf_len > 0) {
      snprintf(err_buf, err_buf_len, "%s: %s", prefix, msg);
    }
  } else {
    set_error(err_buf, err_buf_len, msg);
  }
  api->ReleaseStatus(status);
  return -1;
}

int VadCreate(
    const char* model_path,
    int sample_rate,
    int log_level,
    VadHandle** out_handle,
    char* err_buf,
    size_t err_buf_len) {
  OrtStatus* status = NULL;
  VadHandle* h = NULL;
  const OrtApi* api = NULL;

  set_error(err_buf, err_buf_len, NULL);

  if (model_path == NULL || out_handle == NULL) {
    set_error(err_buf, err_buf_len, "invalid arguments");
    return -1;
  }

  api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  if (api == NULL) {
    set_error(err_buf, err_buf_len, "failed to get ONNX Runtime API");
    return -1;
  }

  h = (VadHandle*)calloc(1, sizeof(VadHandle));
  if (h == NULL) {
    set_error(err_buf, err_buf_len, "out of memory");
    return -1;
  }
  h->api = api;
  h->sample_rate = (int64_t)sample_rate;

  status = api->CreateEnv((OrtLoggingLevel)log_level, "vad", &h->env);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to create env", err_buf, err_buf_len);
  }

  status = api->CreateSessionOptions(&h->session_opts);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to create session options", err_buf, err_buf_len);
  }

  status = api->SetIntraOpNumThreads(h->session_opts, 1);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to set intra threads", err_buf, err_buf_len);
  }

  status = api->SetInterOpNumThreads(h->session_opts, 1);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to set inter threads", err_buf, err_buf_len);
  }

  status = api->SetSessionGraphOptimizationLevel(h->session_opts, ORT_ENABLE_ALL);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to set optimization level", err_buf, err_buf_len);
  }

  status = api->CreateSession(h->env, model_path, h->session_opts, &h->session);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to create session", err_buf, err_buf_len);
  }

  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &h->memory_info);
  if (status != NULL) {
    VadDestroy(h);
    return fail_status(api, status, "failed to create memory info", err_buf, err_buf_len);
  }

  *out_handle = h;
  return 0;
}

int VadInfer(
    VadHandle* handle,
    const float* pcm,
    size_t pcm_len,
    float* state_inout,
    float* prob_out,
    char* err_buf,
    size_t err_buf_len) {
  OrtStatus* status = NULL;
  const OrtApi* api = NULL;
  OrtValue* pcm_value = NULL;
  OrtValue* state_value = NULL;
  OrtValue* rate_value = NULL;
  OrtValue* outputs[2] = {NULL, NULL};
  int64_t pcm_dims[2] = {1, 0};
  int64_t state_dims[3] = {2, 1, 128};
  int64_t rate_dims[1] = {1};
  const char* input_names[3] = {"input", "state", "sr"};
  const char* output_names[2] = {"output", "stateN"};
  int64_t rate[1] = {0};
  float* prob_ptr = NULL;
  float* state_n_ptr = NULL;
  void* raw_ptr = NULL;

  set_error(err_buf, err_buf_len, NULL);

  if (handle == NULL || pcm == NULL || pcm_len == 0 || state_inout == NULL || prob_out == NULL) {
    set_error(err_buf, err_buf_len, "invalid arguments");
    return -1;
  }

  api = handle->api;
  rate[0] = handle->sample_rate;
  pcm_dims[1] = (int64_t)pcm_len;

  status = api->CreateTensorWithDataAsOrtValue(
      handle->memory_info,
      (void*)pcm,
      pcm_len * sizeof(float),
      pcm_dims,
      2,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &pcm_value);
  if (status != NULL) {
    return fail_status(api, status, "failed to create pcm tensor", err_buf, err_buf_len);
  }

  status = api->CreateTensorWithDataAsOrtValue(
      handle->memory_info,
      (void*)state_inout,
      kStateLen * sizeof(float),
      state_dims,
      3,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
      &state_value);
  if (status != NULL) {
    api->ReleaseValue(pcm_value);
    return fail_status(api, status, "failed to create state tensor", err_buf, err_buf_len);
  }

  status = api->CreateTensorWithDataAsOrtValue(
      handle->memory_info,
      (void*)rate,
      sizeof(rate),
      rate_dims,
      1,
      ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
      &rate_value);
  if (status != NULL) {
    api->ReleaseValue(state_value);
    api->ReleaseValue(pcm_value);
    return fail_status(api, status, "failed to create rate tensor", err_buf, err_buf_len);
  }

  {
    const OrtValue* inputs[3] = {pcm_value, state_value, rate_value};
    status = api->Run(
        handle->session,
        NULL,
        input_names,
        inputs,
        3,
        output_names,
        2,
        outputs);
  }
  if (status != NULL) {
    api->ReleaseValue(rate_value);
    api->ReleaseValue(state_value);
    api->ReleaseValue(pcm_value);
    return fail_status(api, status, "failed to run inference", err_buf, err_buf_len);
  }

  status = api->GetTensorMutableData(outputs[0], &raw_ptr);
  if (status != NULL) {
    api->ReleaseValue(outputs[1]);
    api->ReleaseValue(outputs[0]);
    api->ReleaseValue(rate_value);
    api->ReleaseValue(state_value);
    api->ReleaseValue(pcm_value);
    return fail_status(api, status, "failed to read output tensor", err_buf, err_buf_len);
  }
  prob_ptr = (float*)raw_ptr;

  status = api->GetTensorMutableData(outputs[1], &raw_ptr);
  if (status != NULL) {
    api->ReleaseValue(outputs[1]);
    api->ReleaseValue(outputs[0]);
    api->ReleaseValue(rate_value);
    api->ReleaseValue(state_value);
    api->ReleaseValue(pcm_value);
    return fail_status(api, status, "failed to read state tensor", err_buf, err_buf_len);
  }
  state_n_ptr = (float*)raw_ptr;

  *prob_out = prob_ptr[0];
  memcpy(state_inout, state_n_ptr, kStateLen * sizeof(float));

  api->ReleaseValue(outputs[1]);
  api->ReleaseValue(outputs[0]);
  api->ReleaseValue(rate_value);
  api->ReleaseValue(state_value);
  api->ReleaseValue(pcm_value);

  return 0;
}

void VadDestroy(VadHandle* handle) {
  if (handle == NULL) {
    return;
  }
  if (handle->api != NULL) {
    if (handle->memory_info != NULL) {
      handle->api->ReleaseMemoryInfo(handle->memory_info);
    }
    if (handle->session != NULL) {
      handle->api->ReleaseSession(handle->session);
    }
    if (handle->session_opts != NULL) {
      handle->api->ReleaseSessionOptions(handle->session_opts);
    }
    if (handle->env != NULL) {
      handle->api->ReleaseEnv(handle->env);
    }
  }
  free(handle);
}
