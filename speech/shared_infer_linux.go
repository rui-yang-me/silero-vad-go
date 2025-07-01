//go:build !darwin

package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"unsafe"
)

// infer 使用共享模型进行推理，但每个上下文有独立的状态
func (dc *DetectorContext) infer(pcm []float32) (float32, error) {
	if dc == nil || dc.model == nil {
		return 0, fmt.Errorf("invalid detector context")
	}

	// 使用读锁保护共享资源的访问
	dc.model.mu.RLock()
	defer dc.model.mu.RUnlock()

	// 创建PCM输入张量
	var pcmValue *C.OrtValue
	pcmInputDims := []C.long{
		1,
		C.long(len(pcm)),
	}
	status := C.OrtApiCreateTensorWithDataAsOrtValue(
		dc.model.api,
		dc.model.memoryInfo,
		unsafe.Pointer(&pcm[0]),
		C.size_t(len(pcm)*4),
		&pcmInputDims[0],
		C.size_t(len(pcmInputDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&pcmValue,
	)
	defer C.OrtApiReleaseStatus(dc.model.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create pcm value: %s", C.GoString(C.OrtApiGetErrorMessage(dc.model.api, status)))
	}
	defer C.OrtApiReleaseValue(dc.model.api, pcmValue)

	// 创建状态输入张量（使用上下文的独立状态）
	var stateValue *C.OrtValue
	stateNodeInputDims := []C.long{2, 1, 128}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(
		dc.model.api,
		dc.model.memoryInfo,
		unsafe.Pointer(&dc.state[0]),
		C.size_t(stateLen*4),
		&stateNodeInputDims[0],
		C.size_t(len(stateNodeInputDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&stateValue,
	)
	defer C.OrtApiReleaseStatus(dc.model.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create state value: %s", C.GoString(C.OrtApiGetErrorMessage(dc.model.api, status)))
	}
	defer C.OrtApiReleaseValue(dc.model.api, stateValue)

	// 创建采样率输入张量
	var rateValue *C.OrtValue
	rateInputDims := []C.long{1}
	rate := []C.int64_t{C.int64_t(dc.model.cfg.SampleRate)}
	status = C.OrtApiCreateTensorWithDataAsOrtValue(
		dc.model.api,
		dc.model.memoryInfo,
		unsafe.Pointer(&rate[0]),
		C.size_t(8),
		&rateInputDims[0],
		C.size_t(len(rateInputDims)),
		C.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
		&rateValue,
	)
	defer C.OrtApiReleaseStatus(dc.model.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to create rate value: %s", C.GoString(C.OrtApiGetErrorMessage(dc.model.api, status)))
	}
	defer C.OrtApiReleaseValue(dc.model.api, rateValue)

	// 运行推理
	inputs := []*C.OrtValue{pcmValue, stateValue, rateValue}
	outputs := []*C.OrtValue{nil, nil}

	inputNames := []*C.char{
		dc.model.cStrings["input"],
		dc.model.cStrings["state"],
		dc.model.cStrings["sr"],
	}
	outputNames := []*C.char{
		dc.model.cStrings["output"],
		dc.model.cStrings["stateN"],
	}

	status = C.OrtApiRun(
		dc.model.api,
		dc.model.session,
		nil,
		&inputNames[0],
		&inputs[0],
		C.size_t(len(inputNames)),
		&outputNames[0],
		C.size_t(len(outputNames)),
		&outputs[0],
	)
	defer C.OrtApiReleaseStatus(dc.model.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to run inference: %s", C.GoString(C.OrtApiGetErrorMessage(dc.model.api, status)))
	}

	// 获取输出张量数据
	var prob unsafe.Pointer
	var stateN unsafe.Pointer

	status = C.OrtApiGetTensorMutableData(dc.model.api, outputs[0], &prob)
	defer C.OrtApiReleaseStatus(dc.model.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get probability tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(dc.model.api, status)))
	}

	status = C.OrtApiGetTensorMutableData(dc.model.api, outputs[1], &stateN)
	defer C.OrtApiReleaseStatus(dc.model.api, status)
	if status != nil {
		return 0, fmt.Errorf("failed to get state tensor data: %s", C.GoString(C.OrtApiGetErrorMessage(dc.model.api, status)))
	}

	// 更新上下文的状态（这是每个上下文独立的）
	C.memcpy(unsafe.Pointer(&dc.state[0]), stateN, stateLen*4)

	// 释放输出张量
	C.OrtApiReleaseValue(dc.model.api, outputs[0])
	C.OrtApiReleaseValue(dc.model.api, outputs[1])

	// 返回语音概率
	return *(*float32)(prob), nil
}
