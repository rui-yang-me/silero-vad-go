package speech

// #cgo CFLAGS: -Wall -Werror -std=c99
// #cgo LDFLAGS: -lonnxruntime
// #include "ort_bridge.h"
import "C"

import (
	"fmt"
	"log/slog"
	"sync"
	"unsafe"
)

// SharedModel 包含可共享的 ONNX 运行时资源
type SharedModel struct {
	api         *C.OrtApi
	env         *C.OrtEnv
	sessionOpts *C.OrtSessionOptions
	session     *C.OrtSession
	memoryInfo  *C.OrtMemoryInfo
	cStrings    map[string]*C.char
	cfg         DetectorConfig
	mu          sync.RWMutex // 保护共享资源的读写锁
}

// DetectorContext 包含每个检测器的独立状态
type DetectorContext struct {
	model      *SharedModel
	state      [stateLen]float32
	ctx        [contextLen]float32
	currSample int
	triggered  bool
	tempEnd    int
}

// NewSharedModel 创建一个可共享的模型实例
func NewSharedModel(cfg DetectorConfig) (*SharedModel, error) {
	if err := cfg.IsValid(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	sm := &SharedModel{
		cfg:      cfg,
		cStrings: map[string]*C.char{},
	}

	// 获取 ONNX Runtime API
	sm.api = C.OrtGetApi()
	if sm.api == nil {
		return nil, fmt.Errorf("failed to get API")
	}

	// 创建环境
	sm.cStrings["loggerName"] = C.CString("vad_shared")
	status := C.OrtApiCreateEnv(sm.api, cfg.LogLevel.OrtLoggingLevel(), sm.cStrings["loggerName"], &sm.env)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create env: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	// 创建会话选项
	status = C.OrtApiCreateSessionOptions(sm.api, &sm.sessionOpts)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session options: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	// 设置线程数
	status = C.OrtApiSetIntraOpNumThreads(sm.api, sm.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set intra threads: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	status = C.OrtApiSetInterOpNumThreads(sm.api, sm.sessionOpts, 1)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set inter threads: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	// 设置图优化级别
	status = C.OrtApiSetSessionGraphOptimizationLevel(sm.api, sm.sessionOpts, C.ORT_ENABLE_ALL)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to set session graph optimization level: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	// 创建会话
	sm.cStrings["modelPath"] = C.CString(sm.cfg.ModelPath)
	status = C.OrtApiCreateSession(sm.api, sm.env, sm.cStrings["modelPath"], sm.sessionOpts, &sm.session)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create session: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	// 创建内存信息
	status = C.OrtApiCreateCpuMemoryInfo(sm.api, C.OrtArenaAllocator, C.OrtMemTypeDefault, &sm.memoryInfo)
	defer C.OrtApiReleaseStatus(sm.api, status)
	if status != nil {
		return nil, fmt.Errorf("failed to create memory info: %s", C.GoString(C.OrtApiGetErrorMessage(sm.api, status)))
	}

	// 创建输入输出名称的C字符串
	sm.cStrings["input"] = C.CString("input")
	sm.cStrings["sr"] = C.CString("sr")
	sm.cStrings["state"] = C.CString("state")
	sm.cStrings["stateN"] = C.CString("stateN")
	sm.cStrings["output"] = C.CString("output")

	return sm, nil
}

// NewContext 创建一个新的检测器上下文
func (sm *SharedModel) NewContext() *DetectorContext {
	return &DetectorContext{
		model: sm,
	}
}

// Destroy 销毁共享模型资源
func (sm *SharedModel) Destroy() error {
	if sm == nil {
		return fmt.Errorf("invalid nil shared model")
	}

	sm.mu.Lock()
	defer sm.mu.Unlock()

	C.OrtApiReleaseMemoryInfo(sm.api, sm.memoryInfo)
	C.OrtApiReleaseSession(sm.api, sm.session)
	C.OrtApiReleaseSessionOptions(sm.api, sm.sessionOpts)
	C.OrtApiReleaseEnv(sm.api, sm.env)

	for _, ptr := range sm.cStrings {
		C.free(unsafe.Pointer(ptr))
	}

	return nil
}

// GetConfig 获取配置（线程安全）
func (sm *SharedModel) GetConfig() DetectorConfig {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.cfg
}

// Detect 检测语音片段
func (dc *DetectorContext) Detect(pcm []float32) ([]Segment, error) {
	if dc == nil || dc.model == nil {
		return nil, fmt.Errorf("invalid nil detector context")
	}

	windowSize := 512
	if dc.model.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	if len(pcm) < windowSize {
		return nil, fmt.Errorf("not enough samples")
	}

	slog.Debug("starting speech detection", slog.Int("samplesLen", len(pcm)))

	minSilenceSamples := dc.model.cfg.MinSilenceDurationMs * dc.model.cfg.SampleRate / 1000
	speechPadSamples := dc.model.cfg.SpeechPadMs * dc.model.cfg.SampleRate / 1000

	var segments []Segment
	for i := 0; i < len(pcm)-windowSize; i += windowSize {
		speechProb, err := dc.infer(pcm[i : i+windowSize])
		// if speechProb >= 0.5 {
		// 	fmt.Printf("===infer speech prob: %f\n", speechProb)
		// }
		if err != nil {
			return nil, fmt.Errorf("infer failed: %w", err)
		}

		dc.currSample += windowSize

		if speechProb >= dc.model.cfg.Threshold && dc.tempEnd != 0 {
			dc.tempEnd = 0
		}

		if speechProb >= dc.model.cfg.Threshold && !dc.triggered {
			dc.triggered = true
			speechStartAt := (float64(dc.currSample-windowSize-speechPadSamples) / float64(dc.model.cfg.SampleRate))

			// 由于padding的存在，起始位置可能为负数，我们将其限制在0
			if speechStartAt < 0 {
				speechStartAt = 0
			}

			slog.Debug("speech start", slog.Float64("startAt", speechStartAt))
			segments = append(segments, Segment{
				SpeechStartAt: speechStartAt,
			})
		}

		if speechProb < (dc.model.cfg.Threshold-0.15) && dc.triggered {
			if dc.tempEnd == 0 {
				dc.tempEnd = dc.currSample
			}

			// 静音时间不够长，继续等待
			if dc.currSample-dc.tempEnd < minSilenceSamples {
				continue
			}

			speechEndAt := (float64(dc.tempEnd+speechPadSamples) / float64(dc.model.cfg.SampleRate))
			dc.tempEnd = 0
			dc.triggered = false
			slog.Debug("speech end", slog.Float64("endAt", speechEndAt))

			if len(segments) < 1 {
				return nil, fmt.Errorf("unexpected speech end")
			}

			segments[len(segments)-1].SpeechEndAt = speechEndAt
		}
	}

	slog.Debug("speech detection done", slog.Int("segmentsLen", len(segments)))

	return segments, nil
}

// Reset 重置检测器状态
func (dc *DetectorContext) Reset() error {
	if dc == nil {
		return fmt.Errorf("invalid nil detector context")
	}

	dc.currSample = 0
	dc.triggered = false
	dc.tempEnd = 0
	for i := 0; i < stateLen; i++ {
		dc.state[i] = 0
	}
	for i := 0; i < contextLen; i++ {
		dc.ctx[i] = 0
	}

	return nil
}

// SetThreshold 设置阈值
func (dc *DetectorContext) SetThreshold(value float32) {
	if dc != nil && dc.model != nil {
		dc.model.mu.Lock()
		dc.model.cfg.Threshold = value
		dc.model.mu.Unlock()
	}
}

// IsSpeech 检测音频中是否包含人声，返回 true/false
// 这是一个优化的方法，一旦检测到人声就立即返回，无需处理完整音频
func (dc *DetectorContext) IsSpeech(pcm []float32) (bool, error) {
	if dc == nil || dc.model == nil {
		return false, fmt.Errorf("invalid nil detector context")
	}

	windowSize := 512
	if dc.model.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	if len(pcm) < windowSize {
		return false, fmt.Errorf("not enough samples")
	}

	slog.Debug("starting speech detection (IsSpeech)", slog.Int("samplesLen", len(pcm)))

	// 重置状态以确保检测的准确性
	dc.currSample = 0
	dc.triggered = false
	dc.tempEnd = 0
	for i := 0; i < stateLen; i++ {
		dc.state[i] = 0
	}

	// 遍历音频窗口
	for i := 0; i < len(pcm)-windowSize; i += windowSize {
		speechProb, err := dc.infer(pcm[i : i+windowSize])
		if err != nil {
			return false, fmt.Errorf("infer failed: %w", err)
		}

		dc.currSample += windowSize

		// 如果检测到语音概率超过阈值，立即返回 true
		if speechProb >= dc.model.cfg.Threshold {
			slog.Debug("speech detected", slog.Float64("probability", float64(speechProb)))
			return true, nil
		}
	}

	slog.Debug("no speech detected")
	return false, nil
}

// IsSpeechQuick 快速检测音频中是否包含人声
// 只检测前几个窗口，适用于需要极快响应的场景
func (dc *DetectorContext) IsSpeechQuick(pcm []float32, maxWindows int) (bool, error) {
	if dc == nil || dc.model == nil {
		return false, fmt.Errorf("invalid nil detector context")
	}

	windowSize := 512
	if dc.model.cfg.SampleRate == 8000 {
		windowSize = 256
	}

	if len(pcm) < windowSize {
		return false, fmt.Errorf("not enough samples")
	}

	if maxWindows <= 0 {
		maxWindows = 5 // 默认检测前5个窗口
	}

	slog.Debug("starting quick speech detection",
		slog.Int("samplesLen", len(pcm)),
		slog.Int("maxWindows", maxWindows))

	// 重置状态
	dc.currSample = 0
	dc.triggered = false
	dc.tempEnd = 0
	for i := 0; i < stateLen; i++ {
		dc.state[i] = 0
	}

	// 只检测指定数量的窗口
	windowCount := 0
	for i := 0; i < len(pcm)-windowSize && windowCount < maxWindows; i += windowSize {
		speechProb, err := dc.infer(pcm[i : i+windowSize])
		if err != nil {
			return false, fmt.Errorf("infer failed: %w", err)
		}

		dc.currSample += windowSize
		windowCount++

		// 如果检测到语音概率超过阈值，立即返回 true
		if speechProb >= dc.model.cfg.Threshold {
			slog.Debug("speech detected quickly",
				slog.Float64("probability", float64(speechProb)),
				slog.Int("windowIndex", windowCount))
			return true, nil
		}
	}

	slog.Debug("no speech detected in quick check")
	return false, nil
}
