# Silero VAD Go - 单模型多线程版本

这个实现提供了一个单模型多线程版本的 Silero VAD，允许多个协程共享同一个模型实例，但每个协程维护独立的状态。

## 主要特性

1. **内存效率**: 模型只加载一次，多个协程共享
2. **线程安全**: 使用读写锁保护共享资源
3. **独立状态**: 每个协程有独立的检测状态
4. **高性能**: 避免了重复的模型加载

## 核心组件

### SharedModel
- 包含可共享的 ONNX Runtime 资源
- 模型、会话、API 等资源只初始化一次
- 使用读写锁确保线程安全

### DetectorContext
- 每个协程的独立检测上下文
- 维护独立的状态信息
- 轻量级，可以频繁创建

## 使用示例

### 基本用法

```go
// 1. 创建共享模型（全局，只创建一次）
sharedModel, err := speech.NewSharedModel(speech.DetectorConfig{
    ModelPath:            "./testfiles/silero_vad.onnx",
    SampleRate:           16000,
    Threshold:            0.5,
    MinSilenceDurationMs: 100,
    SpeechPadMs:          30,
    LogLevel:             speech.LogLevelInfo,
})
if err != nil {
    log.Fatal(err)
}
defer sharedModel.Destroy()

// 2. 在协程中使用
go func() {
    // 每个协程创建自己的上下文
    context := sharedModel.NewContext()
    
    // 处理音频数据
    segments, err := context.Detect(pcmData)
    if err != nil {
        log.Printf("Detection failed: %v", err)
        return
    }
    
    // 处理检测结果...
}()
```

### 并发处理示例

```go
var wg sync.WaitGroup
numWorkers := 4

for i := 0; i < numWorkers; i++ {
    wg.Add(1)
    go func(workerID int) {
        defer wg.Done()
        
        // 每个协程独立的上下文
        context := sharedModel.NewContext()
        
        // 处理音频数据
        segments, err := context.Detect(audioData[workerID])
        if err != nil {
            log.Printf("Worker %d error: %v", workerID, err)
            return
        }
        
        // 处理结果...
    }(i)
}
wg.Wait()
```

### 流式处理示例

```go
context := sharedModel.NewContext()

for {
    // 从音频流读取数据
    chunk, err := readAudioChunk()
    if err != nil {
        break
    }
    
    // 检测语音
    segments, err := context.Detect(chunk)
    if err != nil {
        log.Printf("Detection error: %v", err)
        continue
    }
    
    // 处理检测到的语音片段
    for _, segment := range segments {
        fmt.Printf("Speech: %.2f-%.2f seconds\n", 
            segment.SpeechStartAt, segment.SpeechEndAt)
    }
}
```

### 快速语音检测方法

除了完整的语音段检测，还提供了两个快速检测方法：

#### IsSpeech 方法
```go
// 检测整个音频是否包含人声
hasSpeech, err := context.IsSpeech(pcmData)
if err != nil {
    log.Printf("Detection failed: %v", err)
} else if hasSpeech {
    fmt.Println("Audio contains speech")
} else {
    fmt.Println("Audio contains no speech")
}
```

#### IsSpeechQuick 方法
```go
// 只检测前几个窗口，适用于需要极快响应的场景
maxWindows := 3 // 只检测前3个窗口
hasSpeech, err := context.IsSpeechQuick(pcmData, maxWindows)
if err != nil {
    log.Printf("Quick detection failed: %v", err)
} else {
    fmt.Printf("Quick check result: %v\n", hasSpeech)
}
```

#### 性能对比

- **Detect()**: 完整分析，返回所有语音段的时间信息
- **IsSpeech()**: 一旦检测到语音就返回，比完整检测更快
- **IsSpeechQuick()**: 只检测指定数量的窗口，最快的检测方式

## API 参考

### SharedModel 方法

- `NewSharedModel(cfg DetectorConfig) (*SharedModel, error)`: 创建共享模型
- `NewContext() *DetectorContext`: 创建新的检测上下文
- `Destroy() error`: 销毁共享模型资源
- `GetConfig() DetectorConfig`: 获取配置信息

### DetectorContext 方法

- `Detect(pcm []float32) ([]Segment, error)`: 检测语音片段
- `IsSpeech(pcm []float32) (bool, error)`: 检测音频是否包含人声
- `IsSpeechQuick(pcm []float32, maxWindows int) (bool, error)`: 快速检测音频是否包含人声
- `Reset() error`: 重置检测状态
- `SetThreshold(value float32)`: 设置检测阈值

## 性能对比

### 传统方式（每个协程独立模型）
- 内存使用: N * 模型大小
- 初始化时间: N * 模型加载时间
- 并发安全: 天然隔离

### 新方式（共享模型）
- 内存使用: 1 * 模型大小 + N * 状态大小
- 初始化时间: 1 * 模型加载时间
- 并发安全: 读写锁保护

## 注意事项

1. **资源管理**: 确保在程序结束前调用 `sharedModel.Destroy()`
2. **生命周期**: SharedModel 的生命周期应该长于所有 DetectorContext
3. **错误处理**: 模型初始化失败时，所有协程都无法工作
4. **平台支持**: 目前支持 Darwin 和 Linux 平台

## 构建和运行

```bash
# 构建示例程序
go build -o shared_vad ./cmd/shared_example.go

# 运行示例
./shared_vad
```

## 文件结构

```
speech/
├── shared_detector.go       # 共享模型和上下文定义
├── shared_infer_darwin.go   # macOS 平台的推理实现
├── shared_infer_linux.go    # Linux 平台的推理实现
└── detector.go             # 原始的单线程实现
```

这个实现为需要高并发语音检测的应用提供了一个高效、内存友好的解决方案。
