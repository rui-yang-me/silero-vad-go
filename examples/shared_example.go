package main

import (
	"fmt"
	"log"
	"math"
	"os"
	"sync"
	"time"

	"github.com/rui-yang-me/silero-vad-go/speech"
)

func main() {
	// load model once
	sharedModel, err := speech.NewSharedModel(speech.DetectorConfig{
		ModelPath:            "../testfiles/silero_vad.onnx",
		SampleRate:           16000,
		Threshold:            0.7,
		MinSilenceDurationMs: 100,
		SpeechPadMs:          30,
		LogLevel:             speech.LogLevelInfo,
	})
	if err != nil {
		log.Fatalf("Failed to create shared model: %v", err)
	}
	defer sharedModel.Destroy()

	var audioData [][]float32
	pcmfiles := []string{"81324-14_05_16_000", "80950-14_01_54_000", "80950-14_01_32_000"}
	for _, pcmfile := range pcmfiles {
		data, err := readPCMFile(fmt.Sprintf("../testfiles/%s.pcm", pcmfile))
		if err != nil {
			log.Printf("Failed to read PCM file %s: %v", pcmfile, err)
			continue
		}
		audioData = append(audioData, data)
	}
	if len(audioData) == 0 {
		log.Fatal("No audio data loaded, please check the PCM files.")
	}
	// 使用多个协程并发处理音频数据
	var wg sync.WaitGroup
	numWorkers := len(audioData)

	startTime := time.Now()

	for i := 0; i < numWorkers; i++ {
		wg.Add(1)
		go func(workerID int) {
			startTime := time.Now()
			defer func() {
				log.Printf("[%s]Worker %d finished in %v", pcmfiles[workerID], workerID, time.Since(startTime))
			}()
			defer wg.Done()

			// 每个协程创建自己的检测器上下文
			context := sharedModel.NewContext()

			// 处理音频数据
			segments, err := context.Detect(audioData[workerID])
			if err != nil {
				log.Printf("[%s]Worker %d failed to detect: %v", pcmfiles[workerID], workerID, err)
				return
			}

			// 打印结果
			fmt.Printf("[%s]Worker %d detected %d segments:\n", pcmfiles[workerID], workerID, len(segments))
			for j, segment := range segments {
				fmt.Printf("[%s]  Segment %d: Start=%.3fs, End=%.3fs\n",
					pcmfiles[workerID], j+1, segment.SpeechStartAt, segment.SpeechEndAt)
			}

			// // 重置上下文状态并再次处理
			// if err := context.Reset(); err != nil {
			// 	log.Printf("[%s]Worker %d failed to reset: %v", pcmfiles[workerID], workerID, err)
			// 	return
			// }

			// // 第二次处理
			// segments2, err := context.Detect(audioData[workerID])
			// if err != nil {
			// 	log.Printf("[%s]Worker %d failed to detect (2nd): %v", pcmfiles[workerID], workerID, err)
			// 	return
			// }

			// fmt.Printf("[%s]Worker %d second detection: %d segments\n", pcmfiles[workerID], workerID, len(segments2))

		}(i)
	}

	wg.Wait()
	fmt.Printf("Total execution time: %v\n", time.Since(startTime))

	// 演示动态阈值调整
	// fmt.Println("\n--- Testing threshold adjustment ---")
	// context := sharedModel.NewContext()
	// context.SetThreshold(0.8) // 设置更高的阈值

	// segments, err := context.Detect(audioData[0])
	// if err != nil {
	// 	log.Printf("Failed to detect with new threshold: %v", err)
	// } else {
	// 	fmt.Printf("With threshold 0.8: %d segments detected\n", len(segments))
	// }

	// 演示 IsSpeech 方法
	fmt.Println("\n--- Testing IsSpeech method ---")
	// context2 := sharedModel.NewContext()
	var wg2 sync.WaitGroup
	for i, pcmfile := range pcmfiles {
		wg2.Add(1)
		go func(i int) {
			startTime := time.Now()
			defer func() {
				log.Printf("[%s] IsSpeech worker %d finished in %v", pcmfile,
					i, time.Since(startTime))
			}()
			defer wg2.Done()
			if i >= len(audioData) {
				return
			}
			// 每个协程创建自己的检测器上下文
			context := sharedModel.NewContext()
			isSpeech, err := context.IsSpeech(audioData[i])
			if err != nil {
				log.Printf("[%s] IsSpeech failed: %v", pcmfile, err)
				return
			}
			fmt.Printf("[%s] IsSpeech result: %v\n", pcmfile, isSpeech)
			// // 重置状态用于下一次检测
			// context.Reset()
		}(i)

		// startTime := time.Now()
		// hasSpeech, err := context2.IsSpeech(audioData[i])
		// duration := time.Since(startTime)

		// if err != nil {
		// 	log.Printf("[%s] IsSpeech failed: %v", pcmfile, err)
		// 	continue
		// }

		// fmt.Printf("[%s] Has speech: %v (checked in %v)\n", pcmfile, hasSpeech, duration)

		// // 重置状态用于下一次检测
		// context2.Reset()
	}

	wg2.Wait()

	// 演示快速检测方法
	fmt.Println("\n--- Testing IsSpeechQuick method ---")
	context3 := sharedModel.NewContext()

	for i, pcmfile := range pcmfiles {
		if i >= len(audioData) {
			break
		}

		startTime := time.Now()
		hasSpeech, err := context3.IsSpeechQuick(audioData[i], 3) // 只检测前3个窗口
		duration := time.Since(startTime)

		if err != nil {
			log.Printf("[%s] IsSpeechQuick failed: %v", pcmfile, err)
			continue
		}

		fmt.Printf("[%s] Has speech (quick): %v (checked in %v)\n", pcmfile, hasSpeech, duration)

		// 重置状态用于下一次检测
		context3.Reset()
	}

	// 流式处理示例
	streamProcessingExample(sharedModel)
}

// generateTestAudio 生成测试音频数据（简单的正弦波）
func generateTestAudio(sampleRate int) []float32 {
	duration := 1.0 // 1秒
	samples := int(float64(sampleRate) * duration)
	audio := make([]float32, samples)

	for i := 0; i < samples; i++ {
		// 生成440Hz的正弦波
		t := float64(i) / float64(sampleRate)
		audio[i] = float32(0.5 * math.Sin(2*math.Pi*440*t))
	}

	return audio
}

// read pcm file and convert to float32 slice
func readPCMFile(filePath string) ([]float32, error) {
	data, err := os.ReadFile(filePath)
	if err != nil {
		return nil, err
	}

	// 将PCM数据转换为float32切片
	var audioData []float32
	for i := 0; i < len(data); i += 2 {
		// 读取16位PCM数据并转换为float32
		sample := int16(data[i]) | int16(data[i+1])<<8
		audioData = append(audioData, float32(sample)/32768.0)
	}

	return audioData, nil
}

// 流式处理示例
func streamProcessingExample(sharedModel *speech.SharedModel) {
	fmt.Println("\n--- Stream Processing Example ---")

	// 模拟实时音频流
	context := sharedModel.NewContext()
	chunkSize := 1600 // 100ms chunks at 16kHz

	for i := 0; i < 10; i++ { // 处理10个chunk
		chunk := generateTestAudio(chunkSize)

		segments, err := context.Detect(chunk)
		if err != nil {
			log.Printf("Stream processing error: %v", err)
			continue
		}

		if len(segments) > 0 {
			fmt.Printf("Chunk %d: Detected %d segments\n", i, len(segments))
		}

		// 模拟实时处理的延迟
		time.Sleep(50 * time.Millisecond)
	}
}
