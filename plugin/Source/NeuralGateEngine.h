#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <atomic>

/**
 * NeuralGateEngine - ONNX Runtime inference wrapper for the DeBleed mask estimator.
 *
 * Supports dual-stream input:
 * - Single-stream (legacy): 129 input features
 * - Dual-stream (v2): 257 input features [Stream A (129) | Stream B bass (128)]
 *
 * The input feature size is determined from the loaded ONNX model.
 * Output mask is always 129 bins (corresponding to Stream A).
 *
 * This class handles:
 * - Loading/hot-swapping ONNX models
 * - Running inference on magnitude spectrograms or dual-stream features
 * - Thread-safe model updates
 * - Memory management for tensors
 *
 * The engine is designed for real-time use:
 * - All tensor memory is pre-allocated in prepare()
 * - No allocations in process()
 * - Lock-free model swapping
 */
class NeuralGateEngine
{
public:
    // Output mask is always 129 bins (Stream A)
    static constexpr int N_OUTPUT_BINS = 129;

    // Maximum input features (dual-stream mode)
    static constexpr int N_MAX_INPUT_FEATURES = 257;  // 129 + 128

    // Legacy alias
    static constexpr int N_FREQ_BINS = N_OUTPUT_BINS;

    NeuralGateEngine();
    ~NeuralGateEngine();

    /**
     * Prepare the engine for processing.
     * @param maxFrames Maximum number of STFT frames per process call
     */
    void prepare(int maxFrames);

    /**
     * Load an ONNX model from file.
     * Thread-safe: can be called from any thread.
     * @param modelPath Path to the .onnx file
     * @return true if model loaded successfully
     */
    bool loadModel(const juce::String& modelPath);

    /**
     * Check if a model is currently loaded.
     */
    bool isModelLoaded() const { return modelLoaded.load(); }

    /**
     * Run inference on input features.
     * @param inputFeatures Input data - either single-stream [129 x frames] or dual-stream [257 x frames]
     * @param numFrames Number of STFT frames
     * @return Pointer to mask output [N_OUTPUT_BINS x numFrames], or nullptr if no model
     */
    const float* process(const float* inputFeatures, int numFrames);

    /**
     * Check if the loaded model expects dual-stream input (257 features).
     */
    bool isDualStreamModel() const { return modelInputFeatures == N_MAX_INPUT_FEATURES; }

    /**
     * Get the number of input features the model expects.
     */
    int getModelInputFeatures() const { return modelInputFeatures; }

    /**
     * Get the output mask buffer (for when you need to access it after process).
     */
    const float* getMaskOutput() const { return outputBuffer.data(); }

    /**
     * Unload the current model.
     */
    void unloadModel();

    /**
     * Get the path of the currently loaded model.
     */
    juce::String getModelPath() const { return currentModelPath; }

    /**
     * Get any error message from the last operation.
     */
    juce::String getLastError() const { return lastError; }

private:
    // ONNX Runtime environment and session
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::SessionOptions> sessionOptions;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo;

    // Input/output names
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;
    std::string inputNameStr;
    std::string outputNameStr;

    // Pre-allocated buffers
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    int maxFramesAllocated = 0;

    // Thread-safe state
    std::atomic<bool> modelLoaded{false};
    mutable std::mutex modelMutex;

    // Model info
    juce::String currentModelPath;
    juce::String lastError;

    // Input/output shape info
    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;

    // Number of input features (129 for single-stream, 257 for dual-stream)
    int modelInputFeatures = N_OUTPUT_BINS;  // Default to single-stream

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralGateEngine)
};
