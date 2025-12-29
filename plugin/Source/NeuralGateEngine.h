#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <atomic>

/**
 * NeuralGateEngine - ONNX Runtime inference wrapper for the DeBleed mask estimator.
 *
 * This class handles:
 * - Loading/hot-swapping ONNX models
 * - Running inference on magnitude spectrograms
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
    static constexpr int N_FREQ_BINS = 129;  // Matches Python trainer

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
     * Run inference on magnitude spectrogram data.
     * @param magnitude Input magnitude data [N_FREQ_BINS x numFrames]
     * @param numFrames Number of STFT frames
     * @return Pointer to mask output [N_FREQ_BINS x numFrames], or nullptr if no model
     */
    const float* process(const float* magnitude, int numFrames);

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralGateEngine)
};
