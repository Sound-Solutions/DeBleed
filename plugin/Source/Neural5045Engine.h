#pragma once

#include <JuceHeader.h>
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>
#include <atomic>

/**
 * Neural5045Engine - ONNX Runtime inference wrapper for differentiable EQ parameter prediction.
 *
 * This engine runs a Dilated TCN that predicts filter coefficients for real-time EQ processing.
 * Unlike the mask-based NeuralGateEngine, this outputs continuous EQ parameters:
 *
 * Input: Raw mono audio (sidechain signal)
 * Output: 50 normalized parameters [0, 1] per frame:
 *   - Params 0-47: 16 filters × 3 (freq, gain, Q)
 *   - Param 48: Input gain
 *   - Param 49: Output gain
 *
 * Frame rate: Every 64 samples (~0.67ms @ 96kHz, ~1.33ms @ 48kHz)
 *
 * The engine is designed for real-time use:
 * - All tensor memory is pre-allocated in prepare()
 * - No allocations in process()
 * - Lock-free model swapping
 * - Supports both 48kHz and 96kHz sample rates (model trained at 96kHz)
 */
class Neural5045Engine
{
public:
    // Parameter counts (must match DifferentiableBiquadChain)
    static constexpr int N_FILTERS = 16;
    static constexpr int N_PARAMS_PER_FILTER = 3;  // freq, gain, Q
    static constexpr int N_EXTRA_PARAMS = 2;       // input gain, output gain
    static constexpr int N_PARAMS = N_FILTERS * N_PARAMS_PER_FILTER + N_EXTRA_PARAMS;  // 50

    // Frame size for parameter updates (~21ms at 96kHz for musical dynamics)
    static constexpr int FRAME_SIZE = 2048;  // Samples per parameter frame

    Neural5045Engine();
    ~Neural5045Engine();

    /**
     * Prepare the engine for processing.
     * @param sampleRate Sample rate in Hz (48000 or 96000)
     * @param maxBlockSize Maximum number of audio samples per process call
     */
    void prepare(double sampleRate, int maxBlockSize);

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
    bool isModelLoaded() const { return modelLoaded_.load(); }

    /**
     * Process audio and predict filter parameters.
     *
     * @param audio Mono audio input (sidechain signal)
     * @param numSamples Number of samples in the audio buffer
     * @return Pointer to parameter array [N_PARAMS × numFrames], or nullptr if no model
     *
     * The returned pointer is valid until the next call to process().
     * Parameters are organized as: [param0_frame0, param1_frame0, ..., param49_frame0,
     *                               param0_frame1, param1_frame1, ..., param49_frame1, ...]
     */
    const float* process(const float* audio, int numSamples);

    /**
     * Get parameters for a specific frame from the last process call.
     * @param frameIndex Frame index (0 to getNumFrames()-1)
     * @return Pointer to N_PARAMS values for this frame, or default params if invalid
     */
    const float* getFrameParams(int frameIndex) const;

    /**
     * Get the number of parameter frames from the last process call.
     */
    int getNumFrames() const { return numFramesOutput_; }

    /**
     * Get default (bypass) parameters.
     * These produce unity gain with minimal EQ effect.
     */
    const float* getDefaultParams() const { return defaultParams_.data(); }

    /**
     * Unload the current model.
     */
    void unloadModel();

    /**
     * Get the path of the currently loaded model.
     */
    juce::String getModelPath() const { return currentModelPath_; }

    /**
     * Get any error message from the last operation.
     */
    juce::String getLastError() const { return lastError_; }

    /**
     * Get the sample rate the model was trained at.
     */
    double getTrainingSampleRate() const { return trainingSampleRate_; }

private:
    // Initialize default bypass parameters
    void initializeDefaultParams();

    // ONNX Runtime environment and session
    std::unique_ptr<Ort::Env> env_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<Ort::SessionOptions> sessionOptions_;
    std::unique_ptr<Ort::MemoryInfo> memoryInfo_;

    // Input/output names
    std::vector<const char*> inputNames_;
    std::vector<const char*> outputNames_;
    std::string inputNameStr_;
    std::string outputNameStr_;

    // Pre-allocated buffers
    std::vector<float> inputBuffer_;     // Audio input [1, 1, maxSamples]
    std::vector<float> outputBuffer_;    // Parameters [N_PARAMS × maxFrames]
    std::vector<float> defaultParams_;   // Bypass parameters [N_PARAMS]
    int maxSamplesAllocated_ = 0;
    int maxFramesAllocated_ = 0;

    // Output state from last process call
    int numFramesOutput_ = 0;

    // Thread-safe state
    std::atomic<bool> modelLoaded_{false};
    mutable std::mutex modelMutex_;

    // Model info
    juce::String currentModelPath_;
    juce::String lastError_;

    // Input/output shape info
    std::vector<int64_t> inputShape_;
    std::vector<int64_t> outputShape_;

    // Sample rates
    double currentSampleRate_ = 48000.0;
    static constexpr double trainingSampleRate_ = 96000.0;  // Model trained at 96kHz

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Neural5045Engine)
};
