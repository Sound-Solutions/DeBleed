#pragma once

#include <JuceHeader.h>
#include "STFTProcessor.h"
#include "NeuralGateEngine.h"
#include "BandMapper.h"
#include "EnvelopeFollower.h"
#include <array>

/**
 * SidechainAnalyzer - Combines STFT analysis, neural network inference,
 * and gain mapping for the IIR filter bank.
 *
 * This class handles the "control path" in the zero-latency architecture:
 * Input audio → Dual-Stream STFT → Neural mask → Band mapping → Envelope followers
 *
 * Dual-Stream Analysis:
 * - Stream A (256 FFT): Fast, ~187Hz resolution for transients
 * - Stream B (2048 FFT): Slow, ~23Hz resolution for bass precision
 * - Concatenated features [257 bins] fed to neural network
 *
 * The analysis does NOT add latency to the audio path - it runs in parallel
 * as a sidechain.
 */
class SidechainAnalyzer
{
public:
    // Hybrid 32+160 topology
    static constexpr int NUM_IIR_BANDS = 192;
    static constexpr int N_FREQ_BINS = STFTProcessor::N_FREQ_BINS_A;  // 129 (Stream A)
    static constexpr int N_TOTAL_FEATURES = STFTProcessor::N_TOTAL_FEATURES;  // 257 (dual-stream input)
    static constexpr int N_OUTPUT_BINS = N_TOTAL_FEATURES;  // 257 (dual-stream mask output)

    SidechainAnalyzer();
    ~SidechainAnalyzer() = default;

    /**
     * Initialize the analyzer.
     * @param sampleRate Sample rate in Hz
     * @param maxBlockSize Maximum audio block size
     * @param iirCenterFreqs Center frequencies of the IIR filter bank
     */
    void prepare(double sampleRate, int maxBlockSize,
                 const std::array<float, NUM_IIR_BANDS>& iirCenterFreqs);

    /**
     * Reset internal state.
     */
    void reset();

    /**
     * Analyze audio and compute band gains.
     * Call this once per audio block with a copy of the input signal.
     *
     * @param input Audio samples to analyze
     * @param numSamples Number of samples
     */
    void analyze(const float* input, int numSamples);

    /**
     * Get the current smoothed band gains for the IIR filter bank.
     * These are updated by analyze() and smoothed by envelope followers.
     */
    const std::array<float, NUM_IIR_BANDS>& getBandGains() const { return smoothedBandGains; }

    /**
     * Get raw mask values from neural network (for visualization).
     * Returns pointer to the LATEST frame's mask (257 bins).
     */
    const float* getRawMask() const { return latestFrameMask; }

    /**
     * Get raw magnitude spectrum (for visualization).
     */
    const float* getMagnitude() const { return lastMagnitude; }

    /**
     * Check if a model is loaded.
     */
    bool isModelLoaded() const { return neuralEngine.isModelLoaded(); }

    /**
     * Load a neural network model.
     */
    bool loadModel(const juce::String& modelPath);

    /**
     * Unload the current model.
     */
    void unloadModel();

    /**
     * Get current model path.
     */
    juce::String getModelPath() const { return neuralEngine.getModelPath(); }

    // Parameter setters
    void setStrength(float strength);
    void setAttack(float attackMs);
    void setRelease(float releaseMs);
    void setFrequencyRange(float lowHz, float highHz);
    void setThreshold(float thresholdDb);
    void setFloor(float floorDb);

    /**
     * Get average gain reduction for metering.
     */
    float getAverageGainReduction() const;

    // Debug test mode - bypasses NN and sets a test pattern
    void setDebugTestMode(bool enabled) { debugTestMode = enabled; }
    bool isDebugTestMode() const { return debugTestMode; }
    void setDebugTestBand(int band) { debugTestBand = band; }  // Which band to cut
    void setDebugTestWidth(int width) { debugTestWidth = width; }  // How many bands to cut (1 = single band)

private:
    // Components
    STFTProcessor stftProcessor;
    NeuralGateEngine neuralEngine;
    BandMapper bandMapper;
    EnvelopeFollowerBank envelopeBank;

    // Output gains
    std::array<float, NUM_IIR_BANDS> rawBandGains;
    std::array<float, NUM_IIR_BANDS> smoothedBandGains;

    // Cache for visualization
    const float* lastMask = nullptr;
    const float* latestFrameMask = nullptr;  // Points to the LAST frame in the buffer
    const float* lastMagnitude = nullptr;

    // Parameters
    float strength = 1.0f;
    float thresholdDb = 0.0f;
    float floorDb = -60.0f;

    // Debug test mode - SET TO TRUE TO TEST FILTER BANK ISOLATION
    // When true: bypasses NN, cuts ONLY the specified band(s), all others at unity
    // This lets you hear if the filter bank itself causes smoothing/bleeding
    bool debugTestMode = false;  // <-- DISABLED (normal operation)
    int debugTestBand = 150;     // Band to cut (~8kHz with 192-band hybrid)
    int debugTestWidth = 10;     // Width: try 10 bands to see a deeper notch

    // Sample rate
    double sampleRate = 48000.0;
};
