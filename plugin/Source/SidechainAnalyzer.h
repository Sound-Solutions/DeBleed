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
    static constexpr int NUM_IIR_BANDS = 64;
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
     */
    const float* getRawMask() const { return lastMask; }

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
    const float* lastMagnitude = nullptr;

    // Parameters
    float strength = 1.0f;
    float thresholdDb = 0.0f;
    float floorDb = -60.0f;

    // Sample rate
    double sampleRate = 48000.0;
};
