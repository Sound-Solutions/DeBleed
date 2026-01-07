#pragma once

#include <JuceHeader.h>
#include <atomic>

/**
 * SimpleVAD - Zero-latency Voice Activity Detection
 *
 * Uses RMS energy tracking to detect when vocal is present.
 * Outputs confidence 0-1 where 1 = vocal present (don't cut).
 *
 * This is intentionally simple and causal (no lookahead) for
 * zero-latency live sound applications.
 */
class SimpleVAD
{
public:
    SimpleVAD();
    ~SimpleVAD() = default;

    /**
     * Prepare for playback.
     * @param sampleRate Sample rate in Hz
     */
    void prepare(double sampleRate);

    /**
     * Reset state (call when starting new audio).
     */
    void reset();

    /**
     * Process a single sample and return VAD confidence.
     * @param sample Input audio sample
     * @return Confidence 0-1 (1 = vocal present, 0 = silence)
     */
    float processSample(float sample);

    /**
     * Process a block of audio and return per-sample confidence.
     * @param input Input audio buffer
     * @param confidence Output confidence buffer (must be same size)
     * @param numSamples Number of samples to process
     */
    void processBlock(const float* input, float* confidence, int numSamples);

    /**
     * Get current VAD confidence (thread-safe).
     */
    float getConfidence() const { return currentConfidence_.load(); }

    // Parameter setters (can be called from UI thread)
    void setThresholdDb(float thresholdDb);
    void setKneeDb(float kneeDb);
    void setAttackMs(float attackMs);
    void setReleaseMs(float releaseMs);

    // Parameter getters
    float getThresholdDb() const { return thresholdDb_.load(); }
    float getKneeDb() const { return kneeDb_.load(); }

private:
    // Update coefficients when parameters change
    void updateCoefficients();

    // Sample rate
    double sampleRate_ = 48000.0;

    // RMS envelope follower state
    float rmsEnvelope_ = 0.0f;

    // Smoothed confidence output
    float smoothedConfidence_ = 0.0f;

    // Attack/release coefficients for RMS envelope
    float rmsAttackCoeff_ = 0.0f;
    float rmsReleaseCoeff_ = 0.0f;

    // Confidence smoothing coefficients
    float confAttackCoeff_ = 0.0f;
    float confReleaseCoeff_ = 0.0f;

    // Parameters (atomic for thread-safe access)
    std::atomic<float> thresholdDb_{-40.0f};  // dB threshold for "vocal present"
    std::atomic<float> kneeDb_{10.0f};         // Soft knee width in dB
    std::atomic<float> attackMs_{5.0f};        // RMS attack time
    std::atomic<float> releaseMs_{50.0f};      // RMS release time

    // Current confidence (for UI display)
    std::atomic<float> currentConfidence_{0.0f};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SimpleVAD)
};
