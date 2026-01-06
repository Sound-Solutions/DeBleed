#pragma once

#include <JuceHeader.h>
#include <atomic>

/**
 * SimpleExpander - User-controllable expander/gate
 *
 * Reduces gain when signal falls below threshold.
 * Can be gated by VAD to only expand during silence.
 *
 * User controls:
 * - Threshold (dB): Level below which expansion kicks in
 * - Ratio: Expansion ratio (2:1 to infinity:1)
 * - Attack (ms): How fast expansion engages
 * - Release (ms): How slow expansion releases
 * - Range (dB): Maximum gain reduction
 *
 * Zero latency - purely causal envelope following.
 */
class SimpleExpander
{
public:
    SimpleExpander();
    ~SimpleExpander() = default;

    /**
     * Prepare for playback.
     */
    void prepare(double sampleRate);

    /**
     * Reset state.
     */
    void reset();

    /**
     * Process a single sample.
     * @param sample Input audio sample
     * @param vadConfidence Optional VAD confidence (1 = vocal, disables expansion)
     * @return Processed sample
     */
    float processSample(float sample, float vadConfidence = 0.0f);

    /**
     * Process a block of audio.
     */
    void processBlock(float* audio, const float* vadConfidence, int numSamples);

    // User parameter setters
    void setThresholdDb(float thresholdDb);
    void setRatio(float ratio);
    void setAttackMs(float attackMs);
    void setReleaseMs(float releaseMs);
    void setRangeDb(float rangeDb);
    void setVadGating(bool enabled);  // If true, VAD disables expansion during vocal

    // Getters for UI
    float getThresholdDb() const { return thresholdDb_.load(); }
    float getRatio() const { return ratio_.load(); }
    float getAttackMs() const { return attackMs_.load(); }
    float getReleaseMs() const { return releaseMs_.load(); }
    float getRangeDb() const { return rangeDb_.load(); }
    float getGainReduction() const { return gainReductionDb_.load(); }  // For meters

private:
    void updateCoefficients();

    double sampleRate_ = 48000.0;

    // Envelope follower state
    float envelope_ = 0.0f;

    // Current gain reduction (linear)
    float gainReduction_ = 1.0f;

    // Attack/release coefficients
    float attackCoeff_ = 0.0f;
    float releaseCoeff_ = 0.0f;

    // Parameters (atomic for thread-safe access)
    std::atomic<float> thresholdDb_{-40.0f};
    std::atomic<float> ratio_{4.0f};         // 4:1 expansion
    std::atomic<float> attackMs_{1.0f};      // Fast attack
    std::atomic<float> releaseMs_{100.0f};   // Moderate release
    std::atomic<float> rangeDb_{-40.0f};     // Max 40dB reduction
    std::atomic<bool> vadGating_{true};      // VAD modulates threshold (raises threshold when vocal present)

    // For UI meters
    std::atomic<float> gainReductionDb_{0.0f};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SimpleExpander)
};
