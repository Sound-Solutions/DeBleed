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
 * - Threshold (dB): the OPEN threshold; the gate closes 3 dB below it (fixed hysteresis)
 * - Ratio: Expansion ratio (2:1 to infinity:1)
 * - Open (ms): how fast the gain comes up when the vocal is back
 * - Close (ms): how fast the gain goes down once the level has dropped below the close threshold
 * - Range (dB): Maximum gain reduction
 *
 * The level detector is a fixed fast peak follower; the knobs time the GAIN, not the
 * detector (the DS201 / LSP Gate shape). Zero latency - purely causal.
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
     * Apply a precomputed gain curve to a block of audio.
     */
    float computeGain(float sidechainSample, float vadConfidence = 0.0f);
    void applyGains(float* audio, const float* gains, int numSamples);

    // User parameter setters
    void setThresholdDb(float thresholdDb);
    void setRatio(float ratio);
    void setOpenMs(float openMs);
    void setCloseMs(float closeMs);
    void setRangeDb(float rangeDb);
    void setVadGating(bool enabled);  // If true, VAD disables expansion during vocal

    // Getters for UI
    float getThresholdDb() const { return thresholdDb_.load(); }
    float getRatio() const { return ratio_.load(); }
    float getOpenMs() const { return openMs_.load(); }
    float getCloseMs() const { return closeMs_.load(); }
    float getRangeDb() const { return rangeDb_.load(); }
    float getGainReduction() const { return gainReductionDb_.load(); }  // For meters

private:
    void updateCoefficients();

    double sampleRate_ = 48000.0;

    // Peak detector state and its fixed coefficients (rise 0.1 ms, fall 20 ms)
    float envelope_ = 0.0f;
    float detectorRiseCoeff_ = 0.0f;
    float detectorFallCoeff_ = 0.0f;

    // Hysteresis: open at threshold, close 3 dB under it
    bool open_ = false;
    static constexpr float hysteresisDb = 3.0f;

    // Current gain (linear, 1 = open) and the smoother coefficients the knobs set
    float gainReduction_ = 1.0f;
    float openCoeff_ = 0.0f;
    float closeCoeff_ = 0.0f;

    // Parameters (atomic for thread-safe access)
    std::atomic<float> thresholdDb_{-18.0f};
    std::atomic<float> ratio_{4.0f};         // 4:1 expansion
    std::atomic<float> openMs_{50.0f};
    std::atomic<float> closeMs_{300.0f};
    std::atomic<float> rangeDb_{-20.0f};
    std::atomic<bool> vadGating_{true};      // VAD modulates threshold (raises threshold when vocal present)

    // For UI meters
    std::atomic<float> gainReductionDb_{0.0f};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SimpleExpander)
};
