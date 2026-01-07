#pragma once

#include <JuceHeader.h>
#include <array>
#include <atomic>

/**
 * DynamicEQ - 6-band parametric EQ with VAD-gated gains
 *
 * Each band has:
 * - Fixed frequency (learned during training)
 * - Fixed Q (learned during training)
 * - Fixed max cut depth (learned during training)
 * - Dynamic gain that follows VAD: 0dB when vocal, max_cut when silent
 *
 * Zero latency - uses IIR biquad filters.
 */
class DynamicEQ
{
public:
    static constexpr int NUM_BANDS = 6;

    // Band parameters (loaded from trained model)
    struct BandParams
    {
        float freqHz = 1000.0f;    // Center frequency
        float q = 1.0f;            // Q factor
        float maxCutDb = -6.0f;    // Maximum cut when silent (negative dB)
    };

    DynamicEQ();
    ~DynamicEQ() = default;

    /**
     * Prepare for playback.
     */
    void prepare(double sampleRate, int maxBlockSize);

    /**
     * Reset filter states.
     */
    void reset();

    /**
     * Process a single sample with given VAD confidence.
     * @param sample Input audio sample
     * @param vadConfidence VAD confidence 0-1 (1 = vocal present)
     * @return Processed sample
     */
    float processSample(float sample, float vadConfidence);

    /**
     * Process a block of audio with per-sample VAD confidence.
     * @param audio Audio buffer (modified in place)
     * @param vadConfidence Per-sample VAD confidence
     * @param numSamples Number of samples
     */
    void processBlock(float* audio, const float* vadConfidence, int numSamples);

    /**
     * Set parameters for a band.
     */
    void setBandParams(int bandIndex, const BandParams& params);

    /**
     * Get current band parameters.
     */
    BandParams getBandParams(int bandIndex) const;

    /**
     * Load all band parameters from JSON-like structure.
     */
    void loadParams(const std::array<BandParams, NUM_BANDS>& params);

    /**
     * Set smoothing time for gain transitions.
     */
    void setSmoothingMs(float smoothingMs);

    /**
     * Get current gain for a band (for UI visualization).
     */
    float getCurrentGainDb(int bandIndex) const;

    /**
     * Get frequency response at a given frequency.
     */
    float getFrequencyResponse(float freqHz) const;

private:
    // Biquad filter state per band per channel
    struct BiquadState
    {
        float x1 = 0.0f, x2 = 0.0f;  // Input delay
        float y1 = 0.0f, y2 = 0.0f;  // Output delay
    };

    // Biquad coefficients
    struct BiquadCoeffs
    {
        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f;
        float a1 = 0.0f, a2 = 0.0f;
    };

    // Compute biquad coefficients for peaking EQ
    BiquadCoeffs computePeakingCoeffs(float freqHz, float q, float gainDb) const;

    // Process single sample through one biquad
    float processBiquad(float input, BiquadState& state, const BiquadCoeffs& coeffs);

    // Sample rate
    double sampleRate_ = 48000.0;

    // Band parameters (fixed, loaded from training)
    std::array<BandParams, NUM_BANDS> bandParams_;

    // Current smoothed gain per band (dB)
    std::array<float, NUM_BANDS> currentGainDb_;

    // Target gain per band (computed from VAD)
    std::array<float, NUM_BANDS> targetGainDb_;

    // Filter states (per band, stereo)
    std::array<std::array<BiquadState, 2>, NUM_BANDS> filterStates_;

    // Smoothing coefficient for gain changes
    float gainSmoothingCoeff_ = 0.001f;

    // For UI display
    std::array<std::atomic<float>, NUM_BANDS> displayGainDb_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DynamicEQ)
};
