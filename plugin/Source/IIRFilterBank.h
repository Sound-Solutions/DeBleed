#pragma once

#include <JuceHeader.h>
#include <array>

/**
 * IIRFilterBank - 64-band parallel IIR filter bank for zero-latency spectral processing.
 *
 * Uses "Dry + Bandpass" topology (standard parallel graphic EQ):
 *   Output = Dry_Signal + Sum(Bandpass[i] * GainCoeff[i])
 *
 * - When GainCoeff = 0: No change (dry signal passthrough)
 * - When GainCoeff < 0: Cut that frequency band
 * - When GainCoeff > 0: Boost that frequency band
 *
 * For neural gate (cut-only mode):
 *   GainCoeff = (mask - 1.0)  // mask=1 → 0, mask=0 → -1
 *
 * This ensures unity gain when all bands are at 1.0 (no processing).
 */
class IIRFilterBank
{
public:
    static constexpr int NUM_BANDS = 64;
    static constexpr float MIN_FREQ = 20.0f;
    static constexpr float MAX_FREQ = 20000.0f;
    static constexpr float DEFAULT_Q = 2.0f;
    static constexpr float MIN_GAIN_DB = -60.0f;

    IIRFilterBank();
    ~IIRFilterBank() = default;

    /**
     * Initialize the filter bank for the given sample rate.
     * Must be called before processing.
     */
    void prepare(double sampleRate, int maxBlockSize);

    /**
     * Reset all filter states (call on transport stop, etc.)
     */
    void reset();

    /**
     * Process a block of audio through the parallel filter bank.
     * @param buffer Audio buffer to process in-place
     */
    void process(juce::AudioBuffer<float>& buffer);

    /**
     * Set the gain for a specific band.
     * @param bandIndex Band index (0-63)
     * @param maskValue Mask value (0.0 to 1.0, where 1.0 = pass, 0.0 = cut)
     *                  Internally converted to gainCoeff = (mask - 1.0)
     */
    void setBandGain(int bandIndex, float maskValue);

    /**
     * Set gains for all bands at once.
     * @param gains Array of 64 linear gain values
     */
    void setAllBandGains(const std::array<float, NUM_BANDS>& gains);

    /**
     * Get the center frequency for a band.
     */
    float getCenterFrequency(int bandIndex) const;

    /**
     * Get all center frequencies.
     */
    const std::array<float, NUM_BANDS>& getCenterFrequencies() const { return centerFrequencies; }

    /**
     * Set Q factor for all bands.
     */
    void setQ(float newQ);

private:
    // Calculate logarithmically-spaced center frequencies
    void calculateCenterFrequencies();

    // Update filter coefficients for a single band
    void updateFilterCoefficients(int bandIndex);

    // Sample rate
    double sampleRate = 48000.0;

    // Center frequencies for each band (log-spaced)
    std::array<float, NUM_BANDS> centerFrequencies;

    // Gain coefficients for each band (derived from mask)
    // gainCoeff = (mask - 1.0): mask=1→0, mask=0→-1
    std::array<float, NUM_BANDS> gainCoeffs;

    // Q factor for all bands
    float qFactor = DEFAULT_Q;

    // Bandpass filters (one per band) - extract only the frequency band
    std::array<juce::dsp::IIR::Filter<float>, NUM_BANDS> bandpassFilters;

    // Bandpass filter coefficients
    std::array<juce::dsp::IIR::Coefficients<float>::Ptr, NUM_BANDS> bandpassCoeffs;

    // Temporary buffer for bandpass output accumulation
    juce::AudioBuffer<float> bandpassAccum;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IIRFilterBank)
};
