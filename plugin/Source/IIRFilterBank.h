#pragma once

#include <JuceHeader.h>
#include <array>

/**
 * IIRFilterBank - 64-band parallel bandpass filter bank for zero-latency processing.
 *
 * Uses "Split & Sum" reconstructive topology:
 *   Output = Sum(Bandpass[b] * mask[b]) * normalization
 *
 * - NO dry path - signal is fully decomposed into bands
 * - mask = 0.0 means true silence (band * 0 = 0)
 * - mask = 1.0 means band passes at unity
 * - Normalization compensates for bandpass overlap so sum = 1.0 at unity gains
 *
 * This achieves deep gain reduction (-60 dB+) unlike subtractive topologies.
 */
class IIRFilterBank
{
public:
    // Hybrid 32+160 topology: 32 low bands (20-500Hz) + 160 high bands (500Hz-20kHz)
    static constexpr int NUM_BANDS = 192;
    static constexpr int NUM_LOW_BANDS = 32;
    static constexpr int NUM_HIGH_BANDS = 160;
    static constexpr float CROSSOVER_FREQ = 500.0f;
    static constexpr float MIN_FREQ = 20.0f;
    static constexpr float MAX_FREQ = 20000.0f;

    IIRFilterBank();
    ~IIRFilterBank() = default;

    void prepare(double sampleRate, int maxBlockSize);
    void reset();
    void process(juce::AudioBuffer<float>& buffer);

    /**
     * Set gain for a band (0.0 = full cut, 1.0 = unity)
     */
    void setBandGain(int bandIndex, float gain);

    /**
     * Set all band gains from neural network's 64-band mask.
     */
    void setAllBandGains(const std::array<float, NUM_BANDS>& masks);

    float getCenterFrequency(int bandIndex) const;
    const std::array<float, NUM_BANDS>& getCenterFrequencies() const { return centerFreqs; }
    const std::array<float, NUM_BANDS>& getBandGains() const { return bandGains; }

private:
    void calculateFrequencies();
    void calculateNormalization();
    float getBandpassMagnitudeAt(int bandIndex, float freq) const;

    double sampleRate = 48000.0;

    // Center frequencies (log-spaced from 20Hz to 20kHz)
    std::array<float, NUM_BANDS> centerFreqs;

    // Q factors (calculated based on neighbor spacing)
    std::array<float, NUM_BANDS> qFactors;

    // Global normalization factor - compensates for bandpass overlap
    // Calculated as average across multiple test frequencies for hybrid topology
    float normalizationFactor = 1.0f;

    // Current band gains (0.0 to 1.0) - what the filters are currently at
    std::array<float, NUM_BANDS> bandGains;

    // Target band gains (for smoothing)
    std::array<float, NUM_BANDS> targetGains;

    // Bandpass filters (2nd order biquad per band, stereo)
    struct BandFilter
    {
        juce::dsp::IIR::Filter<float> filter[2];  // Stereo
        juce::dsp::IIR::Coefficients<float>::Ptr coeffs;
    };
    std::array<BandFilter, NUM_BANDS> filters;

    // Temp buffer for bandpass output
    std::vector<float> bandpassOutput;

    // Smoothing coefficient
    float smoothingCoeff = 0.1f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IIRFilterBank)
};
