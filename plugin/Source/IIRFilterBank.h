#pragma once

#include <JuceHeader.h>
#include <array>

/**
 * IIRFilterBank - Hybrid Series/Parallel filter bank using SVF TPT filters.
 *
 * SVF TPT (State Variable Topology Preserving Transform) filters are
 * inherently stable under parameter modulation, eliminating clicks/pops.
 *
 * HYBRID TOPOLOGY (500 Hz crossover):
 *
 * LOW ENGINE (20-500 Hz) - Parallel Bandpass "Split & Sum":
 *   - 32 parallel SVF bandpass filters
 *   - Gain applied AFTER filter (output scaling) for click-free modulation
 *   - Output = Sum(SVF_BP[b].output * smoothedGain[b]) * normalization
 *
 * HIGH ENGINE (500 Hz - 20 kHz) - Series Notch:
 *   - 160 cascaded SVF-based notch filters
 *   - Notch formula: output = input - bandpass * notchDepth
 *   - notchDepth = 0 → bypass, notchDepth = 1 → full notch
 *   - Click-free via SmoothedValue per-sample interpolation
 *
 * Key advantage: Fixed freq/Q set once in prepare(), only gain changes
 * during playback - no coefficient recalculation needed!
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
    static constexpr int BLEND_ZONE_BANDS = 4;  // Bands to blend across crossover

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
     * Set all band gains from neural network mask.
     */
    void setAllBandGains(const std::array<float, NUM_BANDS>& masks);

    float getCenterFrequency(int bandIndex) const;
    const std::array<float, NUM_BANDS>& getCenterFrequencies() const { return centerFreqs; }
    const std::array<float, NUM_BANDS>& getBandGains() const { return currentGains; }

private:
    void calculateFrequencies();
    void calculateLowNormalization();
    float getBandpassMagnitudeAt(int bandIndex, float freq) const;

    double sampleRate = 48000.0;

    // Center frequencies (log-spaced, separate ranges for low/high)
    std::array<float, NUM_BANDS> centerFreqs;

    // Q factors (calculated based on neighbor spacing)
    std::array<float, NUM_BANDS> qFactors;

    // Target band gains (from neural network, 0.0 to 1.0)
    std::array<float, NUM_BANDS> targetGains;

    // Current band gains (for getBandGains() - reflects smoothed state)
    std::array<float, NUM_BANDS> currentGains;

    // ═══════════════════════════════════════════════════════════════
    // LOW ENGINE: SVF Bandpass (parallel, bands 0-31, 20-500 Hz)
    // Gain applied AFTER filter for click-free modulation
    // ═══════════════════════════════════════════════════════════════
    struct SVFBandpass
    {
        juce::dsp::StateVariableTPTFilter<float> filter[2];  // Stereo
        juce::SmoothedValue<float> gainSmoother;              // Click-free gain
    };
    std::array<SVFBandpass, NUM_LOW_BANDS> lowBands;
    float lowNormalization = 1.0f;

    // ═══════════════════════════════════════════════════════════════
    // HIGH ENGINE: SVF Notch (series, bands 32-191, 500 Hz - 20 kHz)
    // Notch = Input - Bandpass × NotchDepth
    // ═══════════════════════════════════════════════════════════════
    struct SVFNotch
    {
        juce::dsp::StateVariableTPTFilter<float> filter[2];  // Stereo
        juce::SmoothedValue<float> notchDepthSmoother;        // 0=bypass, 1=full notch
        float resonance = 1.0f;  // Store for bandpass normalization
    };
    std::array<SVFNotch, NUM_HIGH_BANDS> highBands;

    // ═══════════════════════════════════════════════════════════════
    // CROSSOVER: SVF TPT for seamless blending
    // ═══════════════════════════════════════════════════════════════
    juce::dsp::StateVariableTPTFilter<float> lowpassFilter[2];
    juce::dsp::StateVariableTPTFilter<float> highpassFilter[2];

    // Smoothing time in seconds (10ms for responsive yet click-free modulation)
    static constexpr float SMOOTHING_TIME_SEC = 0.010f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(IIRFilterBank)
};
