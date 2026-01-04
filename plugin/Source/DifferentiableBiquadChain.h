#pragma once

#include <JuceHeader.h>
#include <array>
#include <atomic>

/**
 * DifferentiableBiquadChain - Real-time SVF TPT filter cascade for Neural 5045
 *
 * Uses State Variable Filter in Topology-Preserving Transform form for:
 * - Modulation stability (no clicks when coefficients change)
 * - Numerical stability at high frequencies
 * - Efficient coefficient updates
 *
 * Filter Chain (17 nodes, 50 parameters):
 *   Input → HPF → LowShelf → [12x Peaking] → HighShelf → LPF → Output
 *
 * Parameter Layout (50 total):
 *   [0-47]: 16 filters × 3 params (freq, gain, Q)
 *   [48]: Input gain (dB, -60 to 0)
 *   [49]: Output gain (dB, -60 to 0)
 *
 * All parameters are normalized [0, 1] and decoded internally.
 */
class DifferentiableBiquadChain
{
public:
    // Configuration
    static constexpr int N_FILTERS = 16;
    static constexpr int N_PARAMS_PER_FILTER = 3;  // freq, gain, Q
    static constexpr int N_EXTRA_PARAMS = 2;       // input gain, output gain
    static constexpr int N_TOTAL_PARAMS = N_FILTERS * N_PARAMS_PER_FILTER + N_EXTRA_PARAMS;  // 50

    // Filter roles
    static constexpr int FILTER_HPF = 0;
    static constexpr int FILTER_LOW_SHELF = 1;
    static constexpr int FILTER_PEAK_START = 2;
    static constexpr int FILTER_PEAK_END = 13;   // 12 peaking EQs
    static constexpr int FILTER_HIGH_SHELF = 14;
    static constexpr int FILTER_LPF = 15;

    // Coefficient update rate (~21ms at 96kHz for musical dynamics)
    static constexpr int FRAME_SIZE = 2048;  // Update coeffs every 2048 samples

    // Smoothing time for coefficient interpolation
    static constexpr float SMOOTHING_TIME_MS = 5.0f;

    DifferentiableBiquadChain();
    ~DifferentiableBiquadChain() = default;

    /**
     * Prepare for playback.
     * @param sampleRate Sample rate in Hz
     * @param maxBlockSize Maximum expected block size
     */
    void prepare(double sampleRate, int maxBlockSize);

    /**
     * Reset all filter states (call when starting new audio).
     */
    void reset();

    /**
     * Set all 50 parameters at once (thread-safe).
     * @param params Array of 50 normalized values [0, 1]
     */
    void setParameters(const float* params);

    /**
     * Process audio through the filter chain.
     * @param buffer Audio buffer to process in-place
     */
    void process(juce::AudioBuffer<float>& buffer);

    /**
     * Get frequency response magnitude at a given frequency.
     * Useful for visualization.
     * @param freqHz Frequency in Hz
     * @return Magnitude in linear scale
     */
    float getFrequencyResponse(float freqHz) const;

    /**
     * Get current filter parameters for visualization.
     * @param filterIndex Filter index (0-15)
     * @param freqHz Output: center frequency in Hz
     * @param gainDb Output: gain in dB
     * @param q Output: Q factor
     */
    void getFilterParams(int filterIndex, float& freqHz, float& gainDb, float& q) const;

    // Phase 3: User control overrides

    /**
     * Override HPF cutoff frequency (takes priority over neural prediction).
     * @param freqHz Frequency in Hz (20-500), or <= 0 to use neural prediction
     */
    void setHPFOverride(float freqHz);

    /**
     * Override LPF cutoff frequency (takes priority over neural prediction).
     * @param freqHz Frequency in Hz (5000-20000), or >= 20000 to use neural prediction
     */
    void setLPFOverride(float freqHz);

    /**
     * Add additional output gain offset (summed with neural prediction).
     * @param gainDb Gain offset in dB
     */
    void setOutputGainOffset(float gainDb);

    /**
     * Set coefficient smoothing time.
     * @param timeMs Smoothing time in milliseconds
     */
    void setSmoothingTime(float timeMs);

    /**
     * Set sensitivity (scales how aggressively neural predictions are applied).
     * @param sens 0.0 = bypass (flat EQ), 1.0 = full neural predictions
     */
    void setSensitivity(float sens);

private:
    // SVF TPT filter state (per filter, per channel)
    struct SVFState
    {
        float ic1eq = 0.0f;  // State 1
        float ic2eq = 0.0f;  // State 2
    };

    // SVF coefficients (computed from freq, gain, Q)
    struct SVFCoeffs
    {
        float g = 0.0f;      // tan(pi * fc / fs)
        float k = 1.0f;      // 1/Q (damping)
        float A = 1.0f;      // sqrt(linear_gain) for shelves/peaks

        // Precomputed for efficiency
        float a1 = 0.0f;
        float a2 = 0.0f;
        float a3 = 0.0f;
        float m0 = 1.0f;     // Output mixing coefficients
        float m1 = 0.0f;
        float m2 = 0.0f;
    };

    // Filter type enumeration
    enum class FilterType
    {
        HighPass,
        LowPass,
        Peak,
        LowShelf,
        HighShelf,
        Bypass
    };

    // Compute SVF coefficients for a given filter type
    void computeCoeffs(int filterIndex, float freqHz, float gainDb, float q);

    // Process one sample through one SVF filter
    float processSVF(float input, int filterIndex, int channel);

    // Denormalize parameters
    float denormalizeFreq(float norm, float minHz, float maxHz) const;
    float denormalizeGain(float norm, float minDb, float maxDb) const;
    float denormalizeQ(float norm) const;

    // Get filter type for a given index
    FilterType getFilterType(int filterIndex) const;

    // Sample rate
    double sampleRate_ = 48000.0;
    int maxBlockSize_ = 512;

    // Filter states (per filter, per channel - max 2 channels)
    std::array<std::array<SVFState, 2>, N_FILTERS> filterStates_;

    // Current coefficients (target values)
    std::array<SVFCoeffs, N_FILTERS> targetCoeffs_;

    // Smoothed coefficients (for click-free modulation)
    std::array<SVFCoeffs, N_FILTERS> currentCoeffs_;

    // Smoothing ramp counters
    std::array<juce::SmoothedValue<float>, N_FILTERS * 3> smoothedParams_;

    // Input/output gains
    juce::SmoothedValue<float> inputGainLinear_;
    juce::SmoothedValue<float> outputGainLinear_;

    // Current normalized parameters (for getFilterParams)
    std::array<std::atomic<float>, N_TOTAL_PARAMS> currentParams_;

    // Frame counter for coefficient updates
    int frameCounter_ = 0;

    // Phase 3: User override values
    std::atomic<float> hpfOverride_{-1.0f};      // -1 = use neural, else override freq
    std::atomic<float> lpfOverride_{20001.0f};   // >20000 = use neural, else override freq
    std::atomic<float> outputGainOffset_{0.0f};  // Additional dB to add to output gain
    std::atomic<float> sensitivity_{1.0f};       // 0-1, scales neural predictions
    float smoothingTimeMs_ = 5.0f;               // Coefficient smoothing time

    // Frequency ranges for each filter type (Hz)
    static constexpr float HPF_MIN_FREQ = 20.0f;
    static constexpr float HPF_MAX_FREQ = 500.0f;
    static constexpr float LPF_MIN_FREQ = 5000.0f;
    static constexpr float LPF_MAX_FREQ = 20000.0f;
    static constexpr float SHELF_MIN_FREQ = 50.0f;
    static constexpr float SHELF_MAX_FREQ = 16000.0f;
    static constexpr float PEAK_MIN_FREQ = 100.0f;
    static constexpr float PEAK_MAX_FREQ = 15000.0f;

    // Gain ranges (dB)
    static constexpr float FILTER_MIN_GAIN = -24.0f;
    static constexpr float FILTER_MAX_GAIN = 24.0f;
    static constexpr float BROADBAND_MIN_GAIN = -60.0f;
    static constexpr float BROADBAND_MAX_GAIN = 0.0f;

    // Q ranges
    static constexpr float Q_MIN = 0.5f;
    static constexpr float Q_MAX = 16.0f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DifferentiableBiquadChain)
};
