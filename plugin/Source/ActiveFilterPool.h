#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>
#include <algorithm>

/**
 * ActiveFilterPool - Dynamic "Hunter" filter architecture.
 *
 * Instead of 192 fixed filters, we have 32 "hunters" that chase
 * the deepest cuts in the neural network mask. This gives:
 * - Surgical precision where needed
 * - Zero muffling (unused filters = unity)
 * - Click-free modulation (smoothed freq + gain)
 * - Lower CPU (only processing active cuts)
 */
class ActiveFilterPool
{
public:
    static constexpr int MAX_FILTERS = 32;

    // Dual-stream mask configuration (matches neural network output)
    static constexpr int STREAM_A_BINS = 129;   // 256-point FFT: 187.5 Hz/bin
    static constexpr int STREAM_B_BINS = 128;   // 2048-point FFT low bins: 23.4 Hz/bin
    static constexpr int TOTAL_MASK_SIZE = 257; // STREAM_A_BINS + STREAM_B_BINS
    static constexpr int MASK_SIZE = 129;       // Legacy (for visualization compatibility)

    ActiveFilterPool();
    ~ActiveFilterPool() = default;

    /**
     * Initialize the filter pool.
     * @param sampleRate Sample rate in Hz
     * @param samplesPerBlock Maximum block size
     */
    void prepare(double sampleRate, int samplesPerBlock);

    /**
     * Reset all filters to unity.
     */
    void reset();

    /**
     * Process audio through the hunter filters.
     * @param buffer Audio buffer to process in-place
     * @param neuralMask Mask from neural network (129 bins, 0.0-1.0)
     */
    void process(juce::AudioBuffer<float>& buffer, const float* neuralMask);

    /**
     * Set the strength parameter (0.0 = no effect, 1.0 = full effect).
     */
    void setStrength(float newStrength) { strength = juce::jlimit(0.0f, 1.0f, newStrength); }

    /**
     * Set the floor/range in dB (e.g., -60 = allow cuts down to -60dB).
     */
    void setFloorDb(float newFloorDb) { floorDb = juce::jlimit(-80.0f, 0.0f, newFloorDb); }

    /**
     * Set attack time in ms (compressor-style: how fast cuts ENGAGE).
     */
    void setAttackMs(float ms) { attackMs = juce::jlimit(0.1f, 500.0f, ms); }

    /**
     * Set release time in ms (compressor-style: how fast cuts RELEASE back to unity).
     */
    void setReleaseMs(float ms) { releaseMs = juce::jlimit(1.0f, 2000.0f, ms); }

    /**
     * Set tightness in ms (minimum time a hunter stays at its frequency).
     * Higher values = more stable frequencies, less hopping.
     */
    void setTightnessMs(float ms) { tightnessMs = juce::jlimit(0.0f, 500.0f, ms); }

    /**
     * Set HPF bound - minimum frequency for hunters.
     */
    void setHpfBound(float hz) { hpfBound = juce::jlimit(20.0f, 500.0f, hz); }

    /**
     * Set LPF bound - maximum frequency for hunters.
     */
    void setLpfBound(float hz) { lpfBound = juce::jlimit(1000.0f, 20000.0f, hz); }

    /**
     * State of a single hunter filter (for visualization).
     */
    struct FilterState
    {
        float freq = 1000.0f;
        float gain = 1.0f;
        float q = 4.0f;
        bool active = false;
    };

    /**
     * Get current state of all hunters (for visualization).
     */
    std::array<FilterState, MAX_FILTERS> getFilterStates() const;

    /**
     * Get the frequency for a given FFT bin.
     */
    float getBinFrequency(int bin) const;

private:
    /**
     * A single "hunter" filter that chases problem frequencies.
     */
    struct HunterFilter
    {
        juce::dsp::IIR::Filter<float> filter[2];  // Stereo
        juce::dsp::IIR::Coefficients<float>::Ptr coeffs;
        juce::SmoothedValue<float> smoothGain;    // 0.0 = full cut, 1.0 = unity
        juce::SmoothedValue<float> smoothFreq;    // Target frequency in Hz
        juce::SmoothedValue<float> smoothQ;       // Target Q (width-adaptive)
        float currentFreq = 1000.0f;              // Last applied frequency
        float currentGain = 1.0f;                 // Last applied gain
        float currentQ = 4.0f;                    // Last applied Q
        float targetGainFromMask = 1.0f;          // What the mask wants (before smoothing)
        bool active = false;                      // Is this hunter assigned?
        int samplesAtCurrentFreq = 0;             // Tightness counter - samples since last freq change
    };

    std::array<HunterFilter, MAX_FILTERS> hunters;
    std::array<float, MASK_SIZE> binToFreqA;      // Stream A: bin index -> frequency Hz
    std::array<float, STREAM_B_BINS> binToFreqB;  // Stream B: bin index -> frequency Hz (high resolution lows)
    double sampleRate = 48000.0;

    // User parameters
    float strength = 1.0f;      // 0.0 = no effect, 1.0 = full effect
    float floorDb = -60.0f;     // Minimum gain in dB (range knob)
    float attackMs = 5.0f;      // Compressor attack (how fast cuts engage)
    float releaseMs = 100.0f;   // Compressor release (how fast cuts release)
    float tightnessMs = 50.0f;  // Minimum time before hunter can change frequency
    float hpfBound = 20.0f;     // Minimum frequency for hunters
    float lpfBound = 20000.0f;  // Maximum frequency for hunters
    int currentBlockSize = 512; // For tightness tracking

    /**
     * Scan the mask for valleys and assign hunters to targets.
     */
    void updateTargets(const float* mask);

    // Parameters
    static constexpr float SMOOTHING_TIME = 0.020f;      // 20ms for freq/gain smoothing
    static constexpr float VALLEY_THRESHOLD = 0.95f;     // Only hunt if gain < this
    static constexpr float HYSTERESIS_GAIN = 0.1f;       // 10% gain change required to reassign
    static constexpr float HYSTERESIS_FREQ = 0.2f;       // ~3 semitones freq change required
    static constexpr float FREQ_UPDATE_THRESH = 10.0f;   // Hz change needed to update coeffs
    static constexpr float GAIN_UPDATE_THRESH = 0.02f;   // Gain change needed to update coeffs
    static constexpr float Q_UPDATE_THRESH = 0.5f;       // Q change needed to update coeffs
    static constexpr float MIN_Q = 2.0f;                 // Q for wide cuts (broad)
    static constexpr float MAX_Q = 16.0f;                // Q for narrow cuts (surgical)
    static constexpr float MIN_FREQ = 20.0f;             // Minimum filter frequency
    static constexpr float MAX_FREQ = 20000.0f;          // Maximum filter frequency

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ActiveFilterPool)
};
