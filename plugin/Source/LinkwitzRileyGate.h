#pragma once

#include <JuceHeader.h>
#include <array>

/**
 * LinkwitzRileyGate - 6-band multiband gate with Linkwitz-Riley crossovers.
 *
 * Provides "macro" broadband gating while hunters provide "micro" surgical cuts.
 * Uses LR-4 (24dB/octave) crossovers for phase-coherent band splitting.
 *
 * Bands:
 *   0: Sub       (20-80 Hz)
 *   1: Low       (80-250 Hz)
 *   2: Low-Mid   (250-800 Hz)
 *   3: Mid       (800-2.5k Hz)
 *   4: High-Mid  (2.5k-8k Hz)
 *   5: High      (8k-20k Hz)
 *
 * Key features:
 * - Neural-driven: Gate triggers based on average mask value per band
 * - Adaptive timing: Engine auto-determines attack/release/hold per band
 * - User controls scale the auto values via multipliers
 */
class LinkwitzRileyGate
{
public:
    static constexpr int NUM_BANDS = 6;
    static constexpr int NUM_CROSSOVERS = 5;

    // Default crossover frequencies
    static constexpr std::array<float, NUM_CROSSOVERS> DEFAULT_CROSSOVERS = {{
        80.0f, 250.0f, 800.0f, 2500.0f, 8000.0f
    }};

    // Auto timing per band (slower for lows, faster for highs)
    struct AutoTiming
    {
        float attackMs;
        float releaseMs;
        float holdMs;
    };

    static constexpr std::array<AutoTiming, NUM_BANDS> AUTO_BAND_TIMING = {{
        {35.0f, 350.0f, 75.0f},   // Sub: slow - preserve transients
        {20.0f, 225.0f, 55.0f},   // Low: moderate
        {12.0f, 150.0f, 35.0f},   // Low-Mid: medium
        {6.0f, 100.0f, 20.0f},    // Mid: faster
        {3.0f, 65.0f, 12.0f},     // High-Mid: fast
        {1.5f, 50.0f, 8.0f}       // High: very fast
    }};

    LinkwitzRileyGate();
    ~LinkwitzRileyGate() = default;

    void prepare(double sampleRate, int maxBlockSize);
    void reset();

    /**
     * Process audio through the multiband gate.
     * @param buffer Audio to process in-place
     * @param bandMaskAverages Array of 6 average mask values per band (0.0-1.0)
     */
    void process(juce::AudioBuffer<float>& buffer, const std::array<float, NUM_BANDS>& bandMaskAverages);

    // User global controls (ms values - applied to mid band, other bands scale proportionally)
    void setSensitivity(float percent) { sensitivity = juce::jlimit(-100.0f, 100.0f, percent); }
    void setAttackMs(float ms) { attackMs = juce::jlimit(0.5f, 100.0f, ms); }
    void setReleaseMs(float ms) { releaseMs = juce::jlimit(10.0f, 1000.0f, ms); }
    void setHoldMs(float ms) { holdMs = juce::jlimit(1.0f, 200.0f, ms); }
    void setFloorDb(float db) { floorDb = juce::jlimit(-80.0f, 0.0f, db); }

    // Enable/disable the gate
    void setEnabled(bool enabled) { this->enabled = enabled; }
    bool isEnabled() const { return enabled; }

    // Get current band states for visualization
    struct BandState
    {
        float currentGain;      // 0.0-1.0
        bool gateOpen;          // true = passing signal
        float maskAverage;      // Current mask average
    };
    std::array<BandState, NUM_BANDS> getBandStates() const;

private:
    double sampleRate = 48000.0;
    bool enabled = true;

    // User controls (ms values - these are the "mid band" reference, other bands scale proportionally)
    float sensitivity = 0.0f;    // -100% to +100% - offsets threshold
    float attackMs = 6.0f;       // Mid band attack (others scale: sub=5.8x, high=0.25x)
    float releaseMs = 100.0f;    // Mid band release (others scale similarly)
    float holdMs = 20.0f;        // Mid band hold (others scale similarly)
    float floorDb = -60.0f;      // Maximum attenuation when gated

    // Band scaling factors relative to mid band (index 3)
    // These are computed from AUTO_BAND_TIMING ratios
    static constexpr int MID_BAND_INDEX = 3;
    float getBandAttackMs(int band) const;
    float getBandReleaseMs(int band) const;
    float getBandHoldMs(int band) const;

    // Crossover frequencies
    std::array<float, NUM_CROSSOVERS> crossoverFreqs = DEFAULT_CROSSOVERS;

    // LR-4 crossover filters
    // Each crossover uses two cascaded 2nd-order Butterworth = LR-4
    struct LR4Crossover
    {
        // For stereo: [0] = left, [1] = right
        juce::dsp::StateVariableTPTFilter<float> lp1[2], lp2[2];  // Lowpass cascade
        juce::dsp::StateVariableTPTFilter<float> hp1[2], hp2[2];  // Highpass cascade
    };
    std::array<LR4Crossover, NUM_CROSSOVERS> crossovers;

    // Band processing buffers
    std::array<juce::AudioBuffer<float>, NUM_BANDS> bandBuffers;

    // Gate envelope per band
    struct GateEnvelope
    {
        float currentGain = 1.0f;
        float targetGain = 1.0f;
        int holdCounter = 0;
        bool isGating = false;
        float lastMaskAverage = 1.0f;
    };
    std::array<GateEnvelope, NUM_BANDS> gateEnvelopes;

    void splitBands(const juce::AudioBuffer<float>& input);
    void recombineBands(juce::AudioBuffer<float>& output);
    void updateCrossoverCoefficients();
    void processGateEnvelope(int band, float maskAverage, int numSamples);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LinkwitzRileyGate)
};
