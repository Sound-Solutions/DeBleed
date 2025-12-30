#pragma once

#include <JuceHeader.h>
#include <array>

/**
 * LinkwitzRileyGate - 6-band multiband gate with Linkwitz-Riley crossovers.
 *
 * Fully user-controlled traditional signal-level gate with per-band parameters.
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
 * - Signal-level detection: Gate triggers based on RMS level per band
 * - Per-band parameters: Threshold, attack, release, hold, range, enable
 * - Adjustable crossover frequencies
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

    // Per-band parameters
    struct BandParams
    {
        float thresholdDb = -40.0f;  // -80 to 0 dB
        float attackMs = 5.0f;       // 0.1 to 100 ms
        float releaseMs = 100.0f;    // 10 to 1000 ms
        float holdMs = 50.0f;        // 0 to 500 ms
        float rangeDb = -60.0f;      // -80 to 0 dB (max attenuation)
        bool enabled = true;
    };

    LinkwitzRileyGate();
    ~LinkwitzRileyGate() = default;

    void prepare(double sampleRate, int maxBlockSize);
    void reset();

    /**
     * Process audio through the multiband gate.
     * Gate triggers based on signal level in each band.
     * @param buffer Audio to process in-place
     */
    void process(juce::AudioBuffer<float>& buffer);

    // Master enable/disable
    void setEnabled(bool enabled) { this->masterEnabled = enabled; }
    bool isEnabled() const { return masterEnabled; }

    // Per-band parameter setters
    void setBandThreshold(int band, float db);
    void setBandAttack(int band, float ms);
    void setBandRelease(int band, float ms);
    void setBandHold(int band, float ms);
    void setBandRange(int band, float db);
    void setBandEnabled(int band, bool enabled);

    // Set all parameters for a band at once
    void setBandParams(int band, const BandParams& params);

    // Crossover frequency setters
    void setCrossover(int index, float hz);
    std::array<float, NUM_CROSSOVERS> getCrossoverFrequencies() const { return crossoverFreqs; }

    // Get current band states for visualization
    struct BandState
    {
        float currentGain;      // 0.0-1.0 (1.0 = unity, lower = gating)
        bool gateOpen;          // true = passing signal (above threshold)
        float signalLevelDb;    // Current detected signal level in dB
        float thresholdDb;      // Current threshold setting
    };
    std::array<BandState, NUM_BANDS> getBandStates() const;

private:
    double sampleRate = 48000.0;
    bool masterEnabled = true;

    // Per-band parameters
    std::array<BandParams, NUM_BANDS> bandParams;

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

    // Signal level detector per band (RMS envelope)
    struct LevelDetector
    {
        float rmsEnvelopeDb = -100.0f;   // Smoothed RMS level in dB
        float detectorCoeff = 0.0f;      // Smoothing coefficient
    };
    std::array<LevelDetector, NUM_BANDS> levelDetectors;

    // Gate envelope per band
    struct GateEnvelope
    {
        float currentGain = 1.0f;        // Current applied gain (0 to 1)
        float targetGain = 1.0f;         // Target gain
        int holdCounter = 0;             // Samples remaining in hold phase
        bool isGating = false;           // True when gate is closed/closing
    };
    std::array<GateEnvelope, NUM_BANDS> gateEnvelopes;

    void splitBands(const juce::AudioBuffer<float>& input);
    void recombineBands(juce::AudioBuffer<float>& output);
    void updateCrossoverCoefficients();

    // Detect signal level in a band (returns dB)
    float detectBandLevel(int band, int numSamples);

    // Process gate envelope for a band using signal level detection
    void processGateEnvelope(int band, float signalLevelDb, int numSamples);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LinkwitzRileyGate)
};
