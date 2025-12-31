#pragma once

#include <JuceHeader.h>

/**
 * SimpleSidechainGate - Single-band gate with HPF/LPF sidechain filtering.
 *
 * Like a Shure PSE or dbx noise gate:
 * - Full-band gating (no crossovers)
 * - Sidechain HPF/LPF filters the DETECTION signal (not audio)
 * - Gate only opens when signal within HPF-LPF range exceeds threshold
 * - User-controlled parameters (not neural-driven)
 */
class LinkwitzRileyGate
{
public:
    LinkwitzRileyGate();
    ~LinkwitzRileyGate() = default;

    void prepare(double sampleRate, int maxBlockSize);
    void reset();

    /**
     * Process audio through the gate.
     * @param buffer Audio to process in-place
     */
    void process(juce::AudioBuffer<float>& buffer);

    // Master enable/disable
    void setEnabled(bool enabled) { masterEnabled = enabled; }
    bool isEnabled() const { return masterEnabled; }

    // Gate parameters
    void setThreshold(float db) { thresholdDb = juce::jlimit(-80.0f, 0.0f, db); }
    void setAttack(float ms) { attackMs = juce::jlimit(0.1f, 100.0f, ms); }
    void setRelease(float ms) { releaseMs = juce::jlimit(10.0f, 2000.0f, ms); }
    void setHold(float ms) { holdMs = juce::jlimit(0.0f, 500.0f, ms); }
    void setRange(float db) { rangeDb = juce::jlimit(-80.0f, 0.0f, db); }

    // Sidechain filter bounds (uses existing HPF/LPF parameters)
    void setSidechainHPF(float hz) { sidechainHPF = juce::jlimit(20.0f, 2000.0f, hz); updateFilters(); }
    void setSidechainLPF(float hz) { sidechainLPF = juce::jlimit(500.0f, 20000.0f, hz); updateFilters(); }

    // Set neural mask confidence (0.0 = bleed, 1.0 = singer)
    void setNeuralConfidence(float confidence) { neuralConfidence = juce::jlimit(0.0f, 1.0f, confidence); }

    // Get current state for visualization
    float getCurrentGain() const { return currentGain; }
    float getGainReductionDb() const { return juce::Decibels::gainToDecibels(currentGain); }
    float getDetectedLevelDb() const { return detectedLevelDb; }
    float getNeuralConfidence() const { return neuralConfidence; }
    float getThresholdDb() const { return thresholdDb; }
    bool isGateOpen() const { return !isGating; }

    // Legacy compatibility - these do nothing now but prevent compile errors
    static constexpr int NUM_BANDS = 6;
    static constexpr int NUM_CROSSOVERS = 5;
    struct BandParams { float thresholdDb, attackMs, releaseMs, holdMs, rangeDb; bool enabled; };
    struct BandState { float currentGain, signalLevelDb, thresholdDb; bool gateOpen; };
    void setBandThreshold(int, float) {}
    void setBandAttack(int, float) {}
    void setBandRelease(int, float) {}
    void setBandHold(int, float) {}
    void setBandRange(int, float) {}
    void setBandEnabled(int, bool) {}
    void setBandParams(int, const BandParams&) {}
    void setCrossover(int, float) {}
    std::array<float, NUM_CROSSOVERS> getCrossoverFrequencies() const { return {{80,250,800,2500,8000}}; }
    std::array<BandState, NUM_BANDS> getBandStates() const { return {}; }

private:
    double sampleRate = 48000.0;
    bool masterEnabled = true;

    // Gate parameters
    float thresholdDb = -40.0f;
    float attackMs = 5.0f;
    float releaseMs = 100.0f;
    float holdMs = 50.0f;
    float rangeDb = -60.0f;

    // Sidechain filter frequencies
    float sidechainHPF = 80.0f;
    float sidechainLPF = 12000.0f;

    // Sidechain filters (2-pole each for smooth response)
    juce::dsp::StateVariableTPTFilter<float> hpFilter1, hpFilter2;
    juce::dsp::StateVariableTPTFilter<float> lpFilter1, lpFilter2;

    // Level detection (for visualization)
    float detectedLevelDb = -100.0f;
    float detectorCoeff = 0.0f;

    // Neural confidence (0.0 = bleed, 1.0 = singer)
    float neuralConfidence = 1.0f;
    float smoothedConfidence = 1.0f;

    // Gate state
    float currentGain = 1.0f;
    float targetGain = 1.0f;
    int holdCounter = 0;
    bool isGating = false;

    // Sidechain buffer
    std::vector<float> sidechainBuffer;

    void updateFilters();
    float detectLevel(const float* input, int numSamples);
    void processEnvelope(float levelDb, int numSamples);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(LinkwitzRileyGate)
};
