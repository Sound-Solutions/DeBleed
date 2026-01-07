#pragma once

#include <JuceHeader.h>
#include "DifferentiableBiquadChain.h"
#include "TrainerProcess.h"
#include "SimpleVAD.h"
#include "SpectralVAD.h"
#include "DynamicEQ.h"
#include "SimpleExpander.h"

/**
 * DeBleedAudioProcessor - DeBleed V2: Formant-Based Source Enhancer
 *
 * Zero-latency bleed reduction using spectral voice activity detection
 * and dynamics processing. Uses hardcoded vocal formant weights.
 *
 * Architecture:
 *   Input → SpectralVAD → Expander → Output
 *
 * Processing Flow:
 *   1. Create mono sidechain from input
 *   2. Run SpectralVAD to detect vocal presence (formant-based weights)
 *   3. Apply VAD-controlled expansion to reduce bleed
 *   4. Apply dry/wet mix
 */
class DeBleedAudioProcessor : public juce::AudioProcessor,
                               public juce::AudioProcessorValueTreeState::Listener
{
public:
    DeBleedAudioProcessor();
    ~DeBleedAudioProcessor() override;

    // AudioProcessor overrides
    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override;

    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;

    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return JucePlugin_Name; }

    bool acceptsMidi() const override { return false; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.0; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // Parameter listener
    void parameterChanged(const juce::String& parameterID, float newValue) override;

    // Get biquad chain for visualization (legacy, may be removed)
    const DifferentiableBiquadChain& getBiquadChain() const { return biquadChain_; }

    // Training interface
    TrainerProcess& getTrainerProcess() { return trainerProcess; }

    // Parameter access
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }

    // Parameter IDs
    static const juce::String PARAM_MIX;
    static const juce::String PARAM_BYPASS;
    static const juce::String PARAM_LIVE_MODE;

    // Phase 3 Parameters
    static const juce::String PARAM_OUTPUT_GAIN;
    static const juce::String PARAM_HPF_FREQ;
    static const juce::String PARAM_LPF_FREQ;
    static const juce::String PARAM_SENSITIVITY;
    static const juce::String PARAM_SMOOTHING;

    // V2 Expander Parameters
    static const juce::String PARAM_EXP_THRESHOLD;
    static const juce::String PARAM_EXP_RATIO;
    static const juce::String PARAM_EXP_ATTACK;
    static const juce::String PARAM_EXP_RELEASE;
    static const juce::String PARAM_EXP_RANGE;
    static const juce::String PARAM_USE_V2;  // Toggle between v1 and v2 architectures

    // V2 component access for visualization
    const SpectralVAD& getSpectralVAD() const { return spectralVAD_; }
    const DynamicEQ& getDynamicEQ() const { return dynamicEQ_; }
    const SimpleExpander& getExpander() const { return expander_; }
    float getVADConfidence() const { return spectralVAD_.getConfidence(); }
    float getOutputLevelDb() const { return outputLevelDb_.load(); }

    // Load v2 learned parameters from JSON
    bool loadV2Params(const juce::String& jsonPath);

private:
    // Create parameter layout
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Parameters
    juce::AudioProcessorValueTreeState parameters;

    // Atomic parameter values for real-time access
    std::atomic<float> mix{1.0f};
    std::atomic<bool> bypassed{false};

    // Phase 3 parameter atomics
    std::atomic<float> outputGain{0.0f};    // dB (-24 to +12)
    std::atomic<float> hpfFreq{20.0f};      // Hz (20 to 500)
    std::atomic<float> lpfFreq{20000.0f};   // Hz (5000 to 20000)
    std::atomic<float> sensitivity{1.0f};   // 0 to 1
    std::atomic<float> smoothing{50.0f};    // ms (1 to 200)

    // V2 Expander parameter atomics
    std::atomic<float> expThreshold{-40.0f};  // dB
    std::atomic<float> expRatio{4.0f};        // ratio
    std::atomic<float> expAttack{1.0f};       // ms
    std::atomic<float> expRelease{100.0f};    // ms
    std::atomic<float> expRange{-40.0f};      // dB
    std::atomic<bool> useV2{true};            // Use v2 architecture

    // Biquad chain (legacy, may be removed)
    DifferentiableBiquadChain biquadChain_;  // SVF TPT filter cascade

    // V2: Zero-latency vocal-preservation architecture
    SpectralVAD spectralVAD_;                // Spectral VAD with learned frequency weights
    DynamicEQ dynamicEQ_;                    // 6-band VAD-gated dynamic EQ
    SimpleExpander expander_;                // User-controlled expander

    // Buffers for processing
    juce::AudioBuffer<float> dryBuffer_;     // Dry signal for wet/dry mix
    std::vector<float> monoSidechain_;       // Mono mix for neural network input
    std::vector<float> vadConfidence_;       // Per-sample VAD confidence for v2

    // Training process
    TrainerProcess trainerProcess;

    // Processing state
    double currentSampleRate_ = 48000.0;
    int currentBlockSize_ = 512;

    // Output level metering
    std::atomic<float> outputLevelDb_{-60.0f};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessor)
};
