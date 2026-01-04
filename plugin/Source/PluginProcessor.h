#pragma once

#include <JuceHeader.h>
#include "Neural5045Engine.h"
#include "DifferentiableBiquadChain.h"
#include "TrainerProcess.h"

/**
 * DeBleedAudioProcessor - Neural 5045: DDSP Source Separation Plugin
 *
 * Uses differentiable IIR filter bank for zero-latency source separation.
 * Neural network predicts ~50 filter coefficients per frame.
 * Audio passes through series biquad cascade with no FFT latency.
 *
 * Architecture:
 *   Input → HPF → LowShelf → [12x Peaking] → HighShelf → LPF → BroadbandGain → Output
 *
 * Processing Flow:
 *   1. Create mono sidechain from input (for neural network analysis)
 *   2. Run Neural5045Engine to predict 50 EQ parameters per frame
 *   3. Apply parameters to DifferentiableBiquadChain (with smoothing)
 *   4. Process audio through the filter cascade (zero latency)
 *   5. Apply dry/wet mix
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

    // Model management
    bool loadModel(const juce::String& modelPath);
    void unloadModel();
    bool isModelLoaded() const { return neural5045Engine_.isModelLoaded(); }
    juce::String getModelPath() const { return neural5045Engine_.getModelPath(); }

    // Get biquad chain for visualization
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

    // Neural 5045 DSP components
    Neural5045Engine neural5045Engine_;      // ONNX inference for EQ parameters
    DifferentiableBiquadChain biquadChain_;  // SVF TPT filter cascade

    // Buffers for processing
    juce::AudioBuffer<float> dryBuffer_;     // Dry signal for wet/dry mix
    std::vector<float> monoSidechain_;       // Mono mix for neural network input

    // Training process
    TrainerProcess trainerProcess;

    // Processing state
    double currentSampleRate_ = 48000.0;
    int currentBlockSize_ = 512;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessor)
};
