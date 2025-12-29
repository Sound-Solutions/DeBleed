#pragma once

#include <JuceHeader.h>
#include "NeuralGateEngine.h"
#include "STFTProcessor.h"
#include "TrainerProcess.h"

/**
 * DeBleedAudioProcessor - Main audio processor for the DeBleed Neural Gate plugin.
 *
 * Handles:
 * - Real-time audio processing with neural mask estimation
 * - STFT/iSTFT for spectral domain processing
 * - Model loading and hot-swapping
 * - Training process management
 * - Plugin parameters (Strength, Mix, Bypass)
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
    bool isModelLoaded() const { return neuralEngine.isModelLoaded(); }
    juce::String getModelPath() const { return neuralEngine.getModelPath(); }

    // Training interface
    TrainerProcess& getTrainerProcess() { return trainerProcess; }

    // Parameter access
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }

    // Parameter IDs
    static const juce::String PARAM_STRENGTH;
    static const juce::String PARAM_MIX;
    static const juce::String PARAM_BYPASS;

private:
    // Create parameter layout
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Parameters
    juce::AudioProcessorValueTreeState parameters;

    // Atomic parameter values for real-time access
    std::atomic<float> strength{1.0f};
    std::atomic<float> mix{1.0f};
    std::atomic<bool> bypassed{false};

    // Audio processing components
    NeuralGateEngine neuralEngine;
    STFTProcessor stftProcessor;

    // Training process
    TrainerProcess trainerProcess;

    // Processing state
    double currentSampleRate = 48000.0;
    int currentBlockSize = 512;

    // Temporary buffers
    std::vector<float> processBuffer;
    std::vector<float> maskBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessor)
};
