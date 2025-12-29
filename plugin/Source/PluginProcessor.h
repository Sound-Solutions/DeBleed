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

    // Latency info
    bool isLowLatencyMode() const { return lowLatency.load(); }
    float getLatencyMs() const { return getLatencySamples() * 1000.0f / static_cast<float>(currentSampleRate); }

    // Training interface
    TrainerProcess& getTrainerProcess() { return trainerProcess; }

    // Parameter access
    juce::AudioProcessorValueTreeState& getParameters() { return parameters; }

    // Parameter IDs
    static const juce::String PARAM_STRENGTH;
    static const juce::String PARAM_MIX;
    static const juce::String PARAM_BYPASS;
    static const juce::String PARAM_LOW_LATENCY;

private:
    // Create parameter layout
    juce::AudioProcessorValueTreeState::ParameterLayout createParameterLayout();

    // Parameters
    juce::AudioProcessorValueTreeState parameters;

    // Atomic parameter values for real-time access
    std::atomic<float> strength{1.0f};
    std::atomic<float> mix{1.0f};
    std::atomic<bool> bypassed{false};
    std::atomic<bool> lowLatency{false};
    std::atomic<bool> needsReinit{false};

    // Audio processing components
    NeuralGateEngine neuralEngine;
    STFTProcessor stftProcessor;

    // Training process
    TrainerProcess trainerProcess;

    // Processing state
    static constexpr double TARGET_SAMPLE_RATE = 48000.0;
    double currentSampleRate = 48000.0;
    int currentBlockSize = 512;
    double resampleRatio = 1.0;
    bool needsResampling = false;

    // Resampling
    juce::LagrangeInterpolator inputResampler;
    juce::LagrangeInterpolator outputResampler;
    std::vector<float> resampledInput;
    std::vector<float> resampledOutput;

    // Resampler history for continuity across blocks
    static constexpr int RESAMPLER_HISTORY_SIZE = 32;
    std::vector<float> inputHistory;
    std::vector<float> outputHistory;

    // Temporary buffers
    std::vector<float> processBuffer;
    std::vector<float> maskBuffer;

    // Transpose buffers for neural network (STFT outputs [frames,bins], NN expects [bins,frames])
    std::vector<float> transposedMagnitude;
    std::vector<float> transposedMask;

    // Mask smoothing (like Waves PSE / Shure 5045)
    std::vector<float> smoothedMask;
    static constexpr float MASK_SMOOTHING_ATTACK = 0.3f;   // Fast attack (30% per frame ~10ms)
    static constexpr float MASK_SMOOTHING_RELEASE = 0.005f; // Very slow release (0.5% per frame ~500ms)

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessor)
};
