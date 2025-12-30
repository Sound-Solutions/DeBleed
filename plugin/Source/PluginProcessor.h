#pragma once

#include <JuceHeader.h>
#include "NeuralGateEngine.h"
#include "STFTProcessor.h"
#include "TrainerProcess.h"
#include "ActiveFilterPool.h"
#include "SidechainAnalyzer.h"

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
    static const juce::String PARAM_ATTACK;
    static const juce::String PARAM_RELEASE;
    static const juce::String PARAM_THRESHOLD;
    static const juce::String PARAM_FLOOR;
    static const juce::String PARAM_LIVE_MODE;

    // Visualization data for thread-safe audio->GUI transfer
    struct VisualizationData
    {
        static constexpr int N_FREQ_BINS = 129;
        static constexpr int STREAM_B_BINS = 128;
        static constexpr int FIFO_SIZE = 64;

        struct FrameData
        {
            std::array<float, N_FREQ_BINS> magnitude{};
            std::array<float, N_FREQ_BINS> mask{};       // Stream A: 187.5 Hz/bin
            std::array<float, STREAM_B_BINS> maskB{};    // Stream B: 23.4 Hz/bin (high-res lows)
        };

        juce::AbstractFifo fifo{FIFO_SIZE};
        std::array<FrameData, FIFO_SIZE> frameBuffer;
        std::atomic<float> averageGainReductionDb{0.0f};

        void pushFrame(const float* mag, const float* msk, const float* mskB = nullptr)
        {
            int start1, size1, start2, size2;
            fifo.prepareToWrite(1, start1, size1, start2, size2);
            if (size1 > 0)
            {
                std::memcpy(frameBuffer[start1].magnitude.data(), mag, N_FREQ_BINS * sizeof(float));
                std::memcpy(frameBuffer[start1].mask.data(), msk, N_FREQ_BINS * sizeof(float));
                if (mskB != nullptr)
                    std::memcpy(frameBuffer[start1].maskB.data(), mskB, STREAM_B_BINS * sizeof(float));
                else
                    frameBuffer[start1].maskB.fill(1.0f);  // Default to unity if not provided
                fifo.finishedWrite(1);
            }
        }

        bool popFrame(FrameData& out)
        {
            int start1, size1, start2, size2;
            fifo.prepareToRead(1, start1, size1, start2, size2);
            if (size1 > 0)
            {
                out = frameBuffer[start1];
                fifo.finishedRead(1);
                return true;
            }
            return false;
        }
    };

    VisualizationData& getVisualizationData() { return visualizationData; }

    // Hunter filter states for visualization (32 dynamic filters)
    static constexpr int NUM_HUNTERS = ActiveFilterPool::MAX_FILTERS;
    std::array<ActiveFilterPool::FilterState, NUM_HUNTERS> getHunterStates() const;

    // Access to the filter pool for visualization
    const ActiveFilterPool& getFilterPool() const { return activeFilterPool; }

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

    // New parameters
    std::atomic<float> attackMs{0.1f};   // Fast attack = gate opens quickly
    std::atomic<float> releaseMs{500.0f}; // Slow release = gate closes slowly
    std::atomic<float> threshold{0.0f};
    std::atomic<float> floorDb{-60.0f};

    // Visualization data
    VisualizationData visualizationData;

    // Helper to convert frequency to bin index
    int freqToBin(float freqHz) const;

    // Audio processing components (legacy FFT-based - kept for compatibility)
    NeuralGateEngine neuralEngine;
    STFTProcessor stftProcessor;

    // NEW: Dynamic Hunter Filter Pool (32 surgical filters)
    ActiveFilterPool activeFilterPool;
    SidechainAnalyzer sidechainAnalyzer;

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

    // Sidechain analysis buffer (copy of input for IIR mode)
    std::vector<float> sidechainBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessor)
};
