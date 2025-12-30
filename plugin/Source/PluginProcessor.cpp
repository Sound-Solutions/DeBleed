#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <algorithm>

// Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_STRENGTH = "strength";
const juce::String DeBleedAudioProcessor::PARAM_MIX = "mix";
const juce::String DeBleedAudioProcessor::PARAM_BYPASS = "bypass";
const juce::String DeBleedAudioProcessor::PARAM_LOW_LATENCY = "lowLatency";
const juce::String DeBleedAudioProcessor::PARAM_ATTACK = "attack";
const juce::String DeBleedAudioProcessor::PARAM_RELEASE = "release";
const juce::String DeBleedAudioProcessor::PARAM_THRESHOLD = "threshold";
const juce::String DeBleedAudioProcessor::PARAM_FLOOR = "floor";
const juce::String DeBleedAudioProcessor::PARAM_LIVE_MODE = "liveMode";

DeBleedAudioProcessor::DeBleedAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("DeBleedParams"), createParameterLayout())
{
    // Add parameter listeners
    parameters.addParameterListener(PARAM_STRENGTH, this);
    parameters.addParameterListener(PARAM_MIX, this);
    parameters.addParameterListener(PARAM_BYPASS, this);
    parameters.addParameterListener(PARAM_LOW_LATENCY, this);
    parameters.addParameterListener(PARAM_ATTACK, this);
    parameters.addParameterListener(PARAM_RELEASE, this);
    parameters.addParameterListener(PARAM_THRESHOLD, this);
    parameters.addParameterListener(PARAM_FLOOR, this);

    // Initialize atomic values
    strength.store(*parameters.getRawParameterValue(PARAM_STRENGTH));
    mix.store(*parameters.getRawParameterValue(PARAM_MIX));
    bypassed.store(*parameters.getRawParameterValue(PARAM_BYPASS) > 0.5f);
    lowLatency.store(*parameters.getRawParameterValue(PARAM_LOW_LATENCY) > 0.5f);
    attackMs.store(*parameters.getRawParameterValue(PARAM_ATTACK));
    releaseMs.store(*parameters.getRawParameterValue(PARAM_RELEASE));
    threshold.store(*parameters.getRawParameterValue(PARAM_THRESHOLD));
    floorDb.store(*parameters.getRawParameterValue(PARAM_FLOOR));
}

DeBleedAudioProcessor::~DeBleedAudioProcessor()
{
    parameters.removeParameterListener(PARAM_STRENGTH, this);
    parameters.removeParameterListener(PARAM_MIX, this);
    parameters.removeParameterListener(PARAM_BYPASS, this);
    parameters.removeParameterListener(PARAM_LOW_LATENCY, this);
    parameters.removeParameterListener(PARAM_ATTACK, this);
    parameters.removeParameterListener(PARAM_RELEASE, this);
    parameters.removeParameterListener(PARAM_THRESHOLD, this);
    parameters.removeParameterListener(PARAM_FLOOR, this);
}

juce::AudioProcessorValueTreeState::ParameterLayout DeBleedAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Strength: How aggressively to apply the mask (0.0 = pass-through, 1.0 = full)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_STRENGTH, 1},
        "Strength",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        1.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value * 100.0f, 0) + "%"; },
        nullptr
    ));

    // Mix: Dry/Wet blend
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_MIX, 1},
        "Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        1.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value * 100.0f, 0) + "%"; },
        nullptr
    ));

    // Bypass
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_BYPASS, 1},
        "Bypass",
        false
    ));

    // Low Latency Mode
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_LOW_LATENCY, 1},
        "Low Latency",
        false
    ));

    // Attack time (ms) - how fast gate opens
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_ATTACK, 1},
        "Attack",
        juce::NormalisableRange<float>(0.1f, 100.0f, 0.1f, 0.5f),  // Skewed for finer control at low end
        10.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " ms"; },
        nullptr
    ));

    // Release time (ms) - how fast gate closes (gain falls toward floor)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_RELEASE, 1},
        "Release",
        juce::NormalisableRange<float>(10.0f, 2000.0f, 1.0f, 0.4f),
        500.0f,  // Slow release for smooth gate close
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 0) + " ms"; },
        nullptr
    ));

    // Threshold - input magnitude (dB) below this keeps gate closed
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_THRESHOLD, 1},
        "Threshold",
        juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),
        -80.0f,  // Default: effectively disabled (very low)
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; },
        nullptr
    ));

    // Range - depth of attenuation when gated (mask=0 gives this level)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_FLOOR, 1},
        "Range",
        juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),
        -80.0f,  // Default: full attenuation when gated
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; },
        nullptr
    ));

    // Live Mode - prevents accidental training during live shows (UI-only, no audio effect)
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_LIVE_MODE, 1},
        "Live Mode",
        false  // Default: off (training enabled)
    ));

    return {params.begin(), params.end()};
}

void DeBleedAudioProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == PARAM_STRENGTH)
        strength.store(newValue);
    else if (parameterID == PARAM_MIX)
        mix.store(newValue);
    else if (parameterID == PARAM_BYPASS)
        bypassed.store(newValue > 0.5f);
    else if (parameterID == PARAM_LOW_LATENCY)
    {
        bool newLowLatency = newValue > 0.5f;
        if (lowLatency.load() != newLowLatency)
        {
            lowLatency.store(newLowLatency);
            // Mark that we need to reinitialize STFT processor
            needsReinit.store(true);
        }
    }
    else if (parameterID == PARAM_ATTACK)
        attackMs.store(newValue);
    else if (parameterID == PARAM_RELEASE)
        releaseMs.store(newValue);
    else if (parameterID == PARAM_THRESHOLD)
        threshold.store(newValue);
    else if (parameterID == PARAM_FLOOR)
        floorDb.store(newValue);
}

int DeBleedAudioProcessor::freqToBin(float freqHz) const
{
    // Convert frequency to FFT bin index
    float binFloat = freqHz * stftProcessor.getFFTSize() / static_cast<float>(TARGET_SAMPLE_RATE);
    return std::clamp(static_cast<int>(binFloat), 0, STFTProcessor::N_FREQ_BINS - 1);
}

void DeBleedAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;

    // Check if we need to resample (for sidechain analysis)
    needsResampling = (std::abs(sampleRate - TARGET_SAMPLE_RATE) > 1.0);
    resampleRatio = TARGET_SAMPLE_RATE / sampleRate;

    // Calculate resampled block size for sidechain
    int resampledBlockSize = needsResampling
        ? static_cast<int>(std::ceil(samplesPerBlock * resampleRatio)) + 16
        : samplesPerBlock;

    // === NEW: Dynamic Hunter Filter Pool (Zero-Latency Audio Path) ===
    activeFilterPool.prepare(sampleRate, samplesPerBlock);

    // === Sidechain Analyzer (Control Path) ===
    // Note: We still use 192-band analysis internally, but filter pool uses raw 129-bin mask
    std::array<float, 192> dummyFreqs;  // Sidechain analyzer expects this but we don't use it anymore
    for (int i = 0; i < 192; ++i)
        dummyFreqs[i] = 20.0f * std::pow(1000.0f, static_cast<float>(i) / 191.0f);
    sidechainAnalyzer.prepare(TARGET_SAMPLE_RATE, resampledBlockSize, dummyFreqs);

    // Allocate sidechain buffer
    sidechainBuffer.resize(resampledBlockSize, 0.0f);

    // === LEGACY: Keep STFT/Neural for visualization and fallback ===
    stftProcessor.setMode(lowLatency.load() ? STFTProcessor::Mode::LowLatency : STFTProcessor::Mode::Quality);
    stftProcessor.prepare(TARGET_SAMPLE_RATE, resampledBlockSize);

    int hopLength = stftProcessor.getHopLength();
    int maxFrames = (resampledBlockSize / hopLength) + 8;
    neuralEngine.prepare(maxFrames);

    // Allocate buffers (kept for visualization)
    processBuffer.resize(resampledBlockSize * 2, 0.0f);
    maskBuffer.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);
    transposedMagnitude.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 0.0f);
    transposedMask.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);

    if (smoothedMask.size() != STFTProcessor::N_FREQ_BINS)
    {
        smoothedMask.resize(STFTProcessor::N_FREQ_BINS, 1.0f);
    }

    // Allocate resampling buffers for sidechain
    if (needsResampling)
    {
        resampledInput.resize(resampledBlockSize + 64, 0.0f);
        resampledOutput.resize(samplesPerBlock + 64, 0.0f);

        inputHistory.resize(RESAMPLER_HISTORY_SIZE, 0.0f);
        outputHistory.resize(RESAMPLER_HISTORY_SIZE, 0.0f);

        inputResampler.reset();
        outputResampler.reset();
    }

    // === ZERO LATENCY ===
    // IIR filters are causal - no lookahead needed
    // Only report minimal latency for filter group delay (~1-2 samples)
    setLatencySamples(0);
}

void DeBleedAudioProcessor::releaseResources()
{
    activeFilterPool.reset();
    sidechainAnalyzer.reset();
    stftProcessor.reset();
}

bool DeBleedAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Support mono and stereo
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() &&
        layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // Input must match output
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void DeBleedAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();
    int numSamples = buffer.getNumSamples();

    // Clear any unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, numSamples);

    // Early return if bypassed
    if (bypassed.load())
        return;

    // Check if model is loaded in sidechain analyzer
    if (!sidechainAnalyzer.isModelLoaded())
        return;  // Pass through (no processing)

    // Get parameter values
    float currentStrength = strength.load();
    float currentMix = mix.load();

    // Early bypass if mix is 0%
    if (currentMix < 0.001f)
        return;

    const float* inputData = buffer.getReadPointer(0);

    // Keep a copy of the dry signal for mix
    std::vector<float> drySignal(inputData, inputData + numSamples);

    // === SIDECHAIN ANALYSIS (Control Path) ===
    // Prepare data for sidechain (resample to 48kHz if needed)
    const float* analysisInput = inputData;
    int analysisNumSamples = numSamples;

    if (needsResampling)
    {
        analysisNumSamples = static_cast<int>(numSamples * resampleRatio);
        sidechainBuffer.resize(analysisNumSamples + 16);

        std::vector<float> inputWithHistory(RESAMPLER_HISTORY_SIZE + numSamples);
        std::memcpy(inputWithHistory.data(), inputHistory.data(), RESAMPLER_HISTORY_SIZE * sizeof(float));
        std::memcpy(inputWithHistory.data() + RESAMPLER_HISTORY_SIZE, inputData, numSamples * sizeof(float));

        inputResampler.process(1.0 / resampleRatio,
                               inputWithHistory.data() + RESAMPLER_HISTORY_SIZE,
                               sidechainBuffer.data(),
                               analysisNumSamples);

        int historyStart = numSamples - RESAMPLER_HISTORY_SIZE;
        if (historyStart >= 0)
            std::memcpy(inputHistory.data(), inputData + historyStart, RESAMPLER_HISTORY_SIZE * sizeof(float));

        analysisInput = sidechainBuffer.data();
    }

    // Update sidechain analyzer parameters
    sidechainAnalyzer.setStrength(currentStrength);
    sidechainAnalyzer.setAttack(attackMs.load());
    sidechainAnalyzer.setRelease(releaseMs.load());
    sidechainAnalyzer.setThreshold(threshold.load());
    sidechainAnalyzer.setFloor(floorDb.load());

    // Run sidechain analysis (STFT → Neural Net → Band Mapping → Envelopes)
    sidechainAnalyzer.analyze(analysisInput, analysisNumSamples);

    // Get raw neural mask (129 bins) for the hunter filter pool
    const float* rawMask = sidechainAnalyzer.getRawMask();

    // === AUDIO PATH: Dynamic Hunter Filters ===
    // The ActiveFilterPool finds valleys in the mask and assigns 32 filters to chase them
    activeFilterPool.setStrength(currentStrength);
    activeFilterPool.setFloorDb(floorDb.load());
    activeFilterPool.process(buffer, rawMask);

    // === VISUALIZATION ===
    // Also run through legacy STFT for visualization data
    int numFrames = stftProcessor.processBlock(analysisInput, analysisNumSamples);
    if (numFrames > 0)
    {
        const float* magnitude = stftProcessor.getMagnitudeData();
        const float* rawMask = sidechainAnalyzer.getRawMask();

        if (rawMask != nullptr)
        {
            for (int frame = 0; frame < numFrames; ++frame)
            {
                visualizationData.pushFrame(
                    magnitude + frame * STFTProcessor::N_FREQ_BINS,
                    rawMask  // Use sidechain analyzer's mask for visualization
                );
            }
        }
    }

    // Update gain reduction meter
    float avgReductionDb = sidechainAnalyzer.getAverageGainReduction();
    visualizationData.averageGainReductionDb.store(avgReductionDb);

    // === MIX: Dry/Wet blend ===
    float* outputData = buffer.getWritePointer(0);
    if (currentMix < 1.0f)
    {
        float wet = currentMix;
        float dry = 1.0f - currentMix;

        for (int i = 0; i < numSamples; ++i)
        {
            outputData[i] = outputData[i] * wet + drySignal[i] * dry;
        }
    }

    // === SAFETY: Final hard limit ===
    for (int i = 0; i < numSamples; ++i)
    {
        if (std::isnan(outputData[i]) || std::isinf(outputData[i]))
            outputData[i] = 0.0f;
        else
            outputData[i] = std::clamp(outputData[i], -1.0f, 1.0f);
    }

    // Copy to other channels if stereo
    for (int channel = 1; channel < totalNumOutputChannels; ++channel)
    {
        buffer.copyFrom(channel, 0, outputData, numSamples);
    }
}

bool DeBleedAudioProcessor::loadModel(const juce::String& modelPath)
{
    // Load model in both the legacy engine (for visualization) and sidechain analyzer
    bool legacyLoaded = neuralEngine.loadModel(modelPath);
    bool sidechainLoaded = sidechainAnalyzer.loadModel(modelPath);
    return legacyLoaded && sidechainLoaded;
}

void DeBleedAudioProcessor::unloadModel()
{
    neuralEngine.unloadModel();
    sidechainAnalyzer.unloadModel();
}

juce::AudioProcessorEditor* DeBleedAudioProcessor::createEditor()
{
    return new DeBleedAudioProcessorEditor(*this);
}

void DeBleedAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();

    // Add model path to state
    state.setProperty("modelPath", neuralEngine.getModelPath(), nullptr);

    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void DeBleedAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

    if (xmlState != nullptr)
    {
        if (xmlState->hasTagName(parameters.state.getType()))
        {
            parameters.replaceState(juce::ValueTree::fromXml(*xmlState));

            // Restore model
            juce::String modelPath = parameters.state.getProperty("modelPath", "").toString();
            if (modelPath.isNotEmpty())
            {
                loadModel(modelPath);
            }
        }
    }
}

std::array<ActiveFilterPool::FilterState, DeBleedAudioProcessor::NUM_HUNTERS> DeBleedAudioProcessor::getHunterStates() const
{
    return activeFilterPool.getFilterStates();
}

// Plugin instantiation
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeBleedAudioProcessor();
}
