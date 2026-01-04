#include "PluginProcessor.h"
#include "PluginEditor.h"

// Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_MIX = "mix";
const juce::String DeBleedAudioProcessor::PARAM_BYPASS = "bypass";
const juce::String DeBleedAudioProcessor::PARAM_LIVE_MODE = "liveMode";

// Phase 3 Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_OUTPUT_GAIN = "outputGain";
const juce::String DeBleedAudioProcessor::PARAM_HPF_FREQ = "hpfFreq";
const juce::String DeBleedAudioProcessor::PARAM_LPF_FREQ = "lpfFreq";
const juce::String DeBleedAudioProcessor::PARAM_SENSITIVITY = "sensitivity";
const juce::String DeBleedAudioProcessor::PARAM_SMOOTHING = "smoothing";

DeBleedAudioProcessor::DeBleedAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("DeBleedParams"), createParameterLayout())
{
    // Add parameter listeners
    parameters.addParameterListener(PARAM_MIX, this);
    parameters.addParameterListener(PARAM_BYPASS, this);
    parameters.addParameterListener(PARAM_OUTPUT_GAIN, this);
    parameters.addParameterListener(PARAM_HPF_FREQ, this);
    parameters.addParameterListener(PARAM_LPF_FREQ, this);
    parameters.addParameterListener(PARAM_SENSITIVITY, this);
    parameters.addParameterListener(PARAM_SMOOTHING, this);

    // Initialize atomic values
    mix.store(*parameters.getRawParameterValue(PARAM_MIX));
    bypassed.store(*parameters.getRawParameterValue(PARAM_BYPASS) > 0.5f);
    outputGain.store(*parameters.getRawParameterValue(PARAM_OUTPUT_GAIN));
    hpfFreq.store(*parameters.getRawParameterValue(PARAM_HPF_FREQ));
    lpfFreq.store(*parameters.getRawParameterValue(PARAM_LPF_FREQ));
    sensitivity.store(*parameters.getRawParameterValue(PARAM_SENSITIVITY));
    smoothing.store(*parameters.getRawParameterValue(PARAM_SMOOTHING));
}

DeBleedAudioProcessor::~DeBleedAudioProcessor()
{
    parameters.removeParameterListener(PARAM_MIX, this);
    parameters.removeParameterListener(PARAM_BYPASS, this);
    parameters.removeParameterListener(PARAM_OUTPUT_GAIN, this);
    parameters.removeParameterListener(PARAM_HPF_FREQ, this);
    parameters.removeParameterListener(PARAM_LPF_FREQ, this);
    parameters.removeParameterListener(PARAM_SENSITIVITY, this);
    parameters.removeParameterListener(PARAM_SMOOTHING, this);
}

juce::AudioProcessorValueTreeState::ParameterLayout DeBleedAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

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

    // Live Mode - prevents accidental training during live shows (UI-only)
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_LIVE_MODE, 1},
        "Live Mode",
        false
    ));

    // Phase 3: Neural 5045 Parameters

    // Output Gain - master output level in dB
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_OUTPUT_GAIN, 1},
        "Output Gain",
        juce::NormalisableRange<float>(-24.0f, 12.0f, 0.1f),
        0.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) {
            if (value >= 0.0f)
                return "+" + juce::String(value, 1) + " dB";
            return juce::String(value, 1) + " dB";
        },
        nullptr
    ));

    // HPF Frequency - highpass filter cutoff
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_HPF_FREQ, 1},
        "HPF Freq",
        juce::NormalisableRange<float>(20.0f, 500.0f, 1.0f, 0.5f), // Skewed for better low-end control
        20.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(static_cast<int>(value)) + " Hz"; },
        nullptr
    ));

    // LPF Frequency - lowpass filter cutoff
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_LPF_FREQ, 1},
        "LPF Freq",
        juce::NormalisableRange<float>(5000.0f, 20000.0f, 1.0f, 0.5f), // Skewed for better high-end control
        20000.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) {
            if (value >= 1000.0f)
                return juce::String(value / 1000.0f, 1) + " kHz";
            return juce::String(static_cast<int>(value)) + " Hz";
        },
        nullptr
    ));

    // Sensitivity - how aggressively to apply neural predictions (0 = subtle, 1 = full)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_SENSITIVITY, 1},
        "Sensitivity",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        1.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(static_cast<int>(value * 100)) + "%"; },
        nullptr
    ));

    // Smoothing - coefficient interpolation time in ms
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_SMOOTHING, 1},
        "Smoothing",
        juce::NormalisableRange<float>(1.0f, 200.0f, 1.0f, 0.5f),
        50.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(static_cast<int>(value)) + " ms"; },
        nullptr
    ));

    return {params.begin(), params.end()};
}

void DeBleedAudioProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == PARAM_MIX)
        mix.store(newValue);
    else if (parameterID == PARAM_BYPASS)
        bypassed.store(newValue > 0.5f);
    else if (parameterID == PARAM_OUTPUT_GAIN)
        outputGain.store(newValue);
    else if (parameterID == PARAM_HPF_FREQ)
        hpfFreq.store(newValue);
    else if (parameterID == PARAM_LPF_FREQ)
        lpfFreq.store(newValue);
    else if (parameterID == PARAM_SENSITIVITY)
        sensitivity.store(newValue);
    else if (parameterID == PARAM_SMOOTHING)
        smoothing.store(newValue);
}

void DeBleedAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate_ = sampleRate;
    currentBlockSize_ = samplesPerBlock;

    // Prepare Neural5045 engine for inference
    neural5045Engine_.prepare(sampleRate, samplesPerBlock);

    // Prepare the biquad filter chain
    biquadChain_.prepare(sampleRate, samplesPerBlock);

    // Allocate buffers
    int numChannels = std::max(getTotalNumInputChannels(), getTotalNumOutputChannels());
    dryBuffer_.setSize(numChannels, samplesPerBlock);
    monoSidechain_.resize(samplesPerBlock);

    // Zero latency - IIR filters are causal, no lookahead
    setLatencySamples(0);

    DBG("DeBleedAudioProcessor prepared: " << sampleRate << " Hz, "
        << samplesPerBlock << " samples/block, " << numChannels << " channels");
}

void DeBleedAudioProcessor::releaseResources()
{
    // Reset filter states
    biquadChain_.reset();
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

    // Get parameter values
    float currentMix = mix.load();

    // Early bypass if mix is 0%
    if (currentMix < 0.001f)
        return;

    // =========================================================================
    // Phase 3: Update biquad chain with user overrides
    // =========================================================================
    float currentHpfFreq = hpfFreq.load();
    float currentLpfFreq = lpfFreq.load();
    float currentOutputGain = outputGain.load();
    float currentSensitivity = sensitivity.load();
    float currentSmoothing = smoothing.load();

    // HPF: Override if > 20Hz (user has adjusted from default)
    biquadChain_.setHPFOverride(currentHpfFreq > 20.5f ? currentHpfFreq : -1.0f);

    // LPF: Override if < 20kHz (user has adjusted from default)
    biquadChain_.setLPFOverride(currentLpfFreq < 19500.0f ? currentLpfFreq : 20001.0f);

    // Output gain offset
    biquadChain_.setOutputGainOffset(currentOutputGain);

    // Sensitivity
    biquadChain_.setSensitivity(currentSensitivity);

    // Smoothing time
    biquadChain_.setSmoothingTime(currentSmoothing);

    // =========================================================================
    // Neural 5045 Processing
    // =========================================================================

    // 1. Store dry signal for wet/dry mix
    if (currentMix < 0.999f)
    {
        for (int ch = 0; ch < totalNumInputChannels; ++ch)
            dryBuffer_.copyFrom(ch, 0, buffer.getReadPointer(ch), numSamples);
    }

    // 2. Create mono sidechain for neural network analysis
    //    Sum all input channels to mono (normalized by channel count)
    if (totalNumInputChannels > 0 && numSamples > 0)
    {
        // Ensure buffer is large enough
        if (static_cast<int>(monoSidechain_.size()) < numSamples)
            monoSidechain_.resize(numSamples);

        // Start with first channel
        const float* ch0 = buffer.getReadPointer(0);
        std::copy(ch0, ch0 + numSamples, monoSidechain_.begin());

        // Add remaining channels
        for (int ch = 1; ch < totalNumInputChannels; ++ch)
        {
            const float* chData = buffer.getReadPointer(ch);
            for (int s = 0; s < numSamples; ++s)
                monoSidechain_[s] += chData[s];
        }

        // Normalize by channel count
        if (totalNumInputChannels > 1)
        {
            float normFactor = 1.0f / static_cast<float>(totalNumInputChannels);
            for (int s = 0; s < numSamples; ++s)
                monoSidechain_[s] *= normFactor;
        }
    }

    // 3. Run neural network to predict EQ parameters
    //    Returns pointer to [N_PARAMS × numFrames] array
    const float* allParams = neural5045Engine_.process(monoSidechain_.data(), numSamples);
    int numFrames = neural5045Engine_.getNumFrames();

    // 4. Process audio frame-by-frame with corresponding neural parameters
    //    Each frame is FRAME_SIZE samples (2048 = ~21ms at 96kHz)
    //    This ensures the filter coefficients track the neural network predictions
    constexpr int FRAME_SIZE = Neural5045Engine::FRAME_SIZE;

    int samplesRemaining = numSamples;
    int bufferOffset = 0;

    for (int frame = 0; frame < numFrames && samplesRemaining > 0; ++frame)
    {
        // Get parameters for this frame
        const float* frameParams = neural5045Engine_.getFrameParams(frame);
        if (frameParams != nullptr)
        {
            biquadChain_.setParameters(frameParams);
        }

        // Calculate samples to process in this frame
        int samplesThisFrame = std::min(FRAME_SIZE, samplesRemaining);

        // Create a sub-buffer for this frame's audio
        juce::AudioBuffer<float> frameBuffer(buffer.getArrayOfWritePointers(),
                                              totalNumInputChannels,
                                              bufferOffset,
                                              samplesThisFrame);

        // 5. Process this frame through the filter cascade
        //    HPF → LowShelf → [12x Peaking] → HighShelf → LPF → Gain
        biquadChain_.process(frameBuffer);

        bufferOffset += samplesThisFrame;
        samplesRemaining -= samplesThisFrame;
    }

    // 6. Apply wet/dry mix
    if (currentMix < 0.999f)
    {
        float wetGain = currentMix;
        float dryGain = 1.0f - currentMix;

        for (int ch = 0; ch < totalNumInputChannels; ++ch)
        {
            float* wetData = buffer.getWritePointer(ch);
            const float* dryData = dryBuffer_.getReadPointer(ch);

            for (int s = 0; s < numSamples; ++s)
                wetData[s] = wetData[s] * wetGain + dryData[s] * dryGain;
        }
    }
}

bool DeBleedAudioProcessor::loadModel(const juce::String& modelPath)
{
    // Reset biquad chain when loading new model
    biquadChain_.reset();

    // Load the Neural5045 ONNX model
    bool success = neural5045Engine_.loadModel(modelPath);

    if (success)
    {
        DBG("Neural5045 model loaded: " << modelPath);
    }
    else
    {
        DBG("Failed to load Neural5045 model: " << neural5045Engine_.getLastError());
    }

    return success;
}

void DeBleedAudioProcessor::unloadModel()
{
    neural5045Engine_.unloadModel();
    biquadChain_.reset();
}

juce::AudioProcessorEditor* DeBleedAudioProcessor::createEditor()
{
    return new DeBleedAudioProcessorEditor(*this);
}

void DeBleedAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();

    // Add model path to state
    state.setProperty("modelPath", neural5045Engine_.getModelPath(), nullptr);

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

// Plugin instantiation
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeBleedAudioProcessor();
}
