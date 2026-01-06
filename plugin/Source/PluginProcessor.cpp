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

// V2 Expander Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_EXP_THRESHOLD = "expThreshold";
const juce::String DeBleedAudioProcessor::PARAM_EXP_RATIO = "expRatio";
const juce::String DeBleedAudioProcessor::PARAM_EXP_ATTACK = "expAttack";
const juce::String DeBleedAudioProcessor::PARAM_EXP_RELEASE = "expRelease";
const juce::String DeBleedAudioProcessor::PARAM_EXP_RANGE = "expRange";
const juce::String DeBleedAudioProcessor::PARAM_USE_V2 = "useV2";

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
    parameters.addParameterListener(PARAM_EXP_THRESHOLD, this);
    parameters.addParameterListener(PARAM_EXP_RATIO, this);
    parameters.addParameterListener(PARAM_EXP_ATTACK, this);
    parameters.addParameterListener(PARAM_EXP_RELEASE, this);
    parameters.addParameterListener(PARAM_EXP_RANGE, this);
    parameters.addParameterListener(PARAM_USE_V2, this);

    // Initialize atomic values
    mix.store(*parameters.getRawParameterValue(PARAM_MIX));
    bypassed.store(*parameters.getRawParameterValue(PARAM_BYPASS) > 0.5f);
    outputGain.store(*parameters.getRawParameterValue(PARAM_OUTPUT_GAIN));
    hpfFreq.store(*parameters.getRawParameterValue(PARAM_HPF_FREQ));
    lpfFreq.store(*parameters.getRawParameterValue(PARAM_LPF_FREQ));
    sensitivity.store(*parameters.getRawParameterValue(PARAM_SENSITIVITY));
    smoothing.store(*parameters.getRawParameterValue(PARAM_SMOOTHING));
    expThreshold.store(*parameters.getRawParameterValue(PARAM_EXP_THRESHOLD));
    expRatio.store(*parameters.getRawParameterValue(PARAM_EXP_RATIO));
    expAttack.store(*parameters.getRawParameterValue(PARAM_EXP_ATTACK));
    expRelease.store(*parameters.getRawParameterValue(PARAM_EXP_RELEASE));
    expRange.store(*parameters.getRawParameterValue(PARAM_EXP_RANGE));
    useV2.store(*parameters.getRawParameterValue(PARAM_USE_V2) > 0.5f);
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
    parameters.removeParameterListener(PARAM_EXP_THRESHOLD, this);
    parameters.removeParameterListener(PARAM_EXP_RATIO, this);
    parameters.removeParameterListener(PARAM_EXP_ATTACK, this);
    parameters.removeParameterListener(PARAM_EXP_RELEASE, this);
    parameters.removeParameterListener(PARAM_EXP_RANGE, this);
    parameters.removeParameterListener(PARAM_USE_V2, this);
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

    // =========================================================================
    // V2 Architecture Parameters - Expander
    // =========================================================================

    // Use V2 - toggle between v1 (neural EQ) and v2 (VAD + dynamic EQ + expander)
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_USE_V2, 1},
        "Use V2",
        true  // Default to v2 architecture
    ));

    // Expander Threshold - level below which expansion kicks in
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_EXP_THRESHOLD, 1},
        "Exp Threshold",
        juce::NormalisableRange<float>(-60.0f, 0.0f, 0.1f),
        -40.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; },
        nullptr
    ));

    // Expander Ratio
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_EXP_RATIO, 1},
        "Exp Ratio",
        juce::NormalisableRange<float>(1.0f, 20.0f, 0.1f, 0.5f),
        4.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + ":1"; },
        nullptr
    ));

    // Expander Attack
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_EXP_ATTACK, 1},
        "Exp Attack",
        juce::NormalisableRange<float>(0.1f, 50.0f, 0.1f, 0.5f),
        1.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " ms"; },
        nullptr
    ));

    // Expander Release
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_EXP_RELEASE, 1},
        "Exp Release",
        juce::NormalisableRange<float>(10.0f, 500.0f, 1.0f, 0.5f),
        100.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(static_cast<int>(value)) + " ms"; },
        nullptr
    ));

    // Expander Range - maximum gain reduction
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_EXP_RANGE, 1},
        "Exp Range",
        juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),  // Extended to -80dB for full gating
        -40.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; },
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
    // V2 Expander parameters
    else if (parameterID == PARAM_EXP_THRESHOLD)
    {
        expThreshold.store(newValue);
        expander_.setThresholdDb(newValue);
    }
    else if (parameterID == PARAM_EXP_RATIO)
    {
        expRatio.store(newValue);
        expander_.setRatio(newValue);
    }
    else if (parameterID == PARAM_EXP_ATTACK)
    {
        expAttack.store(newValue);
        expander_.setAttackMs(newValue);
    }
    else if (parameterID == PARAM_EXP_RELEASE)
    {
        expRelease.store(newValue);
        expander_.setReleaseMs(newValue);
    }
    else if (parameterID == PARAM_EXP_RANGE)
    {
        expRange.store(newValue);
        expander_.setRangeDb(newValue);
    }
    else if (parameterID == PARAM_USE_V2)
    {
        useV2.store(newValue > 0.5f);
        DBG("V2 mode changed to: " << (newValue > 0.5f ? "ENABLED" : "DISABLED"));
    }
}

void DeBleedAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate_ = sampleRate;
    currentBlockSize_ = samplesPerBlock;

    // V1: Prepare Neural5045 engine for inference
    neural5045Engine_.prepare(sampleRate, samplesPerBlock);
    biquadChain_.prepare(sampleRate, samplesPerBlock);

    // V2: Prepare zero-latency components
    spectralVAD_.prepare(sampleRate);
    dynamicEQ_.prepare(sampleRate, samplesPerBlock);
    expander_.prepare(sampleRate);

    // Initialize expander with current parameter values
    expander_.setThresholdDb(expThreshold.load());
    expander_.setRatio(expRatio.load());
    expander_.setAttackMs(expAttack.load());
    expander_.setReleaseMs(expRelease.load());
    expander_.setRangeDb(expRange.load());

    // Allocate buffers
    int numChannels = std::max(getTotalNumInputChannels(), getTotalNumOutputChannels());
    dryBuffer_.setSize(numChannels, samplesPerBlock);
    monoSidechain_.resize(samplesPerBlock);
    vadConfidence_.resize(samplesPerBlock);

    // Zero latency - IIR filters are causal, no lookahead
    setLatencySamples(0);

    DBG("DeBleedAudioProcessor prepared: " << sampleRate << " Hz, "
        << samplesPerBlock << " samples/block, " << numChannels << " channels");
}

void DeBleedAudioProcessor::releaseResources()
{
    // Reset V1 filter states
    biquadChain_.reset();

    // Reset V2 components
    spectralVAD_.reset();
    dynamicEQ_.reset();
    expander_.reset();
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

    // Get common parameter values
    float currentMix = mix.load();
    float currentOutputGain = outputGain.load();

    // Early bypass if mix is 0%
    if (currentMix < 0.001f)
        return;

    // Store dry signal for wet/dry mix
    if (currentMix < 0.999f)
    {
        for (int ch = 0; ch < totalNumInputChannels; ++ch)
            dryBuffer_.copyFrom(ch, 0, buffer.getReadPointer(ch), numSamples);
    }

    // =========================================================================
    // V2 Architecture: VAD + Expander (Zero Latency)
    // =========================================================================

    // Ensure buffers are large enough
    if (static_cast<int>(monoSidechain_.size()) < numSamples)
        monoSidechain_.resize(numSamples);
    if (static_cast<int>(vadConfidence_.size()) < numSamples)
        vadConfidence_.resize(numSamples);

    // Create mono sidechain for VAD
    const float* ch0 = buffer.getReadPointer(0);
    std::copy(ch0, ch0 + numSamples, monoSidechain_.begin());

    if (totalNumInputChannels > 1)
    {
        for (int ch = 1; ch < totalNumInputChannels; ++ch)
        {
            const float* chData = buffer.getReadPointer(ch);
            for (int s = 0; s < numSamples; ++s)
                monoSidechain_[s] += chData[s];
        }
        float normFactor = 1.0f / static_cast<float>(totalNumInputChannels);
        for (int s = 0; s < numSamples; ++s)
            monoSidechain_[s] *= normFactor;
    }

    // 1. Run Spectral VAD to get per-sample confidence
    spectralVAD_.processBlock(monoSidechain_.data(), vadConfidence_.data(), numSamples);

    // 2. Process each channel through expander
    for (int ch = 0; ch < totalNumInputChannels; ++ch)
    {
        float* channelData = buffer.getWritePointer(ch);
        expander_.processBlock(channelData, vadConfidence_.data(), numSamples);
    }

    // 3. Apply output gain
    if (std::abs(currentOutputGain) > 0.01f)
    {
        float gainLinear = std::pow(10.0f, currentOutputGain / 20.0f);
        for (int ch = 0; ch < totalNumInputChannels; ++ch)
        {
            float* channelData = buffer.getWritePointer(ch);
            for (int s = 0; s < numSamples; ++s)
                channelData[s] *= gainLinear;
        }
    }

    // Apply wet/dry mix
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

    // Calculate output level for metering
    float peakLevel = 0.0f;
    for (int ch = 0; ch < totalNumInputChannels; ++ch)
    {
        const float* data = buffer.getReadPointer(ch);
        for (int s = 0; s < numSamples; ++s)
        {
            float absVal = std::abs(data[s]);
            if (absVal > peakLevel)
                peakLevel = absVal;
        }
    }
    float levelDb = 20.0f * std::log10(peakLevel + 1e-10f);
    outputLevelDb_.store(std::max(-60.0f, levelDb));
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

bool DeBleedAudioProcessor::loadV2Params(const juce::String& jsonPath)
{
    // Load learned band parameters from JSON exported by trainer_v2.py
    juce::File jsonFile(jsonPath);
    if (!jsonFile.existsAsFile())
    {
        DBG("V2 params file not found: " << jsonPath);
        return false;
    }

    juce::String jsonContent = jsonFile.loadFileAsString();
    auto json = juce::JSON::parse(jsonContent);

    if (json.isVoid())
    {
        DBG("Failed to parse V2 params JSON");
        return false;
    }

    // Parse dynamic EQ bands
    auto* bandsArray = json.getProperty("dynamic_eq_bands", juce::var()).getArray();
    if (bandsArray != nullptr)
    {
        std::array<DynamicEQ::BandParams, DynamicEQ::NUM_BANDS> eqParams;

        for (int i = 0; i < std::min(static_cast<int>(bandsArray->size()), DynamicEQ::NUM_BANDS); ++i)
        {
            auto band = (*bandsArray)[i];
            eqParams[i].freqHz = static_cast<float>(band.getProperty("freq_hz", 1000.0));
            eqParams[i].q = static_cast<float>(band.getProperty("q", 1.0));
            eqParams[i].maxCutDb = static_cast<float>(band.getProperty("max_cut_db", -6.0));
        }

        dynamicEQ_.loadParams(eqParams);
        DBG("Loaded " << bandsArray->size() << " dynamic EQ bands from V2 params");
    }

    // Parse Spectral VAD parameters
    auto spectralVadParams = json.getProperty("spectral_vad", juce::var());
    if (!spectralVadParams.isVoid())
    {
        // Load threshold and knee
        float thresholdDb = static_cast<float>(spectralVadParams.getProperty("threshold_db", -35.0));
        float kneeDb = static_cast<float>(spectralVadParams.getProperty("knee_db", 15.0));
        spectralVAD_.setThresholdDb(thresholdDb);
        spectralVAD_.setKneeDb(kneeDb);

        // Load band frequencies
        auto* freqArray = spectralVadParams.getProperty("band_frequencies", juce::var()).getArray();
        if (freqArray != nullptr && freqArray->size() == SpectralVAD::NUM_BANDS)
        {
            std::array<float, SpectralVAD::NUM_BANDS> frequencies;
            for (int i = 0; i < SpectralVAD::NUM_BANDS; ++i)
                frequencies[i] = static_cast<float>((*freqArray)[i]);
            spectralVAD_.setBandFrequencies(frequencies);
        }

        // Load learned band weights
        auto* weightsArray = spectralVadParams.getProperty("band_weights", juce::var()).getArray();
        if (weightsArray != nullptr && weightsArray->size() == SpectralVAD::NUM_BANDS)
        {
            std::array<float, SpectralVAD::NUM_BANDS> weights;
            for (int i = 0; i < SpectralVAD::NUM_BANDS; ++i)
                weights[i] = static_cast<float>((*weightsArray)[i]);
            spectralVAD_.setBandWeights(weights);

            DBG("Loaded Spectral VAD weights:");
            for (int i = 0; i < SpectralVAD::NUM_BANDS; ++i)
                DBG("  Band " << i << " (" << static_cast<float>((*freqArray)[i]) << "Hz): weight=" << weights[i]);
        }

        DBG("Loaded Spectral VAD params: threshold=" << thresholdDb << "dB, knee=" << kneeDb << "dB");
    }

    DBG("V2 params loaded successfully from: " << jsonPath);
    return true;
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
