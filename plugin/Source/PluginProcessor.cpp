#include "PluginProcessor.h"
#include "PluginEditor.h"

// Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_STRENGTH = "strength";
const juce::String DeBleedAudioProcessor::PARAM_MIX = "mix";
const juce::String DeBleedAudioProcessor::PARAM_BYPASS = "bypass";

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

    // Initialize atomic values
    strength.store(*parameters.getRawParameterValue(PARAM_STRENGTH));
    mix.store(*parameters.getRawParameterValue(PARAM_MIX));
    bypassed.store(*parameters.getRawParameterValue(PARAM_BYPASS) > 0.5f);
}

DeBleedAudioProcessor::~DeBleedAudioProcessor()
{
    parameters.removeParameterListener(PARAM_STRENGTH, this);
    parameters.removeParameterListener(PARAM_MIX, this);
    parameters.removeParameterListener(PARAM_BYPASS, this);
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
}

void DeBleedAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;

    // Prepare STFT processor
    stftProcessor.prepare(sampleRate, samplesPerBlock);

    // Prepare neural engine with max frames we might need
    // For a block of 512 samples with hop of 128: ~4 frames
    int maxFrames = (samplesPerBlock / STFTProcessor::HOP_LENGTH) + 8;
    neuralEngine.prepare(maxFrames);

    // Allocate buffers
    processBuffer.resize(samplesPerBlock * 2, 0.0f);
    maskBuffer.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);

    // Set latency
    setLatencySamples(stftProcessor.getLatencySamples());
}

void DeBleedAudioProcessor::releaseResources()
{
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

    // Check if model is loaded
    if (!neuralEngine.isModelLoaded())
        return;  // Pass through

    // Get parameter values
    float currentStrength = strength.load();
    float currentMix = mix.load();

    // Process each channel (typically mono processing, mix later)
    // For simplicity, we'll process the first channel and apply to all
    // A more sophisticated implementation would process channels independently

    const float* inputData = buffer.getReadPointer(0);

    // Keep a copy of the dry signal
    std::vector<float> drySignal(inputData, inputData + numSamples);

    // Process through STFT
    int numFrames = stftProcessor.processBlock(inputData, numSamples);

    if (numFrames > 0)
    {
        // Get magnitude data
        const float* magnitude = stftProcessor.getMagnitudeData();

        // Run neural network inference
        const float* mask = neuralEngine.process(magnitude, numFrames);

        // Apply strength parameter to mask
        if (currentStrength < 1.0f)
        {
            int numElements = numFrames * STFTProcessor::N_FREQ_BINS;
            maskBuffer.resize(numElements);

            for (int i = 0; i < numElements; ++i)
            {
                // Blend mask towards 1.0 (pass-through) based on strength
                maskBuffer[i] = mask[i] * currentStrength + (1.0f - currentStrength);
            }
            mask = maskBuffer.data();
        }

        // Reconstruct audio with mask applied
        float* outputData = buffer.getWritePointer(0);
        stftProcessor.applyMaskAndReconstruct(mask, outputData, numSamples);

        // Apply dry/wet mix
        if (currentMix < 1.0f)
        {
            float wet = currentMix;
            float dry = 1.0f - currentMix;

            for (int i = 0; i < numSamples; ++i)
            {
                outputData[i] = outputData[i] * wet + drySignal[i] * dry;
            }
        }

        // Copy to other channels if stereo
        for (int channel = 1; channel < totalNumOutputChannels; ++channel)
        {
            buffer.copyFrom(channel, 0, outputData, numSamples);
        }
    }
}

bool DeBleedAudioProcessor::loadModel(const juce::String& modelPath)
{
    return neuralEngine.loadModel(modelPath);
}

void DeBleedAudioProcessor::unloadModel()
{
    neuralEngine.unloadModel();
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

// Plugin instantiation
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeBleedAudioProcessor();
}
