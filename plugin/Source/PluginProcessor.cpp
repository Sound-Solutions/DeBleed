#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <algorithm>

// Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_STRENGTH = "strength";
const juce::String DeBleedAudioProcessor::PARAM_MIX = "mix";
const juce::String DeBleedAudioProcessor::PARAM_BYPASS = "bypass";
const juce::String DeBleedAudioProcessor::PARAM_LOW_LATENCY = "lowLatency";

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

    // Initialize atomic values
    strength.store(*parameters.getRawParameterValue(PARAM_STRENGTH));
    mix.store(*parameters.getRawParameterValue(PARAM_MIX));
    bypassed.store(*parameters.getRawParameterValue(PARAM_BYPASS) > 0.5f);
    lowLatency.store(*parameters.getRawParameterValue(PARAM_LOW_LATENCY) > 0.5f);
}

DeBleedAudioProcessor::~DeBleedAudioProcessor()
{
    parameters.removeParameterListener(PARAM_STRENGTH, this);
    parameters.removeParameterListener(PARAM_MIX, this);
    parameters.removeParameterListener(PARAM_BYPASS, this);
    parameters.removeParameterListener(PARAM_LOW_LATENCY, this);
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
}

void DeBleedAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;

    // Check if we need to resample
    needsResampling = (std::abs(sampleRate - TARGET_SAMPLE_RATE) > 1.0);
    resampleRatio = TARGET_SAMPLE_RATE / sampleRate;

    // Calculate resampled block size
    int resampledBlockSize = needsResampling
        ? static_cast<int>(std::ceil(samplesPerBlock * resampleRatio)) + 16
        : samplesPerBlock;

    // Set STFT mode based on low latency parameter
    stftProcessor.setMode(lowLatency.load() ? STFTProcessor::Mode::LowLatency : STFTProcessor::Mode::Quality);

    // Prepare STFT processor at target sample rate
    stftProcessor.prepare(TARGET_SAMPLE_RATE, resampledBlockSize);

    // Prepare neural engine with max frames we might need
    int hopLength = stftProcessor.getHopLength();
    int maxFrames = (resampledBlockSize / hopLength) + 8;
    neuralEngine.prepare(maxFrames);

    // Allocate buffers
    processBuffer.resize(resampledBlockSize * 2, 0.0f);
    maskBuffer.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);

    // Allocate transpose buffers for neural network data layout conversion
    transposedMagnitude.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 0.0f);
    transposedMask.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);

    // Allocate smoothed mask buffer (one value per frequency bin, persists between blocks)
    if (smoothedMask.size() != STFTProcessor::N_FREQ_BINS)
    {
        smoothedMask.resize(STFTProcessor::N_FREQ_BINS, 1.0f);  // Initialize to pass-through
    }

    // Allocate resampling buffers and history
    if (needsResampling)
    {
        resampledInput.resize(resampledBlockSize + 64, 0.0f);
        resampledOutput.resize(samplesPerBlock + 64, 0.0f);

        // Initialize history buffers for resampler continuity
        inputHistory.resize(RESAMPLER_HISTORY_SIZE, 0.0f);
        outputHistory.resize(RESAMPLER_HISTORY_SIZE, 0.0f);

        inputResampler.reset();
        outputResampler.reset();
    }

    // Set latency (account for resampling)
    int latencySamples = stftProcessor.getLatencySamples();
    if (needsResampling)
    {
        latencySamples = static_cast<int>(latencySamples / resampleRatio);
    }
    setLatencySamples(latencySamples);
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

    // Check if we need to reinitialize due to mode change
    if (needsReinit.exchange(false))
    {
        // Reinitialize with new mode
        stftProcessor.setMode(lowLatency.load() ? STFTProcessor::Mode::LowLatency : STFTProcessor::Mode::Quality);
        stftProcessor.prepare(TARGET_SAMPLE_RATE, currentBlockSize);
        stftProcessor.reset();

        // Update latency
        int latencySamples = stftProcessor.getLatencySamples();
        if (needsResampling)
        {
            latencySamples = static_cast<int>(latencySamples / resampleRatio);
        }
        setLatencySamples(latencySamples);
    }

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

    // Early bypass if mix is 0% - skip all STFT processing
    if (currentMix < 0.001f)
        return;

    const float* inputData = buffer.getReadPointer(0);

    // Keep a copy of the dry signal
    std::vector<float> drySignal(inputData, inputData + numSamples);

    // Prepare data for processing (resample if needed)
    const float* processInput = inputData;
    int processNumSamples = numSamples;

    if (needsResampling)
    {
        // Downsample input to 48kHz with history for continuity
        processNumSamples = static_cast<int>(numSamples * resampleRatio);
        resampledInput.resize(processNumSamples + 16);

        // Create temporary buffer with history prepended for resampler look-back
        std::vector<float> inputWithHistory(RESAMPLER_HISTORY_SIZE + numSamples);
        std::memcpy(inputWithHistory.data(), inputHistory.data(), RESAMPLER_HISTORY_SIZE * sizeof(float));
        std::memcpy(inputWithHistory.data() + RESAMPLER_HISTORY_SIZE, inputData, numSamples * sizeof(float));

        // Process starting after history (resampler can look back into history)
        inputResampler.process(1.0 / resampleRatio,
                               inputWithHistory.data() + RESAMPLER_HISTORY_SIZE,
                               resampledInput.data(),
                               processNumSamples);

        // Update history with last samples from current input
        int historyStart = numSamples - RESAMPLER_HISTORY_SIZE;
        if (historyStart >= 0)
            std::memcpy(inputHistory.data(), inputData + historyStart, RESAMPLER_HISTORY_SIZE * sizeof(float));

        processInput = resampledInput.data();
    }

    // Process through STFT
    int numFrames = stftProcessor.processBlock(processInput, processNumSamples);

    if (numFrames > 0)
    {
        // Get magnitude data from STFT
        const float* magnitude = stftProcessor.getMagnitudeData();

        // Run neural network inference
        const float* nnMask = neuralEngine.process(magnitude, numFrames);

        // Clamp mask to [0, 1] range, check for NaN/Inf, and apply temporal smoothing
        // Smoothing prevents clicks by making mask changes gradual (like Waves PSE / Shure 5045)
        for (int frame = 0; frame < numFrames; ++frame)
        {
            for (int bin = 0; bin < STFTProcessor::N_FREQ_BINS; ++bin)
            {
                int idx = frame * STFTProcessor::N_FREQ_BINS + bin;
                float maskVal = nnMask[idx];

                // Safety: handle NaN/Inf and clamp to valid range
                if (std::isnan(maskVal) || std::isinf(maskVal))
                    maskVal = 1.0f;
                else
                    maskVal = std::clamp(maskVal, 0.0f, 1.0f);

                // Apply asymmetric smoothing (fast attack, slow release)
                float smoothingCoeff = (maskVal > smoothedMask[bin])
                    ? MASK_SMOOTHING_ATTACK   // Opening up (attack)
                    : MASK_SMOOTHING_RELEASE; // Closing down (release)

                smoothedMask[bin] += smoothingCoeff * (maskVal - smoothedMask[bin]);
                transposedMask[idx] = smoothedMask[bin];
            }
        }

        const float* mask = transposedMask.data();

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

        if (needsResampling)
        {
            // Reconstruct at 48kHz
            std::vector<float> processedAt48k(processNumSamples);
            stftProcessor.applyMaskAndReconstruct(mask, processedAt48k.data(), processNumSamples);

            // Upsample back to DAW rate with history for continuity
            std::vector<float> outputWithHistory(RESAMPLER_HISTORY_SIZE + processNumSamples);
            std::memcpy(outputWithHistory.data(), outputHistory.data(), RESAMPLER_HISTORY_SIZE * sizeof(float));
            std::memcpy(outputWithHistory.data() + RESAMPLER_HISTORY_SIZE, processedAt48k.data(), processNumSamples * sizeof(float));

            outputResampler.process(resampleRatio,
                                    outputWithHistory.data() + RESAMPLER_HISTORY_SIZE,
                                    outputData,
                                    numSamples);

            // Update output history
            int historyStart = processNumSamples - RESAMPLER_HISTORY_SIZE;
            if (historyStart >= 0)
                std::memcpy(outputHistory.data(), processedAt48k.data() + historyStart, RESAMPLER_HISTORY_SIZE * sizeof(float));
        }
        else
        {
            stftProcessor.applyMaskAndReconstruct(mask, outputData, numSamples);
        }

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

        // SAFETY: Final hard limit to prevent any extreme output
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
    else
    {
        // No complete frames yet, but still read any pending overlap-add output
        float* outputData = buffer.getWritePointer(0);

        if (needsResampling)
        {
            std::vector<float> processedAt48k(processNumSamples);
            stftProcessor.readOutput(processedAt48k.data(), processNumSamples);

            // Upsample with history for continuity
            std::vector<float> outputWithHistory(RESAMPLER_HISTORY_SIZE + processNumSamples);
            std::memcpy(outputWithHistory.data(), outputHistory.data(), RESAMPLER_HISTORY_SIZE * sizeof(float));
            std::memcpy(outputWithHistory.data() + RESAMPLER_HISTORY_SIZE, processedAt48k.data(), processNumSamples * sizeof(float));

            outputResampler.process(resampleRatio,
                                    outputWithHistory.data() + RESAMPLER_HISTORY_SIZE,
                                    outputData,
                                    numSamples);

            // Update output history
            int historyStart = processNumSamples - RESAMPLER_HISTORY_SIZE;
            if (historyStart >= 0)
                std::memcpy(outputHistory.data(), processedAt48k.data() + historyStart, RESAMPLER_HISTORY_SIZE * sizeof(float));
        }
        else
        {
            stftProcessor.readOutput(outputData, numSamples);
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
