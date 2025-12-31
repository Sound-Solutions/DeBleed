#include "LinkwitzRileyGate.h"

LinkwitzRileyGate::LinkwitzRileyGate()
{
}

void LinkwitzRileyGate::prepare(double newSampleRate, int maxBlockSize)
{
    sampleRate = newSampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maxBlockSize);
    spec.numChannels = 1;

    // Prepare sidechain filters
    hpFilter1.prepare(spec);
    hpFilter2.prepare(spec);
    lpFilter1.prepare(spec);
    lpFilter2.prepare(spec);

    hpFilter1.setType(juce::dsp::StateVariableTPTFilterType::highpass);
    hpFilter2.setType(juce::dsp::StateVariableTPTFilterType::highpass);
    lpFilter1.setType(juce::dsp::StateVariableTPTFilterType::lowpass);
    lpFilter2.setType(juce::dsp::StateVariableTPTFilterType::lowpass);

    updateFilters();

    // Level detector time constant (~10ms)
    detectorCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * 0.010f));

    // Allocate sidechain buffer
    sidechainBuffer.resize(maxBlockSize, 0.0f);

    reset();
}

void LinkwitzRileyGate::reset()
{
    hpFilter1.reset();
    hpFilter2.reset();
    lpFilter1.reset();
    lpFilter2.reset();

    currentGain = 1.0f;
    targetGain = 1.0f;
    holdCounter = 0;
    isGating = false;
    detectedLevelDb = -100.0f;
}

void LinkwitzRileyGate::updateFilters()
{
    hpFilter1.setCutoffFrequency(sidechainHPF);
    hpFilter2.setCutoffFrequency(sidechainHPF);
    lpFilter1.setCutoffFrequency(sidechainLPF);
    lpFilter2.setCutoffFrequency(sidechainLPF);
}

float LinkwitzRileyGate::detectLevel(const float* input, int numSamples)
{
    // Copy input to sidechain buffer and apply HPF/LPF
    for (int s = 0; s < numSamples; ++s)
    {
        float sample = input[s];

        // Cascaded HPF (24dB/oct)
        sample = hpFilter1.processSample(0, sample);
        sample = hpFilter2.processSample(0, sample);

        // Cascaded LPF (24dB/oct)
        sample = lpFilter1.processSample(0, sample);
        sample = lpFilter2.processSample(0, sample);

        sidechainBuffer[s] = sample;
    }

    // Calculate RMS of filtered sidechain
    float sumSquares = 0.0f;
    for (int s = 0; s < numSamples; ++s)
    {
        sumSquares += sidechainBuffer[s] * sidechainBuffer[s];
    }

    float rms = (numSamples > 0) ? std::sqrt(sumSquares / static_cast<float>(numSamples)) : 0.0f;
    float rmsDb = juce::Decibels::gainToDecibels(rms, -100.0f);

    // Smooth the level (fast attack, slower release for detection)
    float coeff = (rmsDb > detectedLevelDb) ? 0.3f : detectorCoeff;
    detectedLevelDb = detectedLevelDb * coeff + rmsDb * (1.0f - coeff);

    return detectedLevelDb;
}

void LinkwitzRileyGate::processEnvelope(float levelDb, int numSamples)
{
    // NEURAL EXPANDER: Uses neural confidence + threshold to drive expansion
    // - neuralConfidence = raw mask value (0.0 = bleed, 1.0 = singer)
    // - thresholdDb = user control (-80 to 0 dB, maps to confidence threshold)

    // Timing coefficients
    float attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * attackMs / 1000.0f));
    float releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * releaseMs / 1000.0f));

    // Floor gain (maximum attenuation)
    float floorGain = juce::Decibels::decibelsToGain(rangeDb);

    // Convert threshold dB to confidence threshold:
    // -80 dB → 0.0 (gate always open, no GR)
    //   0 dB → 1.0 (needs full confidence to open)
    float confThreshold = 1.0f + (thresholdDb / 80.0f);
    confThreshold = juce::jlimit(0.0f, 1.0f, confThreshold);

    // Smooth the neural confidence
    float confCoeff = (neuralConfidence < smoothedConfidence) ? 0.3f : 0.1f;
    smoothedConfidence += confCoeff * (neuralConfidence - smoothedConfidence);

    // Calculate desired gain based on threshold
    float desiredGain = 1.0f;  // Default: fully open

    if (confThreshold <= 0.01f)
    {
        // Threshold at minimum - gate always fully open, no GR
        desiredGain = 1.0f;
    }
    else if (confThreshold >= 0.99f)
    {
        // Threshold at maximum - only full confidence opens
        desiredGain = (smoothedConfidence >= 0.99f) ? 1.0f : floorGain;
    }
    else
    {
        // Normal operation: confidence below threshold = closed, above = open
        if (smoothedConfidence >= confThreshold)
        {
            // Above threshold - gate open
            desiredGain = 1.0f;
        }
        else
        {
            // Below threshold - gate closes proportionally
            float ratio = smoothedConfidence / confThreshold;
            desiredGain = floorGain + ratio * (1.0f - floorGain);
        }
    }

    // Hold logic: when gate wants to close, wait for hold time first
    int holdSamples = static_cast<int>(holdMs * sampleRate / 1000.0f);

    if (desiredGain >= 0.99f)
    {
        // Gate wants to open - reset hold counter, open immediately
        holdCounter = holdSamples;
        targetGain = desiredGain;
    }
    else if (holdCounter > 0)
    {
        // Gate wants to close but we're in hold period - stay open
        holdCounter -= numSamples;
        targetGain = 1.0f;  // Keep fully open during hold
    }
    else
    {
        // Hold expired, allow gate to close
        targetGain = desiredGain;
    }

    isGating = (targetGain < 0.99f);

    // Smooth gain changes (attack for reduction, release for recovery)
    float coeff = (targetGain < currentGain) ? attackCoeff : releaseCoeff;

    for (int s = 0; s < numSamples; ++s)
    {
        currentGain = currentGain * coeff + targetGain * (1.0f - coeff);
    }

    currentGain = juce::jlimit(floorGain, 1.0f, currentGain);

    // Store level for visualization
    detectedLevelDb = levelDb;
}

void LinkwitzRileyGate::process(juce::AudioBuffer<float>& buffer)
{
    if (!masterEnabled)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // Use first channel for detection
    const float* inputData = buffer.getReadPointer(0);

    // Detect level through sidechain filters
    float levelDb = detectLevel(inputData, numSamples);

    // Process gate envelope
    processEnvelope(levelDb, numSamples);

    // Apply gain to all channels
    for (int ch = 0; ch < numChannels; ++ch)
    {
        buffer.applyGain(ch, 0, numSamples, currentGain);
    }
}
