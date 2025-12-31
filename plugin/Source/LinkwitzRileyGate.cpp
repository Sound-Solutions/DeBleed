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
    // NEURAL EXPANDER: Uses neural confidence to drive expansion
    // - High confidence (singer present): unity gain
    // - Low confidence (bleed): proportional reduction

    // Timing coefficients
    float attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * attackMs / 1000.0f));
    float releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * releaseMs / 1000.0f));

    // Floor gain (maximum attenuation)
    float floorGain = juce::Decibels::decibelsToGain(rangeDb);

    // Smooth the neural confidence
    float confCoeff = (neuralConfidence < smoothedConfidence) ? 0.3f : 0.1f;
    smoothedConfidence += confCoeff * (neuralConfidence - smoothedConfidence);

    // Map threshold from dB to confidence (0-1)
    // thresholdDb = 0   → confidenceThreshold = 1.0 (need very high confidence to open)
    // thresholdDb = -60 → confidenceThreshold = 0.0 (anything opens it)
    // This gives intuitive control: lower threshold = easier to open
    float confidenceThreshold = 1.0f + (thresholdDb / 60.0f);
    confidenceThreshold = juce::jlimit(0.0f, 1.0f, confidenceThreshold);

    // Calculate target gain based on neural confidence
    if (smoothedConfidence >= confidenceThreshold)
    {
        // High confidence - singer is present, unity gain
        targetGain = 1.0f;
        isGating = false;
    }
    else
    {
        // Low confidence - probably bleed, apply expansion
        // How far below threshold (0-1 range)
        float belowThreshold = confidenceThreshold - smoothedConfidence;

        // Progressive ratio: gentle near threshold, harder as confidence drops
        float ratio = 2.0f + (belowThreshold * 6.0f);  // 2:1 to 8:1
        ratio = std::min(ratio, 8.0f);

        // Convert to dB reduction
        // belowThreshold of 0.5 with ratio 4:1 → significant reduction
        float reductionDb = (belowThreshold * 60.0f) * (1.0f - 1.0f / ratio);

        // Limit to range
        reductionDb = std::min(reductionDb, -rangeDb);

        targetGain = juce::Decibels::decibelsToGain(-reductionDb);
        targetGain = std::max(targetGain, floorGain);
        isGating = true;
    }

    // Smooth gain changes (attack for reduction, release for recovery)
    float coeff = (targetGain < currentGain) ? attackCoeff : releaseCoeff;

    for (int s = 0; s < numSamples; ++s)
    {
        currentGain = currentGain * coeff + targetGain * (1.0f - coeff);
    }

    currentGain = juce::jlimit(floorGain, 1.0f, currentGain);

    // Store level for visualization (still useful to see input level)
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
