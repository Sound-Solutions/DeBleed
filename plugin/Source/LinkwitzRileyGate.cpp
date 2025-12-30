#include "LinkwitzRileyGate.h"

LinkwitzRileyGate::LinkwitzRileyGate()
{
    // Initialize with default per-band timing (scaled from sub to high)
    static const std::array<float, NUM_BANDS> defaultAttacks = {{35.0f, 20.0f, 12.0f, 6.0f, 3.0f, 1.5f}};
    static const std::array<float, NUM_BANDS> defaultReleases = {{350.0f, 225.0f, 150.0f, 100.0f, 65.0f, 50.0f}};
    static const std::array<float, NUM_BANDS> defaultHolds = {{75.0f, 55.0f, 35.0f, 20.0f, 12.0f, 8.0f}};

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        bandParams[b].thresholdDb = -40.0f;
        bandParams[b].attackMs = defaultAttacks[b];
        bandParams[b].releaseMs = defaultReleases[b];
        bandParams[b].holdMs = defaultHolds[b];
        bandParams[b].rangeDb = -60.0f;
        bandParams[b].enabled = true;

        gateEnvelopes[b].currentGain = 1.0f;
        gateEnvelopes[b].targetGain = 1.0f;
        gateEnvelopes[b].holdCounter = 0;
        gateEnvelopes[b].isGating = false;

        levelDetectors[b].rmsEnvelopeDb = -100.0f;
    }
}

void LinkwitzRileyGate::prepare(double newSampleRate, int maxBlockSize)
{
    sampleRate = newSampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maxBlockSize);
    spec.numChannels = 1;  // Process channels separately

    // Initialize crossover filters
    for (int x = 0; x < NUM_CROSSOVERS; ++x)
    {
        auto& xover = crossovers[x];
        for (int ch = 0; ch < 2; ++ch)
        {
            xover.lp1[ch].prepare(spec);
            xover.lp2[ch].prepare(spec);
            xover.hp1[ch].prepare(spec);
            xover.hp2[ch].prepare(spec);

            xover.lp1[ch].setType(juce::dsp::StateVariableTPTFilterType::lowpass);
            xover.lp2[ch].setType(juce::dsp::StateVariableTPTFilterType::lowpass);
            xover.hp1[ch].setType(juce::dsp::StateVariableTPTFilterType::highpass);
            xover.hp2[ch].setType(juce::dsp::StateVariableTPTFilterType::highpass);
        }
    }

    updateCrossoverCoefficients();

    // Allocate band buffers
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        bandBuffers[b].setSize(2, maxBlockSize);
        bandBuffers[b].clear();
    }

    // Initialize level detectors with ~5ms time constant
    float detectorTimeMs = 5.0f;
    float detectorCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * detectorTimeMs / 1000.0f));
    for (auto& detector : levelDetectors)
    {
        detector.detectorCoeff = detectorCoeff;
        detector.rmsEnvelopeDb = -100.0f;
    }

    // Reset envelopes
    for (auto& env : gateEnvelopes)
    {
        env.currentGain = 1.0f;
        env.targetGain = 1.0f;
        env.holdCounter = 0;
        env.isGating = false;
    }
}

void LinkwitzRileyGate::reset()
{
    for (int x = 0; x < NUM_CROSSOVERS; ++x)
    {
        auto& xover = crossovers[x];
        for (int ch = 0; ch < 2; ++ch)
        {
            xover.lp1[ch].reset();
            xover.lp2[ch].reset();
            xover.hp1[ch].reset();
            xover.hp2[ch].reset();
        }
    }

    for (auto& env : gateEnvelopes)
    {
        env.currentGain = 1.0f;
        env.targetGain = 1.0f;
        env.holdCounter = 0;
        env.isGating = false;
    }

    for (auto& detector : levelDetectors)
    {
        detector.rmsEnvelopeDb = -100.0f;
    }

    for (auto& buf : bandBuffers)
    {
        buf.clear();
    }
}

void LinkwitzRileyGate::updateCrossoverCoefficients()
{
    for (int x = 0; x < NUM_CROSSOVERS; ++x)
    {
        float freq = crossoverFreqs[x];
        auto& xover = crossovers[x];

        for (int ch = 0; ch < 2; ++ch)
        {
            // LR-4 = two cascaded Butterworth at same frequency
            xover.lp1[ch].setCutoffFrequency(freq);
            xover.lp2[ch].setCutoffFrequency(freq);
            xover.hp1[ch].setCutoffFrequency(freq);
            xover.hp2[ch].setCutoffFrequency(freq);
        }
    }
}

void LinkwitzRileyGate::splitBands(const juce::AudioBuffer<float>& input)
{
    const int numSamples = input.getNumSamples();
    const int numChannels = juce::jmin(input.getNumChannels(), 2);

    // Clear all band buffers
    for (auto& buf : bandBuffers)
    {
        buf.clear();
    }

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* inputData = input.getReadPointer(ch);

        // Process sample by sample through the crossover chain
        for (int s = 0; s < numSamples; ++s)
        {
            float sample = inputData[s];

            // Process through each crossover
            for (int x = 0; x < NUM_CROSSOVERS; ++x)
            {
                auto& xover = crossovers[x];

                // LR-4 lowpass: cascaded 2nd-order
                float lp = xover.lp1[ch].processSample(0, sample);
                lp = xover.lp2[ch].processSample(0, lp);

                // LR-4 highpass: cascaded 2nd-order
                float hp = xover.hp1[ch].processSample(0, sample);
                hp = xover.hp2[ch].processSample(0, hp);

                // Low goes to current band, high continues to next crossover
                bandBuffers[x].setSample(ch, s, lp);
                sample = hp;
            }

            // Whatever remains after all crossovers goes to the highest band
            bandBuffers[NUM_BANDS - 1].setSample(ch, s, sample);
        }
    }
}

void LinkwitzRileyGate::recombineBands(juce::AudioBuffer<float>& output)
{
    const int numSamples = output.getNumSamples();
    const int numChannels = juce::jmin(output.getNumChannels(), 2);

    // Sum all bands back together
    output.clear();

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        for (int ch = 0; ch < numChannels; ++ch)
        {
            output.addFrom(ch, 0, bandBuffers[b], ch, 0, numSamples);
        }
    }
}

float LinkwitzRileyGate::detectBandLevel(int band, int numSamples)
{
    auto& detector = levelDetectors[band];
    const auto& buffer = bandBuffers[band];
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // Calculate RMS over the block
    float sumSquares = 0.0f;
    int totalSamples = 0;

    for (int ch = 0; ch < numChannels; ++ch)
    {
        const float* data = buffer.getReadPointer(ch);
        for (int s = 0; s < numSamples; ++s)
        {
            sumSquares += data[s] * data[s];
            ++totalSamples;
        }
    }

    float rms = (totalSamples > 0) ? std::sqrt(sumSquares / static_cast<float>(totalSamples)) : 0.0f;
    float rmsDb = juce::Decibels::gainToDecibels(rms, -100.0f);

    // Smooth the RMS envelope (ballistics-style: fast attack, slow release for detection)
    float targetDb = rmsDb;
    float coeff = detector.detectorCoeff;

    // Use faster attack for level detection
    if (targetDb > detector.rmsEnvelopeDb)
    {
        coeff = 0.3f;  // Fast attack for level detection
    }

    detector.rmsEnvelopeDb = detector.rmsEnvelopeDb * coeff + targetDb * (1.0f - coeff);

    return detector.rmsEnvelopeDb;
}

void LinkwitzRileyGate::processGateEnvelope(int band, float signalLevelDb, int numSamples)
{
    auto& env = gateEnvelopes[band];
    const auto& params = bandParams[band];

    // Skip if band is disabled
    if (!params.enabled)
    {
        env.currentGain = 1.0f;
        env.targetGain = 1.0f;
        env.isGating = false;
        return;
    }

    // Hysteresis: 3dB difference between open and close thresholds
    float openThreshold = params.thresholdDb;
    float closeThreshold = params.thresholdDb - 3.0f;

    // Gate logic: signal above threshold = gate open, below = gate closing
    bool signalAboveThreshold = signalLevelDb > (env.isGating ? openThreshold : closeThreshold);

    // Convert timing to coefficients
    float attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * params.attackMs / 1000.0f));
    float releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * params.releaseMs / 1000.0f));
    int holdSamples = static_cast<int>(params.holdMs * sampleRate / 1000.0f);

    // Floor gain (maximum attenuation)
    float floorGain = juce::Decibels::decibelsToGain(params.rangeDb);

    // Gate state machine with hold time
    if (!signalAboveThreshold)
    {
        // Signal below threshold - close gate
        env.isGating = true;
        env.holdCounter = holdSamples;  // Reset hold counter while gating
        env.targetGain = floorGain;
    }
    else
    {
        // Signal above threshold
        if (env.isGating)
        {
            // Was gating - check hold time
            if (env.holdCounter > 0)
            {
                env.holdCounter -= numSamples;
                // Stay at current attenuation during hold
            }
            else
            {
                // Hold finished - open gate
                env.isGating = false;
                env.targetGain = 1.0f;
            }
        }
        else
        {
            // Not gating - stay open
            env.targetGain = 1.0f;
        }
    }

    // Apply smoothing (attack for closing, release for opening)
    // Note: Gate closing = gain decreasing = use attack
    //       Gate opening = gain increasing = use release
    float coeff = (env.targetGain < env.currentGain) ? attackCoeff : releaseCoeff;

    // Smooth over the block
    for (int s = 0; s < numSamples; ++s)
    {
        env.currentGain = env.currentGain * coeff + env.targetGain * (1.0f - coeff);
    }

    env.currentGain = juce::jlimit(floorGain, 1.0f, env.currentGain);
}

void LinkwitzRileyGate::process(juce::AudioBuffer<float>& buffer)
{
    if (!masterEnabled)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // Split into bands
    splitBands(buffer);

    // Process each band: detect level, compute gate envelope, apply gain
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        // Detect signal level in this band
        float signalLevelDb = detectBandLevel(b, numSamples);

        // Process gate envelope
        processGateEnvelope(b, signalLevelDb, numSamples);

        // Apply gain to band buffer
        float gain = gateEnvelopes[b].currentGain;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            bandBuffers[b].applyGain(ch, 0, numSamples, gain);
        }
    }

    // Recombine bands
    recombineBands(buffer);
}

// Per-band parameter setters
void LinkwitzRileyGate::setBandThreshold(int band, float db)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band].thresholdDb = juce::jlimit(-80.0f, 0.0f, db);
}

void LinkwitzRileyGate::setBandAttack(int band, float ms)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band].attackMs = juce::jlimit(0.1f, 100.0f, ms);
}

void LinkwitzRileyGate::setBandRelease(int band, float ms)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band].releaseMs = juce::jlimit(10.0f, 1000.0f, ms);
}

void LinkwitzRileyGate::setBandHold(int band, float ms)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band].holdMs = juce::jlimit(0.0f, 500.0f, ms);
}

void LinkwitzRileyGate::setBandRange(int band, float db)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band].rangeDb = juce::jlimit(-80.0f, 0.0f, db);
}

void LinkwitzRileyGate::setBandEnabled(int band, bool enabled)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band].enabled = enabled;
}

void LinkwitzRileyGate::setBandParams(int band, const BandParams& params)
{
    if (band >= 0 && band < NUM_BANDS)
        bandParams[band] = params;
}

void LinkwitzRileyGate::setCrossover(int index, float hz)
{
    if (index >= 0 && index < NUM_CROSSOVERS)
    {
        crossoverFreqs[index] = juce::jlimit(20.0f, 20000.0f, hz);
        updateCrossoverCoefficients();
    }
}

std::array<LinkwitzRileyGate::BandState, LinkwitzRileyGate::NUM_BANDS> LinkwitzRileyGate::getBandStates() const
{
    std::array<BandState, NUM_BANDS> states;

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        states[b].currentGain = gateEnvelopes[b].currentGain;
        states[b].gateOpen = !gateEnvelopes[b].isGating;
        states[b].signalLevelDb = levelDetectors[b].rmsEnvelopeDb;
        states[b].thresholdDb = bandParams[b].thresholdDb;
    }

    return states;
}
