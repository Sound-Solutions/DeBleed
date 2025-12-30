#include "LinkwitzRileyGate.h"

LinkwitzRileyGate::LinkwitzRileyGate()
{
    // Initialize band buffers and envelopes
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        gateEnvelopes[b].currentGain = 1.0f;
        gateEnvelopes[b].targetGain = 1.0f;
        gateEnvelopes[b].holdCounter = 0;
        gateEnvelopes[b].isGating = false;
        gateEnvelopes[b].lastMaskAverage = 1.0f;
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
            // Q for Butterworth = 0.7071 (1/sqrt(2))
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

    // Start with full signal going into band splitting
    // We process crossovers sequentially: each crossover splits off the high band
    // remaining low content goes to next crossover

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

                // Low goes to current band, high goes to next iteration
                bandBuffers[x].setSample(ch, s, lp);
                sample = hp;  // Continue splitting the high portion
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

// Band scaling: user ms values are the "mid band" reference (index 3)
// Other bands scale proportionally based on AUTO_BAND_TIMING ratios
float LinkwitzRileyGate::getBandAttackMs(int band) const
{
    float scale = AUTO_BAND_TIMING[band].attackMs / AUTO_BAND_TIMING[MID_BAND_INDEX].attackMs;
    return attackMs * scale;
}

float LinkwitzRileyGate::getBandReleaseMs(int band) const
{
    float scale = AUTO_BAND_TIMING[band].releaseMs / AUTO_BAND_TIMING[MID_BAND_INDEX].releaseMs;
    return releaseMs * scale;
}

float LinkwitzRileyGate::getBandHoldMs(int band) const
{
    float scale = AUTO_BAND_TIMING[band].holdMs / AUTO_BAND_TIMING[MID_BAND_INDEX].holdMs;
    return holdMs * scale;
}

void LinkwitzRileyGate::processGateEnvelope(int band, float maskAverage, int numSamples)
{
    auto& env = gateEnvelopes[band];

    // Store for visualization
    env.lastMaskAverage = maskAverage;

    // Calculate threshold with hysteresis to prevent chattering
    // Base threshold: 0.6 = gate triggers when neural network detects 40%+ bleed
    // sensitivity > 0 = lower threshold = more aggressive gating
    // sensitivity < 0 = higher threshold = less aggressive gating
    float thresholdOffset = sensitivity / 100.0f * 0.3f;  // +/-30% offset at full sensitivity
    float openThreshold = 0.7f - thresholdOffset;   // Threshold to open gate (pass signal)
    float closeThreshold = 0.5f - thresholdOffset;  // Threshold to close gate (attenuate)

    // Hysteresis: use different thresholds for opening vs closing
    float threshold = env.isGating ? openThreshold : closeThreshold;

    // Gate closes when mask indicates bleed (low values = bleed detected)
    bool shouldGate = maskAverage < threshold;

    // Get band-scaled timing (user ms values scaled per band)
    float bandAttackMs = getBandAttackMs(band);
    float bandReleaseMs = getBandReleaseMs(band);
    float bandHoldMs = getBandHoldMs(band);

    // Convert to coefficients
    float attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * bandAttackMs / 1000.0f));
    float releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * bandReleaseMs / 1000.0f));
    int holdSamples = static_cast<int>(bandHoldMs * sampleRate / 1000.0f);

    // Floor gain - the maximum attenuation when gated
    float floorGain = juce::Decibels::decibelsToGain(floorDb);

    // Scale target gain proportionally to mask depth (not just binary on/off)
    // This creates smoother, more natural gating behavior
    // When mask=0.5 and threshold=0.6: depth = (0.6-0.5)/(0.6-0) = 0.17 → slight gating
    // When mask=0.1 and threshold=0.6: depth = (0.6-0.1)/(0.6-0) = 0.83 → heavy gating
    float gateDepth = 0.0f;
    if (shouldGate)
    {
        // Calculate how far below threshold we are (0-1 range)
        float closeThresholdLocal = closeThreshold;
        gateDepth = juce::jlimit(0.0f, 1.0f, (closeThresholdLocal - maskAverage) / closeThresholdLocal);
    }

    // Gate state machine with hold time
    if (shouldGate)
    {
        env.isGating = true;
        env.holdCounter = holdSamples;  // Reset hold counter while gating

        // Target gain interpolates between unity and floor based on depth
        env.targetGain = 1.0f - gateDepth * (1.0f - floorGain);
    }
    else
    {
        if (env.isGating)
        {
            // Hold phase - stay at current attenuation
            if (env.holdCounter > 0)
            {
                env.holdCounter -= numSamples;
                // During hold, maintain current target (don't jump to unity)
            }
            else
            {
                // Hold finished - start opening
                env.isGating = false;
                env.targetGain = 1.0f;
            }
        }
        else
        {
            env.targetGain = 1.0f;
        }
    }

    // Smooth the gain
    // Gate closing (gain decreasing) = attack time
    // Gate opening (gain increasing) = release time
    float coeff = (env.targetGain < env.currentGain) ? attackCoeff : releaseCoeff;

    // Apply smoothing over the block
    for (int s = 0; s < numSamples; ++s)
    {
        env.currentGain = env.currentGain * coeff + env.targetGain * (1.0f - coeff);
    }

    env.currentGain = juce::jlimit(floorGain, 1.0f, env.currentGain);
}

void LinkwitzRileyGate::process(juce::AudioBuffer<float>& buffer, const std::array<float, NUM_BANDS>& bandMaskAverages)
{
    if (!enabled)
        return;

    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // Split into bands
    splitBands(buffer);

    // Process gate envelope for each band
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        processGateEnvelope(b, bandMaskAverages[b], numSamples);
    }

    // Apply gain to each band
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        float gain = gateEnvelopes[b].currentGain;

        for (int ch = 0; ch < numChannels; ++ch)
        {
            bandBuffers[b].applyGain(ch, 0, numSamples, gain);
        }
    }

    // Recombine bands
    recombineBands(buffer);
}

std::array<LinkwitzRileyGate::BandState, LinkwitzRileyGate::NUM_BANDS> LinkwitzRileyGate::getBandStates() const
{
    std::array<BandState, NUM_BANDS> states;

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        states[b].currentGain = gateEnvelopes[b].currentGain;
        states[b].gateOpen = !gateEnvelopes[b].isGating;
        states[b].maskAverage = gateEnvelopes[b].lastMaskAverage;
    }

    return states;
}
