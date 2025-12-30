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

void LinkwitzRileyGate::processGateEnvelope(int band, float maskAverage, int numSamples)
{
    auto& env = gateEnvelopes[band];
    const auto& timing = AUTO_BAND_TIMING[band];

    // Store for visualization
    env.lastMaskAverage = maskAverage;

    // Calculate threshold from mask average
    // sensitivity > 0 = lower threshold = more gating
    // sensitivity < 0 = higher threshold = less gating
    // Threshold in linear terms: if mask < threshold, gate closes
    float thresholdOffset = sensitivity / 100.0f * 0.5f;  // +/-50% offset at full sensitivity
    float threshold = 0.8f - thresholdOffset;  // Base threshold 0.8, adjustable

    // Determine if we should be gating
    bool shouldGate = maskAverage < threshold;

    // Calculate timing with user multipliers
    float attackMs = timing.attackMs * attackMult;
    float releaseMs = timing.releaseMs * releaseMult;
    float holdMs = timing.holdMs * holdMult;

    // Convert to coefficients
    float attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * attackMs / 1000.0f));
    float releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * releaseMs / 1000.0f));
    int holdSamples = static_cast<int>(holdMs * sampleRate / 1000.0f);

    // Floor gain
    float floorGain = juce::Decibels::decibelsToGain(floorDb);

    // Gate state machine
    if (shouldGate)
    {
        if (!env.isGating)
        {
            env.isGating = true;
            env.holdCounter = holdSamples;
        }

        // Target is the floor
        env.targetGain = floorGain;
    }
    else
    {
        if (env.isGating)
        {
            // Hold phase - stay closed
            if (env.holdCounter > 0)
            {
                env.holdCounter -= numSamples;
                env.targetGain = floorGain;
            }
            else
            {
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
    // Gate closing (gain decreasing) = attack
    // Gate opening (gain increasing) = release
    float coeff = (env.targetGain < env.currentGain) ? attackCoeff : releaseCoeff;

    // Apply smoothing over the block
    for (int s = 0; s < numSamples; ++s)
    {
        env.currentGain = env.currentGain * coeff + env.targetGain * (1.0f - coeff);
    }

    env.currentGain = juce::jlimit(0.001f, 1.0f, env.currentGain);
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
