#include "IIRFilterBank.h"

IIRFilterBank::IIRFilterBank()
{
    targetGains.fill(1.0f);
    currentGains.fill(1.0f);
    qFactors.fill(8.0f);  // Will be recalculated
    calculateFrequencies();
}

void IIRFilterBank::calculateFrequencies()
{
    // Hybrid 32+160 topology:
    // - Array A: 32 bands from 20 Hz to 500 Hz (logarithmic) - bass
    // - Array B: 160 bands from 500 Hz to 20 kHz (logarithmic) - highs
    //
    // This concentrates resolution where surgical cymbal de-bleeding is needed (highs)
    // while maintaining adequate bass resolution for kick/snare separation.

    // Array A: 32 low bands (20 Hz to 500 Hz)
    for (int i = 0; i < NUM_LOW_BANDS; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(NUM_LOW_BANDS - 1);
        centerFreqs[i] = MIN_FREQ * std::pow(CROSSOVER_FREQ / MIN_FREQ, t);
    }

    // Array B: 160 high bands (500 Hz to 20 kHz)
    for (int i = 0; i < NUM_HIGH_BANDS; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(NUM_HIGH_BANDS - 1);
        centerFreqs[NUM_LOW_BANDS + i] = CROSSOVER_FREQ * std::pow(MAX_FREQ / CROSSOVER_FREQ, t);
    }

    // Calculate Q for each band based on neighbor spacing
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float bandwidth;

        if (i == 0)
        {
            bandwidth = centerFreqs[1] - centerFreqs[0];
        }
        else if (i == NUM_BANDS - 1)
        {
            bandwidth = centerFreqs[NUM_BANDS - 1] - centerFreqs[NUM_BANDS - 2];
        }
        else
        {
            float bwLower = centerFreqs[i] - centerFreqs[i - 1];
            float bwUpper = centerFreqs[i + 1] - centerFreqs[i];
            bandwidth = (bwLower + bwUpper) * 0.5f;
        }

        qFactors[i] = centerFreqs[i] / bandwidth;
        qFactors[i] = juce::jlimit(2.0f, 50.0f, qFactors[i]);
    }
}

float IIRFilterBank::getBandpassMagnitudeAt(int bandIndex, float freq) const
{
    // 2nd-order bandpass magnitude response (for normalization calculation)
    if (freq <= 0.0f || bandIndex < 0 || bandIndex >= NUM_LOW_BANDS)
        return 0.0f;

    float fc = centerFreqs[bandIndex];
    float Q = qFactors[bandIndex];
    float ratio = freq / fc;
    float term = ratio - (1.0f / ratio);
    float denominator = 1.0f + Q * Q * term * term;

    return 1.0f / std::sqrt(denominator);
}

void IIRFilterBank::calculateLowNormalization()
{
    // Calculate normalization for LOW ENGINE only (parallel bandpass)
    const float testFreqs[] = {25.0f, 40.0f, 63.0f, 100.0f, 160.0f, 250.0f, 400.0f};
    const int numTestFreqs = 7;

    float minResponse = 1e10f;

    for (int f = 0; f < numTestFreqs; ++f)
    {
        float totalResponse = 0.0f;
        for (int b = 0; b < NUM_LOW_BANDS; ++b)
        {
            totalResponse += getBandpassMagnitudeAt(b, testFreqs[f]);
        }
        if (totalResponse < minResponse)
            minResponse = totalResponse;
    }

    if (minResponse > 0.001f)
        lowNormalization = 1.0f / minResponse;
    else
        lowNormalization = 1.0f;

    // Makeup gain for tight Q not summing to unity
    lowNormalization *= 1.41f;  // +3dB
}

void IIRFilterBank::prepare(double newSampleRate, int maxBlockSize)
{
    sampleRate = newSampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maxBlockSize);
    spec.numChannels = 1;

    float nyquist = static_cast<float>(sampleRate) * 0.5f;

    // ═══════════════════════════════════════════════════════════════
    // LOW ENGINE: Initialize SVF Bandpass filters (bands 0-31)
    // Fixed freq/Q - only gain changes during playback
    // NOTE: SVF resonance is different from traditional Q!
    // - 0.707 (1/sqrt(2)) = Butterworth (no resonance)
    // - Higher values = more resonance (narrower bandwidth)
    // - Values above ~5 can cause ringing, above ~10 can self-oscillate
    // ═══════════════════════════════════════════════════════════════
    for (int i = 0; i < NUM_LOW_BANDS; ++i)
    {
        float freq = centerFreqs[i];
        if (freq >= nyquist * 0.95f)
            freq = nyquist * 0.95f;

        // Use moderate resonance for musical bandpass response
        // Too low = very wide bands, too high = ringing
        float svfResonance = juce::jlimit(1.0f, 2.0f, qFactors[i]);

        for (int ch = 0; ch < 2; ++ch)
        {
            lowBands[i].filter[ch].setType(juce::dsp::StateVariableTPTFilter<float>::Type::bandpass);
            lowBands[i].filter[ch].setCutoffFrequency(freq);
            lowBands[i].filter[ch].setResonance(svfResonance);
            lowBands[i].filter[ch].prepare(spec);
        }

        // Initialize gain smoother (10ms ramp time)
        lowBands[i].gainSmoother.reset(sampleRate, SMOOTHING_TIME_SEC);
        lowBands[i].gainSmoother.setCurrentAndTargetValue(1.0f);
    }

    // Calculate normalization for low engine
    calculateLowNormalization();

    // ═══════════════════════════════════════════════════════════════
    // HIGH ENGINE: Initialize SVF Bandpass for Notch (bands 32-191)
    // Notch = Input - Bandpass × NotchDepth
    // Fixed freq/Q - only notchDepth changes during playback
    // NOTE: For notch filters, we want moderate Q for smooth cuts
    // Too high Q = ringing, too low Q = wide cuts affecting neighbors
    // ═══════════════════════════════════════════════════════════════
    for (int i = 0; i < NUM_HIGH_BANDS; ++i)
    {
        int bandIdx = NUM_LOW_BANDS + i;
        float freq = centerFreqs[bandIdx];
        if (freq >= nyquist * 0.95f)
            freq = nyquist * 0.95f;

        // Use resonance = 1.0 for unity bandpass gain at center frequency
        // This makes the notch formula work without normalization
        float svfResonance = 1.0f;
        highBands[i].resonance = svfResonance;

        for (int ch = 0; ch < 2; ++ch)
        {
            highBands[i].filter[ch].setType(juce::dsp::StateVariableTPTFilter<float>::Type::bandpass);
            highBands[i].filter[ch].setCutoffFrequency(freq);
            highBands[i].filter[ch].setResonance(svfResonance);
            highBands[i].filter[ch].prepare(spec);
        }

        // Initialize notch depth smoother (10ms ramp time)
        // 0 = bypass (unity), 1 = full notch (cut)
        highBands[i].notchDepthSmoother.reset(sampleRate, SMOOTHING_TIME_SEC);
        highBands[i].notchDepthSmoother.setCurrentAndTargetValue(0.0f);
    }

    // ═══════════════════════════════════════════════════════════════
    // CROSSOVER: SVF TPT filters for seamless blending
    // ═══════════════════════════════════════════════════════════════
    for (int ch = 0; ch < 2; ++ch)
    {
        lowpassFilter[ch].setType(juce::dsp::StateVariableTPTFilter<float>::Type::lowpass);
        lowpassFilter[ch].setCutoffFrequency(CROSSOVER_FREQ);
        lowpassFilter[ch].setResonance(0.707f);  // Butterworth Q
        lowpassFilter[ch].prepare(spec);

        highpassFilter[ch].setType(juce::dsp::StateVariableTPTFilter<float>::Type::highpass);
        highpassFilter[ch].setCutoffFrequency(CROSSOVER_FREQ);
        highpassFilter[ch].setResonance(0.707f);
        highpassFilter[ch].prepare(spec);
    }

    reset();
}

void IIRFilterBank::reset()
{
    // Reset low engine
    for (int i = 0; i < NUM_LOW_BANDS; ++i)
    {
        lowBands[i].filter[0].reset();
        lowBands[i].filter[1].reset();
        lowBands[i].gainSmoother.setCurrentAndTargetValue(1.0f);
    }

    // Reset high engine
    for (int i = 0; i < NUM_HIGH_BANDS; ++i)
    {
        highBands[i].filter[0].reset();
        highBands[i].filter[1].reset();
        highBands[i].notchDepthSmoother.setCurrentAndTargetValue(0.0f);
    }

    // Reset crossover
    for (int ch = 0; ch < 2; ++ch)
    {
        lowpassFilter[ch].reset();
        highpassFilter[ch].reset();
    }

    // Reset gains
    targetGains.fill(1.0f);
    currentGains.fill(1.0f);
}

void IIRFilterBank::process(juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // ═══════════════════════════════════════════════════════════════
    // PRE-PROCESS: Update smoother targets (once per block)
    // ═══════════════════════════════════════════════════════════════

    // Low bands: gain target (0.0 = cut, 1.0 = unity)
    for (int b = 0; b < NUM_LOW_BANDS; ++b)
    {
        float gain = targetGains[b];

        // BLEND ZONE: Gradually reduce effect for bands near crossover (low side)
        // Bands 28-31 blend with high engine
        int distFromCrossover = NUM_LOW_BANDS - 1 - b;  // 0 for band 31, 3 for band 28
        if (distFromCrossover < BLEND_ZONE_BANDS)
        {
            // Check if corresponding high bands are cutting
            int highBandIdx = BLEND_ZONE_BANDS - 1 - distFromCrossover;  // 3,2,1,0
            if (highBandIdx < NUM_HIGH_BANDS)
            {
                float highCut = 1.0f - targetGains[NUM_LOW_BANDS + highBandIdx];
                float followFactor = static_cast<float>(BLEND_ZONE_BANDS - distFromCrossover) / static_cast<float>(BLEND_ZONE_BANDS);
                followFactor *= 0.5f;  // Max 50% follow
                if (highCut > 0.01f)
                {
                    gain = juce::jmax(gain - highCut * followFactor, 0.0f);
                }
            }
        }

        lowBands[b].gainSmoother.setTargetValue(juce::jlimit(0.0f, 1.0f, gain));
    }

    // High bands: notch depth target (0.0 = bypass, 1.0 = full cut)
    for (int b = 0; b < NUM_HIGH_BANDS; ++b)
    {
        int bandIdx = NUM_LOW_BANDS + b;
        float gain = juce::jlimit(0.0f, 1.0f, targetGains[bandIdx]);

        // BLEND ZONE: Gradually introduce notch effect for bands near crossover (high side)
        // Bands 32-35 (b = 0-3) gradually introduce the effect
        if (b < BLEND_ZONE_BANDS)
        {
            float blendFactor = static_cast<float>(b + 1) / static_cast<float>(BLEND_ZONE_BANDS);
            gain = 1.0f + (gain - 1.0f) * blendFactor;
        }

        // Convert gain (0-1) to notch depth (1-0)
        // gain = 1.0 → notchDepth = 0.0 (bypass)
        // gain = 0.0 → notchDepth = 1.0 (full cut)
        float notchDepth = 1.0f - gain;

        highBands[b].notchDepthSmoother.setTargetValue(notchDepth);
    }

    // ═══════════════════════════════════════════════════════════════
    // PROCESS: Per-sample with SmoothedValue interpolation
    // ═══════════════════════════════════════════════════════════════
    for (int ch = 0; ch < numChannels; ++ch)
    {
        float* data = buffer.getWritePointer(ch);

        for (int s = 0; s < numSamples; ++s)
        {
            float input = data[s];

            // ═══════════════════════════════════════════════════════════════
            // LOW ENGINE: Parallel SVF Bandpass with output gain scaling
            // Gain applied AFTER filter - inherently stable, no clicks!
            // ═══════════════════════════════════════════════════════════════
            float lowOutput = 0.0f;
            for (int b = 0; b < NUM_LOW_BANDS; ++b)
            {
                float gain = lowBands[b].gainSmoother.getNextValue();
                float bp = lowBands[b].filter[ch].processSample(0, input);
                lowOutput += bp * gain;

                // Update current gain for visualization
                currentGains[b] = gain;
            }
            lowOutput *= lowNormalization;

            // ═══════════════════════════════════════════════════════════════
            // HIGH ENGINE: Parallel SVF Subtraction (NOT series!)
            // Each bandpass receives the ORIGINAL input, not cascaded output
            // We subtract the bandpass content we want to remove
            // - notchDepth = 0 → subtract nothing (bypass)
            // - notchDepth = 1 → subtract full bandpass (notch at that freq)
            // IMPORTANT: Threshold small values to prevent cumulative over-cutting
            // ═══════════════════════════════════════════════════════════════
            float highOutput = input;
            for (int b = 0; b < NUM_HIGH_BANDS; ++b)
            {
                float notchDepth = highBands[b].notchDepthSmoother.getNextValue();

                // Process bandpass on ORIGINAL INPUT (parallel, not series!)
                float bp = highBands[b].filter[ch].processSample(0, input);

                // Only subtract when notchDepth is significant (> ~6dB cut threshold)
                // This prevents cumulative small cuts from adding up across 160 bands
                if (notchDepth > 0.5f)
                {
                    // Rescale: 0.5-1.0 → 0.0-1.0 for smoother transition
                    float scaledDepth = (notchDepth - 0.5f) * 2.0f;
                    highOutput -= bp * scaledDepth;
                }

                // Update current gain for visualization (1 - notchDepth)
                currentGains[NUM_LOW_BANDS + b] = 1.0f - notchDepth;
            }

            // ═══════════════════════════════════════════════════════════════
            // CROSSOVER: Combine low + high engines
            // ═══════════════════════════════════════════════════════════════
            float lowFiltered = lowpassFilter[ch].processSample(0, lowOutput);
            float highFiltered = highpassFilter[ch].processSample(0, highOutput);

            data[s] = lowFiltered + highFiltered;
        }
    }
}

void IIRFilterBank::setBandGain(int bandIndex, float gain)
{
    if (bandIndex >= 0 && bandIndex < NUM_BANDS)
    {
        targetGains[bandIndex] = juce::jlimit(0.0f, 1.0f, gain);
    }
}

void IIRFilterBank::setAllBandGains(const std::array<float, NUM_BANDS>& masks)
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        targetGains[i] = juce::jlimit(0.0f, 1.0f, masks[i]);
    }
}

float IIRFilterBank::getCenterFrequency(int bandIndex) const
{
    if (bandIndex >= 0 && bandIndex < NUM_BANDS)
        return centerFreqs[bandIndex];
    return 0.0f;
}
