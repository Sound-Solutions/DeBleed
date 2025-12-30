#include "IIRFilterBank.h"

IIRFilterBank::IIRFilterBank()
{
    bandGains.fill(1.0f);
    targetGains.fill(1.0f);
    qFactors.fill(8.0f);  // Will be recalculated
    calculateFrequencies();

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        filters[i].coeffs = new juce::dsp::IIR::Coefficients<float>();
    }
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
    // Q = fc / bandwidth, where bandwidth is the distance to neighbors
    // This ensures each filter only covers its own frequency region
    // Note: Q values will change at the 500 Hz crossover due to density change
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float bandwidth;

        if (i == 0)
        {
            // First band: use distance to next
            bandwidth = centerFreqs[1] - centerFreqs[0];
        }
        else if (i == NUM_BANDS - 1)
        {
            // Last band: use distance to previous
            bandwidth = centerFreqs[NUM_BANDS - 1] - centerFreqs[NUM_BANDS - 2];
        }
        else
        {
            // Middle bands: average of distances to both neighbors
            float bwLower = centerFreqs[i] - centerFreqs[i - 1];
            float bwUpper = centerFreqs[i + 1] - centerFreqs[i];
            bandwidth = (bwLower + bwUpper) * 0.5f;
        }

        // Q = fc / bandwidth
        // With 192 bands, use tighter Q (no multiplier) for sharper selectivity
        // Per-band normalization will compensate for any overlap differences
        qFactors[i] = centerFreqs[i] / bandwidth;

        // Clamp Q to reasonable range (expanded for high-res bands)
        qFactors[i] = juce::jlimit(2.0f, 50.0f, qFactors[i]);
    }
}

float IIRFilterBank::getBandpassMagnitudeAt(int bandIndex, float freq) const
{
    // 2nd-order bandpass magnitude response: |H(f)| = 1 / sqrt(1 + Q² * ((f/fc) - (fc/f))²)
    if (freq <= 0.0f || bandIndex < 0 || bandIndex >= NUM_BANDS)
        return 0.0f;

    float fc = centerFreqs[bandIndex];
    float Q = qFactors[bandIndex];
    float ratio = freq / fc;
    float term = ratio - (1.0f / ratio);  // (f/fc) - (fc/f)
    float denominator = 1.0f + Q * Q * term * term;

    return 1.0f / std::sqrt(denominator);
}

void IIRFilterBank::calculateNormalization()
{
    // Calculate total response at multiple frequencies across the spectrum
    // Use MINIMUM response to ensure we never drop below unity gain
    // (some frequencies may be slightly boosted, but no frequencies will be cut)
    const float testFreqs[] = {25.0f, 40.0f, 63.0f, 100.0f, 160.0f, 250.0f, 400.0f, 630.0f,
                                1000.0f, 1600.0f, 2500.0f, 4000.0f, 6300.0f, 10000.0f, 16000.0f};
    const int numTestFreqs = 15;

    float minResponse = 1e10f;

    for (int f = 0; f < numTestFreqs; ++f)
    {
        float totalResponse = 0.0f;
        for (int b = 0; b < NUM_BANDS; ++b)
        {
            totalResponse += getBandpassMagnitudeAt(b, testFreqs[f]);
        }
        if (totalResponse < minResponse)
            minResponse = totalResponse;
    }

    // Use minimum response for normalization - ensures no frequency drops below unity
    if (minResponse > 0.001f)
        normalizationFactor = 1.0f / minResponse;
    else
        normalizationFactor = 1.0f;

    // Makeup gain to compensate for tight Q not summing to unity
    // With Q = fc/bandwidth (no 1.2x multiplier), bands don't overlap enough
    // ~3dB boost needed empirically
    normalizationFactor *= 1.41f;  // +3dB
}

void IIRFilterBank::prepare(double newSampleRate, int maxBlockSize)
{
    sampleRate = newSampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maxBlockSize);
    spec.numChannels = 1;

    float nyquist = static_cast<float>(sampleRate) * 0.5f;

    // Initialize bandpass filters
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float freq = centerFreqs[i];
        if (freq >= nyquist * 0.95f)
            freq = nyquist * 0.95f;

        // Create bandpass coefficients with calculated Q
        *filters[i].coeffs = *juce::dsp::IIR::Coefficients<float>::makeBandPass(
            sampleRate, freq, qFactors[i]);

        // Assign to both stereo channels
        filters[i].filter[0].coefficients = filters[i].coeffs;
        filters[i].filter[1].coefficients = filters[i].coeffs;

        filters[i].filter[0].prepare(spec);
        filters[i].filter[1].prepare(spec);
    }

    // Calculate normalization factor for flat response at unity gains
    calculateNormalization();

    // Allocate temp buffer
    bandpassOutput.resize(maxBlockSize, 0.0f);

    // Smoothing based on ~5ms
    smoothingCoeff = 1.0f - std::exp(-1.0f / (0.005f * static_cast<float>(sampleRate)));

    reset();
}

void IIRFilterBank::reset()
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        filters[i].filter[0].reset();
        filters[i].filter[1].reset();
    }
}

void IIRFilterBank::process(juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        float* data = buffer.getWritePointer(ch);

        // Process each sample
        for (int s = 0; s < numSamples; ++s)
        {
            float input = data[s];
            float output = 0.0f;  // NO DRY PATH - start with zero

            // Split & Sum topology: Output = Sum(Bandpass * mask) * normalization
            for (int b = 0; b < NUM_BANDS; ++b)
            {
                // Smooth gain toward target
                bandGains[b] += (targetGains[b] - bandGains[b]) * smoothingCoeff;

                // Process through bandpass (always process to maintain filter state)
                float bpOut = filters[b].filter[ch].processSample(input);

                // Multiply by mask and accumulate
                output += bpOut * bandGains[b];
            }

            // Apply global normalization
            output *= normalizationFactor;

            data[s] = output;
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
