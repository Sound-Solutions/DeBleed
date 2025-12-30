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
    // Log-spaced frequencies from 20Hz to 20kHz
    // f[i] = 20 * (20000/20)^(i/(N-1)) = 20 * 1000^(i/63)
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(NUM_BANDS - 1);
        centerFreqs[i] = MIN_FREQ * std::pow(MAX_FREQ / MIN_FREQ, t);
    }

    // Calculate Q for each band based on neighbor spacing
    // Q = fc / bandwidth, where bandwidth is the distance to neighbors
    // This ensures each filter only covers its own frequency region
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
        // Add small multiplier to make filters slightly narrower than spacing
        // This prevents excessive overlap while maintaining coverage
        qFactors[i] = centerFreqs[i] / (bandwidth * 1.2f);

        // Clamp Q to reasonable range
        qFactors[i] = juce::jlimit(4.0f, 30.0f, qFactors[i]);
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
    // Calculate the total bandpass response at 1kHz (a representative mid frequency)
    // This is used to normalize the sum so that unity gains = flat response
    const float testFreq = 1000.0f;
    float totalResponse = 0.0f;

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        totalResponse += getBandpassMagnitudeAt(b, testFreq);
    }

    // Normalization factor: divide by total overlap
    if (totalResponse > 0.001f)
        normalizationFactor = 1.0f / totalResponse;
    else
        normalizationFactor = 1.0f;
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
            // When mask = 0, band contributes nothing (true silence)
            // When mask = 1, band passes at unity
            for (int b = 0; b < NUM_BANDS; ++b)
            {
                // Smooth gain toward target
                bandGains[b] += (targetGains[b] - bandGains[b]) * smoothingCoeff;

                // Process through bandpass (always process to maintain filter state)
                float bpOut = filters[b].filter[ch].processSample(input);

                // Multiply by mask and accumulate
                output += bpOut * bandGains[b];
            }

            // Apply normalization so unity gains = flat response
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
