#include "IIRFilterBank.h"

IIRFilterBank::IIRFilterBank()
{
    // Initialize all gain coefficients to 0 (no change = dry passthrough)
    gainCoeffs.fill(0.0f);

    // Calculate center frequencies
    calculateCenterFrequencies();

    // Initialize coefficient pointers
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        bandpassCoeffs[i] = new juce::dsp::IIR::Coefficients<float>();
    }
}

void IIRFilterBank::calculateCenterFrequencies()
{
    // Logarithmic spacing: f[i] = MIN_FREQ * (MAX_FREQ/MIN_FREQ)^(i/(NUM_BANDS-1))
    const float ratio = MAX_FREQ / MIN_FREQ;

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float t = static_cast<float>(i) / static_cast<float>(NUM_BANDS - 1);
        centerFrequencies[i] = MIN_FREQ * std::pow(ratio, t);
    }
}

void IIRFilterBank::prepare(double newSampleRate, int maxBlockSize)
{
    sampleRate = newSampleRate;

    // Prepare temporary buffer for bandpass accumulation
    bandpassAccum.setSize(2, maxBlockSize);

    // Prepare all bandpass filters
    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(maxBlockSize);
    spec.numChannels = 1;  // Process channels separately

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        // Ensure frequency is valid for sample rate
        float freq = centerFrequencies[i];
        float nyquist = static_cast<float>(sampleRate) * 0.5f;

        if (freq >= nyquist * 0.95f)
            freq = nyquist * 0.95f;

        // Create bandpass coefficients
        *bandpassCoeffs[i] = *juce::dsp::IIR::Coefficients<float>::makeBandPass(
            sampleRate,
            freq,
            qFactor
        );

        bandpassFilters[i].coefficients = bandpassCoeffs[i];
        bandpassFilters[i].prepare(spec);
    }

    reset();
}

void IIRFilterBank::reset()
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        bandpassFilters[i].reset();
    }

    bandpassAccum.clear();
}

void IIRFilterBank::process(juce::AudioBuffer<float>& buffer)
{
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();

    // Process each channel
    for (int ch = 0; ch < numChannels; ++ch)
    {
        float* channelData = buffer.getWritePointer(ch);

        // For each sample, compute: output = dry + sum(bandpass[i] * gainCoeff[i])
        for (int s = 0; s < numSamples; ++s)
        {
            float drySample = channelData[s];
            float bandpassSum = 0.0f;

            // Process ALL bandpass filters to maintain state,
            // but only accumulate those with non-zero gain
            for (int band = 0; band < NUM_BANDS; ++band)
            {
                // Always process to maintain filter state
                float bandpassSample = bandpassFilters[band].processSample(drySample);

                float gainCoeff = gainCoeffs[band];

                // Only accumulate if gainCoeff is non-zero
                if (std::abs(gainCoeff) > 0.001f)
                {
                    // Accumulate: negative gainCoeff = cut, positive = boost
                    bandpassSum += bandpassSample * gainCoeff;
                }
            }

            // Output = Dry + Sum(Bandpass * GainCoeff)
            channelData[s] = drySample + bandpassSum;
        }
    }
}

void IIRFilterBank::setBandGain(int bandIndex, float maskValue)
{
    if (bandIndex < 0 || bandIndex >= NUM_BANDS)
        return;

    // Clamp mask to valid range
    maskValue = juce::jlimit(0.0f, 1.0f, maskValue);

    // Convert mask to gain coefficient:
    // mask = 1.0 → gainCoeff = 0.0 (no change, dry passthrough)
    // mask = 0.0 → gainCoeff = -1.0 (full cut of this band)
    gainCoeffs[bandIndex] = maskValue - 1.0f;
}

void IIRFilterBank::setAllBandGains(const std::array<float, NUM_BANDS>& masks)
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        setBandGain(i, masks[i]);
    }
}

float IIRFilterBank::getCenterFrequency(int bandIndex) const
{
    if (bandIndex < 0 || bandIndex >= NUM_BANDS)
        return 0.0f;

    return centerFrequencies[bandIndex];
}

void IIRFilterBank::setQ(float newQ)
{
    qFactor = juce::jlimit(0.1f, 10.0f, newQ);

    // Update all bandpass filter coefficients with new Q
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        float freq = centerFrequencies[i];
        float nyquist = static_cast<float>(sampleRate) * 0.5f;

        if (freq >= nyquist * 0.95f)
            freq = nyquist * 0.95f;

        *bandpassCoeffs[i] = *juce::dsp::IIR::Coefficients<float>::makeBandPass(
            sampleRate,
            freq,
            qFactor
        );

        bandpassFilters[i].coefficients = bandpassCoeffs[i];
    }
}
