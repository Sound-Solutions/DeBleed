#include "BandMapper.h"

BandMapper::BandMapper()
{
    binFrequencies.fill(0.0f);
    bandCenterFreqs.fill(0.0f);
}

void BandMapper::prepare(double newSampleRate, const std::array<float, NUM_IIR_BANDS>& iirCenterFreqs)
{
    sampleRate = newSampleRate;
    bandCenterFreqs = iirCenterFreqs;

    // Calculate FFT bin frequencies
    // Assuming 256-point FFT (128 positive bins + DC = 129 bins)
    // Each bin = sampleRate / FFT_SIZE
    // Bin 0 = DC, Bin 128 = Nyquist
    const float nyquist = static_cast<float>(sampleRate) * 0.5f;
    const float binWidth = nyquist / (NUM_FFT_BINS - 1);

    for (int bin = 0; bin < NUM_FFT_BINS; ++bin)
    {
        binFrequencies[bin] = bin * binWidth;
    }

    // Precompute mapping weights for each IIR band
    for (int band = 0; band < NUM_IIR_BANDS; ++band)
    {
        mappings[band].binIndices.clear();
        mappings[band].binWeights.clear();
        mappings[band].centerFreq = bandCenterFreqs[band];

        float centerFreq = bandCenterFreqs[band];

        // Skip DC (bin 0) for very low frequencies
        if (centerFreq < 10.0f)
            centerFreq = 10.0f;

        float totalWeight = 0.0f;

        // Find overlapping FFT bins using triangular weighting
        // Weight based on log-frequency distance (octave distance)
        for (int bin = 1; bin < NUM_FFT_BINS; ++bin)  // Skip DC
        {
            float binFreq = binFrequencies[bin];

            if (binFreq < 1.0f)
                continue;

            // Calculate octave distance
            float octaveDistance = std::abs(std::log2(binFreq / centerFreq));

            // Use triangular window within ±0.5 octaves
            // This provides smooth interpolation between adjacent bands
            const float octaveWidth = 0.5f;

            if (octaveDistance < octaveWidth)
            {
                float weight = 1.0f - (octaveDistance / octaveWidth);
                weight = weight * weight;  // Squared for smoother falloff

                mappings[band].binIndices.push_back(bin);
                mappings[band].binWeights.push_back(weight);
                totalWeight += weight;
            }
        }

        // Normalize weights
        if (totalWeight > 0.0f)
        {
            for (float& w : mappings[band].binWeights)
            {
                w /= totalWeight;
            }
        }
    }
}

void BandMapper::map(const float* fftMask, std::array<float, NUM_IIR_BANDS>& bandGains)
{
    for (int band = 0; band < NUM_IIR_BANDS; ++band)
    {
        const auto& mapping = mappings[band];

        // If no bins mapped (shouldn't happen normally), use unity
        if (mapping.binIndices.empty())
        {
            bandGains[band] = 1.0f;
            continue;
        }

        // Weighted average of mapped FFT bins
        float gain = 0.0f;
        for (size_t i = 0; i < mapping.binIndices.size(); ++i)
        {
            int bin = mapping.binIndices[i];
            float weight = mapping.binWeights[i];
            float maskVal = fftMask[bin];

            // Clamp mask value
            maskVal = juce::jlimit(0.0f, 1.0f, maskVal);

            gain += maskVal * weight;
        }

        bandGains[band] = gain;
    }
}

void BandMapper::mapWithStrength(const float* fftMask,
                                  std::array<float, NUM_IIR_BANDS>& bandGains,
                                  float strength)
{
    // First get raw mapped values
    map(fftMask, bandGains);

    // Apply strength scaling
    // strength = 0: no effect (gain = 1.0)
    // strength = 1: full effect (gain = mapped value)
    // strength > 1: exaggerated effect
    for (int band = 0; band < NUM_IIR_BANDS; ++band)
    {
        float rawGain = bandGains[band];

        // Interpolate between unity and raw gain based on strength
        if (strength <= 1.0f)
        {
            bandGains[band] = 1.0f + (rawGain - 1.0f) * strength;
        }
        else
        {
            // For strength > 1, exaggerate the effect
            // First apply full effect, then push further
            float exaggeration = strength - 1.0f;
            float extraReduction = (1.0f - rawGain) * exaggeration;
            bandGains[band] = rawGain - extraReduction;
        }

        // Clamp final gain
        bandGains[band] = juce::jlimit(0.0f, 1.0f, bandGains[band]);
    }
}

void BandMapper::setFrequencyRange(float /*lowHz*/, float /*highHz*/)
{
    // Frequency range filtering removed - all bands respond to neural mask
}
