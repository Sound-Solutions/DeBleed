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
        //
        // With 64 bands spanning 10 octaves (20Hz-20kHz), each band covers ~0.156 octaves.
        // Use a window width of ~0.12 octaves (slightly narrower than band spacing)
        // to ensure each band primarily responds to its own frequency region.
        // Too wide = bands "grab each other", losing mask resolution.
        const float octaveWidth = 0.12f;  // Was 0.5f - too wide!

        for (int bin = 1; bin < NUM_FFT_BINS; ++bin)  // Skip DC
        {
            float binFreq = binFrequencies[bin];

            if (binFreq < 1.0f)
                continue;

            // Calculate octave distance
            float octaveDistance = std::abs(std::log2(binFreq / centerFreq));

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

        // If no bins mapped, use nearest bin (fallback for narrow window)
        if (mapping.binIndices.empty())
        {
            float centerFreq = bandCenterFreqs[band];
            float binIndexF = centerFreq * 256.0f / static_cast<float>(sampleRate);
            int bin = juce::jlimit(1, NUM_FFT_BINS - 1, static_cast<int>(binIndexF + 0.5f));
            bandGains[band] = juce::jlimit(0.0f, 1.0f, fftMask[bin]);
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

void BandMapper::mapDualStream(const float* dualMask, std::array<float, NUM_IIR_BANDS>& bandGains)
{
    // Dual-stream mask layout:
    // dualMask[0..128] = Stream A (129 bins, ~187Hz resolution, for highs)
    // dualMask[129..256] = Stream B (128 bins, ~23Hz resolution, for bass)
    //
    // Hybrid 32+160 topology mapping:
    // - Bands 0-31 (20-500Hz): Use Stream B (23.4 Hz resolution)
    // - Bands 32-191 (500Hz-20kHz): Use Stream A (187.5 Hz resolution)
    //
    // This ensures bass bands get high-resolution control from Stream B's 2048-point FFT,
    // while high bands use the faster Stream A response.

    // Bin spacing constants (assuming 48kHz sample rate)
    constexpr float streamABinHz = 48000.0f / 256.0f;   // ~187.5 Hz per bin
    constexpr float streamBBinHz = 48000.0f / 2048.0f;  // ~23.4 Hz per bin

    for (int band = 0; band < NUM_IIR_BANDS; ++band)
    {
        float centerFreq = bandCenterFreqs[band];

        if (band < NUM_LOW_BANDS)  // Bands 0-31: Use Stream B
        {
            // LOW BANDS: Use Stream B's high-resolution bins (23.4Hz per bin)
            // Direct linear interpolation between nearest bins
            float binIndexF = centerFreq / streamBBinHz;
            int binLow = static_cast<int>(binIndexF);
            int binHigh = binLow + 1;

            binLow = juce::jlimit(0, NUM_STREAM_B_BINS - 1, binLow);
            binHigh = juce::jlimit(0, NUM_STREAM_B_BINS - 1, binHigh);

            float frac = binIndexF - static_cast<float>(binLow);
            frac = juce::jlimit(0.0f, 1.0f, frac);

            // Stream B mask starts at index 129
            float maskLow = dualMask[NUM_STREAM_A_BINS + binLow];
            float maskHigh = dualMask[NUM_STREAM_A_BINS + binHigh];

            bandGains[band] = maskLow * (1.0f - frac) + maskHigh * frac;
        }
        else  // Bands 32-191: Use Stream A
        {
            // HIGH BANDS: Use Stream A bins - DIRECT mapping with linear interpolation
            float binIndexF = centerFreq / streamABinHz;
            int binLow = static_cast<int>(binIndexF);
            int binHigh = binLow + 1;

            binLow = juce::jlimit(1, NUM_STREAM_A_BINS - 1, binLow);
            binHigh = juce::jlimit(1, NUM_STREAM_A_BINS - 1, binHigh);

            float frac = binIndexF - static_cast<float>(binLow);
            frac = juce::jlimit(0.0f, 1.0f, frac);

            float maskLow = dualMask[binLow];
            float maskHigh = dualMask[binHigh];

            bandGains[band] = juce::jlimit(0.0f, 1.0f, maskLow * (1.0f - frac) + maskHigh * frac);
        }
    }
}

void BandMapper::mapDualStreamWithStrength(const float* dualMask,
                                            std::array<float, NUM_IIR_BANDS>& bandGains,
                                            float strength)
{
    // First get raw mapped values
    mapDualStream(dualMask, bandGains);

    // Apply strength scaling (same as mapWithStrength)
    for (int band = 0; band < NUM_IIR_BANDS; ++band)
    {
        float rawGain = bandGains[band];

        if (strength <= 1.0f)
        {
            // Interpolate between unity and raw gain
            bandGains[band] = 1.0f + (rawGain - 1.0f) * strength;
        }
        else
        {
            // Exaggerate the effect for strength > 1
            float exaggeration = strength - 1.0f;
            float extraReduction = (1.0f - rawGain) * exaggeration;
            bandGains[band] = rawGain - extraReduction;
        }

        bandGains[band] = juce::jlimit(0.0f, 1.0f, bandGains[band]);
    }
}
