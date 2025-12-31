#include "DifferentiableBiquadChain.h"
#include <cmath>

DifferentiableBiquadChain::DifferentiableBiquadChain()
{
    // Initialize all parameters to neutral (0.5 = center)
    for (auto& p : currentParams_)
        p.store(0.5f);

    // Initialize gains to unity (normalized 1.0 = 0dB for broadband)
    currentParams_[N_FILTERS * N_PARAMS_PER_FILTER].store(1.0f);      // Input gain
    currentParams_[N_FILTERS * N_PARAMS_PER_FILTER + 1].store(1.0f);  // Output gain
}

void DifferentiableBiquadChain::prepare(double sampleRate, int maxBlockSize)
{
    sampleRate_ = sampleRate;
    maxBlockSize_ = maxBlockSize;

    // Reset filter states
    reset();

    // Initialize smoothed values
    for (auto& sv : smoothedParams_)
    {
        sv.reset(sampleRate, SMOOTHING_TIME_MS / 1000.0);
        sv.setCurrentAndTargetValue(0.5f);
    }

    inputGainLinear_.reset(sampleRate, SMOOTHING_TIME_MS / 1000.0);
    inputGainLinear_.setCurrentAndTargetValue(1.0f);

    outputGainLinear_.reset(sampleRate, SMOOTHING_TIME_MS / 1000.0);
    outputGainLinear_.setCurrentAndTargetValue(1.0f);

    // Initialize coefficients to neutral (bypass-ish)
    for (int i = 0; i < N_FILTERS; ++i)
    {
        computeCoeffs(i, 1000.0f, 0.0f, 1.0f);
        currentCoeffs_[i] = targetCoeffs_[i];
    }

    frameCounter_ = 0;
}

void DifferentiableBiquadChain::reset()
{
    // Clear all filter states
    for (auto& filterStates : filterStates_)
    {
        for (auto& state : filterStates)
        {
            state.ic1eq = 0.0f;
            state.ic2eq = 0.0f;
        }
    }

    frameCounter_ = 0;
}

void DifferentiableBiquadChain::setParameters(const float* params)
{
    // Store all parameters atomically
    for (int i = 0; i < N_TOTAL_PARAMS; ++i)
    {
        currentParams_[i].store(params[i]);
    }

    // Update smoothed targets for each filter
    for (int f = 0; f < N_FILTERS; ++f)
    {
        int baseIdx = f * N_PARAMS_PER_FILTER;
        smoothedParams_[baseIdx + 0].setTargetValue(params[baseIdx + 0]);  // freq
        smoothedParams_[baseIdx + 1].setTargetValue(params[baseIdx + 1]);  // gain
        smoothedParams_[baseIdx + 2].setTargetValue(params[baseIdx + 2]);  // Q
    }

    // Update gain targets
    float inputGainDb = denormalizeGain(params[N_FILTERS * N_PARAMS_PER_FILTER],
                                         BROADBAND_MIN_GAIN, BROADBAND_MAX_GAIN);
    float outputGainDb = denormalizeGain(params[N_FILTERS * N_PARAMS_PER_FILTER + 1],
                                          BROADBAND_MIN_GAIN, BROADBAND_MAX_GAIN);

    inputGainLinear_.setTargetValue(std::pow(10.0f, inputGainDb / 20.0f));
    outputGainLinear_.setTargetValue(std::pow(10.0f, outputGainDb / 20.0f));
}

void DifferentiableBiquadChain::process(juce::AudioBuffer<float>& buffer)
{
    int numSamples = buffer.getNumSamples();
    int numChannels = std::min(buffer.getNumChannels(), 2);

    for (int sample = 0; sample < numSamples; ++sample)
    {
        // Update coefficients every FRAME_SIZE samples
        if (frameCounter_ == 0)
        {
            for (int f = 0; f < N_FILTERS; ++f)
            {
                int baseIdx = f * N_PARAMS_PER_FILTER;

                // Get smoothed parameter values
                float freqNorm = smoothedParams_[baseIdx + 0].getNextValue();
                float gainNorm = smoothedParams_[baseIdx + 1].getNextValue();
                float qNorm = smoothedParams_[baseIdx + 2].getNextValue();

                // Denormalize based on filter type
                FilterType type = getFilterType(f);
                float freqHz, gainDb, q;

                switch (type)
                {
                    case FilterType::HighPass:
                        freqHz = denormalizeFreq(freqNorm, HPF_MIN_FREQ, HPF_MAX_FREQ);
                        gainDb = 0.0f;  // HPF doesn't use gain
                        q = denormalizeQ(qNorm);
                        break;

                    case FilterType::LowPass:
                        freqHz = denormalizeFreq(freqNorm, LPF_MIN_FREQ, LPF_MAX_FREQ);
                        gainDb = 0.0f;  // LPF doesn't use gain
                        q = denormalizeQ(qNorm);
                        break;

                    case FilterType::LowShelf:
                    case FilterType::HighShelf:
                        freqHz = denormalizeFreq(freqNorm, SHELF_MIN_FREQ, SHELF_MAX_FREQ);
                        gainDb = denormalizeGain(gainNorm, FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                        q = denormalizeQ(qNorm);
                        break;

                    case FilterType::Peak:
                    default:
                        freqHz = denormalizeFreq(freqNorm, PEAK_MIN_FREQ, PEAK_MAX_FREQ);
                        gainDb = denormalizeGain(gainNorm, FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                        q = denormalizeQ(qNorm);
                        break;
                }

                computeCoeffs(f, freqHz, gainDb, q);
            }

            // Interpolate from current to target coefficients
            for (int f = 0; f < N_FILTERS; ++f)
            {
                currentCoeffs_[f] = targetCoeffs_[f];
            }
        }

        frameCounter_ = (frameCounter_ + 1) % FRAME_SIZE;

        // Get smoothed gains
        float inGain = inputGainLinear_.getNextValue();
        float outGain = outputGainLinear_.getNextValue();

        // Process each channel
        for (int ch = 0; ch < numChannels; ++ch)
        {
            float* channelData = buffer.getWritePointer(ch);

            // Apply input gain
            float x = channelData[sample] * inGain;

            // Process through all filters in series
            for (int f = 0; f < N_FILTERS; ++f)
            {
                x = processSVF(x, f, ch);
            }

            // Apply output gain and write back
            channelData[sample] = x * outGain;

            // Safety clamp
            if (std::isnan(channelData[sample]) || std::isinf(channelData[sample]))
                channelData[sample] = 0.0f;
        }
    }

    // Advance smoothers for remaining samples in the frame
    for (auto& sv : smoothedParams_)
        sv.skip(numSamples);
}

float DifferentiableBiquadChain::processSVF(float input, int filterIndex, int channel)
{
    const SVFCoeffs& c = currentCoeffs_[filterIndex];
    SVFState& s = filterStates_[filterIndex][channel];

    // SVF TPT processing (Topology-Preserving Transform)
    // Based on Cytomic technical papers

    float v3 = input - s.ic2eq;
    float v1 = c.a1 * s.ic1eq + c.a2 * v3;
    float v2 = s.ic2eq + c.a2 * v1 + c.a3 * v3;

    // Update state
    s.ic1eq = 2.0f * v1 - s.ic1eq;
    s.ic2eq = 2.0f * v2 - s.ic2eq;

    // Mix outputs based on filter type
    // low = v2, band = v1, high = input - c.k * v1 - v2
    float low = v2;
    float band = v1;
    float high = input - c.k * v1 - v2;

    // General mixing formula: output = m0*high + m1*band + m2*low
    // But we store it differently for efficiency
    return c.m0 * input + c.m1 * band + c.m2 * low;
}

void DifferentiableBiquadChain::computeCoeffs(int filterIndex, float freqHz, float gainDb, float q)
{
    SVFCoeffs& c = targetCoeffs_[filterIndex];

    // Clamp frequency to valid range
    float nyquist = static_cast<float>(sampleRate_) * 0.5f;
    freqHz = std::clamp(freqHz, 20.0f, nyquist * 0.95f);

    // Clamp Q
    q = std::clamp(q, 0.1f, 100.0f);

    // Basic SVF coefficients
    c.g = std::tan(juce::MathConstants<float>::pi * freqHz / static_cast<float>(sampleRate_));

    FilterType type = getFilterType(filterIndex);

    // Compute type-specific coefficients
    switch (type)
    {
        case FilterType::HighPass:
        {
            c.k = 1.0f / q;
            c.A = 1.0f;

            float g2 = c.g * c.g;
            c.a1 = 1.0f / (1.0f + c.g * (c.g + c.k));
            c.a2 = c.g * c.a1;
            c.a3 = c.g * c.a2;

            // Highpass: output = high = input - k*band - low
            c.m0 = 1.0f;
            c.m1 = -c.k;
            c.m2 = -1.0f;
            break;
        }

        case FilterType::LowPass:
        {
            c.k = 1.0f / q;
            c.A = 1.0f;

            c.a1 = 1.0f / (1.0f + c.g * (c.g + c.k));
            c.a2 = c.g * c.a1;
            c.a3 = c.g * c.a2;

            // Lowpass: output = low = v2
            c.m0 = 0.0f;
            c.m1 = 0.0f;
            c.m2 = 1.0f;
            break;
        }

        case FilterType::Peak:
        {
            float A = std::pow(10.0f, gainDb / 40.0f);  // sqrt of linear gain
            c.A = A;

            // For peaking EQ, k depends on whether we're boosting or cutting
            if (gainDb >= 0.0f)
            {
                // Boost
                c.k = 1.0f / (q * A);
            }
            else
            {
                // Cut
                c.k = A / q;
            }

            c.a1 = 1.0f / (1.0f + c.g * (c.g + c.k));
            c.a2 = c.g * c.a1;
            c.a3 = c.g * c.a2;

            // Peak: output = input + (A^2 - 1) * k * band
            // But we need different formulation for boost vs cut
            if (gainDb >= 0.0f)
            {
                c.m0 = 1.0f;
                c.m1 = c.k * (A * A - 1.0f);
                c.m2 = 0.0f;
            }
            else
            {
                // For cut, we need to invert the response
                float invA = 1.0f / A;
                c.m0 = 1.0f;
                c.m1 = c.k * (invA * invA - 1.0f);
                c.m2 = 0.0f;
            }
            break;
        }

        case FilterType::LowShelf:
        {
            float A = std::pow(10.0f, gainDb / 40.0f);
            c.A = A;

            // Low shelf uses modified g
            float sqrtA = std::sqrt(A);
            float gShelf = c.g / sqrtA;

            if (gainDb >= 0.0f)
            {
                // Boost
                c.k = 1.0f / q;
                c.a1 = 1.0f / (1.0f + gShelf * (gShelf + c.k));
                c.a2 = gShelf * c.a1;
                c.a3 = gShelf * c.a2;

                c.m0 = 1.0f;
                c.m1 = c.k * (A - 1.0f);
                c.m2 = A * A - 1.0f;
            }
            else
            {
                // Cut - different topology
                c.k = 1.0f / q;
                float gShelfCut = c.g * sqrtA;
                c.a1 = 1.0f / (1.0f + gShelfCut * (gShelfCut + c.k));
                c.a2 = gShelfCut * c.a1;
                c.a3 = gShelfCut * c.a2;

                float invA = 1.0f / A;
                c.m0 = 1.0f;
                c.m1 = c.k * (invA - 1.0f);
                c.m2 = invA * invA - 1.0f;
            }
            break;
        }

        case FilterType::HighShelf:
        {
            float A = std::pow(10.0f, gainDb / 40.0f);
            c.A = A;

            float sqrtA = std::sqrt(A);
            float gShelf = c.g * sqrtA;

            if (gainDb >= 0.0f)
            {
                // Boost
                c.k = 1.0f / q;
                c.a1 = 1.0f / (1.0f + gShelf * (gShelf + c.k));
                c.a2 = gShelf * c.a1;
                c.a3 = gShelf * c.a2;

                c.m0 = A * A;
                c.m1 = c.k * (1.0f - A) * A;
                c.m2 = 1.0f - A * A;
            }
            else
            {
                // Cut
                c.k = 1.0f / q;
                float gShelfCut = c.g / sqrtA;
                c.a1 = 1.0f / (1.0f + gShelfCut * (gShelfCut + c.k));
                c.a2 = gShelfCut * c.a1;
                c.a3 = gShelfCut * c.a2;

                float invA = 1.0f / A;
                c.m0 = invA * invA;
                c.m1 = c.k * (1.0f - invA) * invA;
                c.m2 = 1.0f - invA * invA;
            }
            break;
        }

        case FilterType::Bypass:
        default:
            c.k = 1.0f;
            c.A = 1.0f;
            c.a1 = 0.0f;
            c.a2 = 0.0f;
            c.a3 = 0.0f;
            c.m0 = 1.0f;
            c.m1 = 0.0f;
            c.m2 = 0.0f;
            break;
    }
}

DifferentiableBiquadChain::FilterType DifferentiableBiquadChain::getFilterType(int filterIndex) const
{
    if (filterIndex == FILTER_HPF)
        return FilterType::HighPass;
    if (filterIndex == FILTER_LPF)
        return FilterType::LowPass;
    if (filterIndex == FILTER_LOW_SHELF)
        return FilterType::LowShelf;
    if (filterIndex == FILTER_HIGH_SHELF)
        return FilterType::HighShelf;
    if (filterIndex >= FILTER_PEAK_START && filterIndex <= FILTER_PEAK_END)
        return FilterType::Peak;

    return FilterType::Bypass;
}

float DifferentiableBiquadChain::denormalizeFreq(float norm, float minHz, float maxHz) const
{
    // Log-scale frequency mapping
    float logMin = std::log(minHz);
    float logMax = std::log(maxHz);
    return std::exp(logMin + norm * (logMax - logMin));
}

float DifferentiableBiquadChain::denormalizeGain(float norm, float minDb, float maxDb) const
{
    // Linear dB mapping
    return minDb + norm * (maxDb - minDb);
}

float DifferentiableBiquadChain::denormalizeQ(float norm) const
{
    // Log-scale Q mapping
    float logMin = std::log(Q_MIN);
    float logMax = std::log(Q_MAX);
    return std::exp(logMin + norm * (logMax - logMin));
}

float DifferentiableBiquadChain::getFrequencyResponse(float freqHz) const
{
    // Compute combined magnitude response of all filters
    // Using the transfer function evaluated at the given frequency

    float omega = 2.0f * juce::MathConstants<float>::pi * freqHz / static_cast<float>(sampleRate_);
    std::complex<float> z = std::polar(1.0f, omega);
    std::complex<float> totalResponse(1.0f, 0.0f);

    for (int f = 0; f < N_FILTERS; ++f)
    {
        const SVFCoeffs& c = currentCoeffs_[f];

        // Compute SVF frequency response
        // This is an approximation - for accurate results we'd need the full transfer function
        float g = c.g;
        float k = c.k;

        std::complex<float> s = (z - 1.0f) / (z + 1.0f);  // Bilinear transform approximation
        s = s * static_cast<float>(sampleRate_);  // Scale

        // Simplified response based on filter type
        FilterType type = getFilterType(f);
        float gain = 1.0f;

        switch (type)
        {
            case FilterType::Peak:
                gain = c.A * c.A;  // Approximate peak gain
                break;
            case FilterType::LowShelf:
            case FilterType::HighShelf:
                gain = c.A * c.A;
                break;
            default:
                break;
        }

        totalResponse *= std::complex<float>(gain, 0.0f);
    }

    // Include input/output gains
    float inputGain = inputGainLinear_.getCurrentValue();
    float outputGain = outputGainLinear_.getCurrentValue();

    return std::abs(totalResponse) * inputGain * outputGain;
}

void DifferentiableBiquadChain::getFilterParams(int filterIndex, float& freqHz, float& gainDb, float& q) const
{
    if (filterIndex < 0 || filterIndex >= N_FILTERS)
    {
        freqHz = 1000.0f;
        gainDb = 0.0f;
        q = 1.0f;
        return;
    }

    int baseIdx = filterIndex * N_PARAMS_PER_FILTER;
    float freqNorm = currentParams_[baseIdx + 0].load();
    float gainNorm = currentParams_[baseIdx + 1].load();
    float qNorm = currentParams_[baseIdx + 2].load();

    FilterType type = getFilterType(filterIndex);

    switch (type)
    {
        case FilterType::HighPass:
            freqHz = denormalizeFreq(freqNorm, HPF_MIN_FREQ, HPF_MAX_FREQ);
            gainDb = 0.0f;
            break;
        case FilterType::LowPass:
            freqHz = denormalizeFreq(freqNorm, LPF_MIN_FREQ, LPF_MAX_FREQ);
            gainDb = 0.0f;
            break;
        case FilterType::LowShelf:
        case FilterType::HighShelf:
            freqHz = denormalizeFreq(freqNorm, SHELF_MIN_FREQ, SHELF_MAX_FREQ);
            gainDb = denormalizeGain(gainNorm, FILTER_MIN_GAIN, FILTER_MAX_GAIN);
            break;
        case FilterType::Peak:
        default:
            freqHz = denormalizeFreq(freqNorm, PEAK_MIN_FREQ, PEAK_MAX_FREQ);
            gainDb = denormalizeGain(gainNorm, FILTER_MIN_GAIN, FILTER_MAX_GAIN);
            break;
    }

    q = denormalizeQ(qNorm);
}
