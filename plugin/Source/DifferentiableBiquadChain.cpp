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

    // Initialize coefficients to safe bypass values
    // This ensures getFrequencyResponse() works even before prepare() is called
    for (int i = 0; i < N_FILTERS; ++i)
    {
        SVFCoeffs c;
        c.g = 0.0f;
        c.k = 1.0f;
        c.A = 1.0f;
        c.a1 = 1.0f;
        c.a2 = 0.0f;
        c.a3 = 0.0f;
        c.m0 = 1.0f;
        c.m1 = 0.0f;
        c.m2 = 0.0f;
        targetCoeffs_[i] = c;
        currentCoeffs_[i] = c;
    }

    // Initialize filter states to zero
    for (auto& filterStates : filterStates_)
    {
        for (auto& state : filterStates)
        {
            state.ic1eq = 0.0f;
            state.ic2eq = 0.0f;
        }
    }
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

    // Gain smoothing needs to be slower than coefficient smoothing to avoid pops
    // Use 200ms minimum for gains (neural predictions can swing wildly)
    const double gainSmoothingTimeSecs = std::max(smoothingTimeMs_ * 10.0, 200.0) / 1000.0;
    inputGainLinear_.reset(sampleRate, gainSmoothingTimeSecs);
    inputGainLinear_.setCurrentAndTargetValue(1.0f);

    outputGainLinear_.reset(sampleRate, gainSmoothingTimeSecs);
    outputGainLinear_.setCurrentAndTargetValue(1.0f);

    // Initialize coefficients to neutral (bypass-ish)
    for (int i = 0; i < N_FILTERS; ++i)
    {
        FilterType type = getFilterType(i);

        // HPF and LPF are disabled - set to true bypass
        if (type == FilterType::HighPass || type == FilterType::LowPass)
        {
            SVFCoeffs& c = targetCoeffs_[i];
            c.g = 0.0f;
            c.k = 1.0f;
            c.A = 1.0f;
            c.a1 = 0.0f;
            c.a2 = 0.0f;
            c.a3 = 0.0f;
            c.m0 = 1.0f;  // output = input (bypass)
            c.m1 = 0.0f;
            c.m2 = 0.0f;
        }
        else
        {
            computeCoeffs(i, 1000.0f, 0.0f, 1.0f);
        }
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

    // Add user output gain offset (Phase 3)
    outputGainDb += outputGainOffset_.load();

    inputGainLinear_.setTargetValue(std::pow(10.0f, inputGainDb / 20.0f));
    outputGainLinear_.setTargetValue(std::pow(10.0f, outputGainDb / 20.0f));
}

void DifferentiableBiquadChain::process(juce::AudioBuffer<float>& buffer)
{
    int numSamples = buffer.getNumSamples();
    int numChannels = std::min(buffer.getNumChannels(), 2);

    // Coefficient interpolation - use exponential smoothing
    // Use user-controllable smoothing time (default 50ms from param, scaled up for safety)
    // Longer = smoother but more latent; shorter = more responsive but can pop
    const float effectiveSmoothingMs = std::max(smoothingTimeMs_ * 10.0f, 100.0f);  // Min 100ms
    const float tau = static_cast<float>(sampleRate_) * effectiveSmoothingMs / 1000.0f;
    const float coeffSmoothingCoeff = 1.0f - std::exp(-1.0f / tau);

    for (int sample = 0; sample < numSamples; ++sample)
    {
        // Update target coefficients every FRAME_SIZE samples
        if (frameCounter_ == 0)
        {
            // Get Phase 3 overrides
            float hpfOver = hpfOverride_.load();
            float lpfOver = lpfOverride_.load();
            float sens = sensitivity_.load();

            for (int f = 0; f < N_FILTERS; ++f)
            {
                // Skip HPF and LPF - disabled, user can add their own
                FilterType type = getFilterType(f);
                if (type == FilterType::HighPass || type == FilterType::LowPass)
                    continue;

                int baseIdx = f * N_PARAMS_PER_FILTER;

                // Get smoothed parameter values
                float freqNorm = smoothedParams_[baseIdx + 0].getNextValue();
                float gainNorm = smoothedParams_[baseIdx + 1].getNextValue();
                float qNorm = smoothedParams_[baseIdx + 2].getNextValue();

                // Denormalize based on filter type
                float freqHz, gainDb, q;

                switch (type)
                {
                    case FilterType::LowShelf:
                    case FilterType::HighShelf:
                        freqHz = denormalizeFreq(freqNorm, SHELF_MIN_FREQ, SHELF_MAX_FREQ);
                        gainDb = denormalizeGain(gainNorm, FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                        // Apply sensitivity: 0 = no EQ (0dB), 1 = full EQ
                        gainDb *= sens;
                        // Clamp to cuts only, max -12dB to prevent over-muffling
                        gainDb = std::clamp(gainDb, -12.0f, 0.0f);
                        q = denormalizeQ(qNorm);
                        break;

                    case FilterType::Peak:
                    default:
                        freqHz = denormalizeFreq(freqNorm, PEAK_MIN_FREQ, PEAK_MAX_FREQ);
                        gainDb = denormalizeGain(gainNorm, FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                        // Apply sensitivity: 0 = no EQ (0dB), 1 = full EQ
                        gainDb *= sens;
                        // Clamp to cuts only, max -12dB to prevent over-muffling
                        gainDb = std::clamp(gainDb, -12.0f, 0.0f);
                        q = denormalizeQ(qNorm);
                        break;
                }

                computeCoeffs(f, freqHz, gainDb, q);
            }
        }

        // Smoothly interpolate coefficients every sample to avoid clicks
        for (int f = 0; f < N_FILTERS; ++f)
        {
            // Skip HPF and LPF - they stay at bypass
            FilterType type = getFilterType(f);
            if (type == FilterType::HighPass || type == FilterType::LowPass)
                continue;

            SVFCoeffs& curr = currentCoeffs_[f];
            const SVFCoeffs& tgt = targetCoeffs_[f];

            curr.g  += coeffSmoothingCoeff * (tgt.g  - curr.g);
            curr.k  += coeffSmoothingCoeff * (tgt.k  - curr.k);
            curr.A  += coeffSmoothingCoeff * (tgt.A  - curr.A);
            curr.a1 += coeffSmoothingCoeff * (tgt.a1 - curr.a1);
            curr.a2 += coeffSmoothingCoeff * (tgt.a2 - curr.a2);
            curr.a3 += coeffSmoothingCoeff * (tgt.a3 - curr.a3);
            curr.m0 += coeffSmoothingCoeff * (tgt.m0 - curr.m0);
            curr.m1 += coeffSmoothingCoeff * (tgt.m1 - curr.m1);
            curr.m2 += coeffSmoothingCoeff * (tgt.m2 - curr.m2);
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

    // NOTE: Do NOT call skip() on smoothedParams_ here!
    // The smoothers are sampled once per FRAME_SIZE when we call getNextValue(),
    // and the coefficient interpolation handles per-sample smoothing.
    // Calling skip() causes the smoothers to advance faster than they're sampled,
    // resulting in discontinuities (pops) when parameters jump.
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
            // This maintains consistent Q across boost/cut
            if (gainDb >= 0.0f)
            {
                // Boost: k = 1/(Q*A)
                c.k = 1.0f / (q * A);
            }
            else
            {
                // Cut: k = A/Q
                c.k = A / q;
            }

            c.a1 = 1.0f / (1.0f + c.g * (c.g + c.k));
            c.a2 = c.g * c.a1;
            c.a3 = c.g * c.a2;

            // Peak: output = input + (A^2 - 1) * k * band
            // Same formula for both boost and cut - sign is automatic
            // For boost (A > 1): A^2 - 1 > 0, positive contribution
            // For cut (A < 1): A^2 - 1 < 0, negative contribution
            c.m0 = 1.0f;
            c.m1 = c.k * (A * A - 1.0f);
            c.m2 = 0.0f;
            break;
        }

        case FilterType::LowShelf:
        {
            float A = std::pow(10.0f, gainDb / 40.0f);
            c.A = A;
            c.k = 1.0f / q;

            // Low shelf uses modified g based on gain direction
            float sqrtA = std::sqrt(A);

            if (gainDb >= 0.0f)
            {
                // Boost: g_shelf = g / sqrt(A)
                float gShelf = c.g / sqrtA;
                c.a1 = 1.0f / (1.0f + gShelf * (gShelf + c.k));
                c.a2 = gShelf * c.a1;
                c.a3 = gShelf * c.a2;
            }
            else
            {
                // Cut: g_shelf = g * sqrt(A)
                float gShelf = c.g * sqrtA;
                c.a1 = 1.0f / (1.0f + gShelf * (gShelf + c.k));
                c.a2 = gShelf * c.a1;
                c.a3 = gShelf * c.a2;
            }

            // Same mixing formula for both - sign handled by A value
            // For boost (A > 1): positive gains
            // For cut (A < 1): A - 1 < 0 and A^2 - 1 < 0
            c.m0 = 1.0f;
            c.m1 = c.k * (A - 1.0f);
            c.m2 = A * A - 1.0f;
            break;
        }

        case FilterType::HighShelf:
        {
            float A = std::pow(10.0f, gainDb / 40.0f);
            c.A = A;
            c.k = 1.0f / q;

            float sqrtA = std::sqrt(A);

            if (gainDb >= 0.0f)
            {
                // Boost: g_shelf = g * sqrt(A)
                float gShelf = c.g * sqrtA;
                c.a1 = 1.0f / (1.0f + gShelf * (gShelf + c.k));
                c.a2 = gShelf * c.a1;
                c.a3 = gShelf * c.a2;
            }
            else
            {
                // Cut: g_shelf = g / sqrt(A)
                float gShelf = c.g / sqrtA;
                c.a1 = 1.0f / (1.0f + gShelf * (gShelf + c.k));
                c.a2 = gShelf * c.a1;
                c.a3 = gShelf * c.a2;
            }

            // Same mixing formula for both - sign handled by A value
            c.m0 = A * A;
            c.m1 = c.k * (1.0f - A) * A;
            c.m2 = 1.0f - A * A;
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
    // Compute combined magnitude response of all filters using SVF transfer function
    // H(s) = m0 + m1 * s / (s^2 + k*s + 1) + m2 / (s^2 + k*s + 1)

    if (sampleRate_ <= 0.0)
        return 1.0f;

    // Compute normalized angular frequency
    float w0 = 2.0f * juce::MathConstants<float>::pi * freqHz / static_cast<float>(sampleRate_);

    // For frequency response, evaluate at s = j*omega using the bilinear transform relationship
    // z = e^(j*omega), and we need the discrete-time response

    std::complex<float> totalResponse(1.0f, 0.0f);

    for (int f = 0; f < N_FILTERS; ++f)
    {
        const SVFCoeffs& c = currentCoeffs_[f];

        // SVF frequency response using the mixing coefficients
        // The SVF output is: y = m0*input + m1*bandpass + m2*lowpass

        // Use the prewarped frequency response
        float g = c.g;
        float k = c.k;

        // At frequency w, the SVF responses are:
        // Lowpass: g^2 / (1 + k*g + g^2)
        // Bandpass: g / (1 + k*g + g^2)
        // Highpass: 1 / (1 + k*g + g^2)

        // For our frequency of interest, compute the actual g at that frequency
        float gw = std::tan(w0 / 2.0f);

        // Common denominator for SVF at this frequency
        float denom = 1.0f + k * gw + gw * gw;
        if (denom < 1e-10f) denom = 1e-10f;

        float hp_mag = 1.0f / denom;
        float bp_mag = gw / denom;
        float lp_mag = gw * gw / denom;

        // Combined response using mixing coefficients: m0*hp + m1*bp + m2*lp
        // But m0 represents the input feed-through, so it's:
        // output = m0*input + m1*band + m2*low
        // For HPF: m0=1, m1=-k, m2=-1 -> response = hp_mag - k*bp_mag - lp_mag
        // This is complex, but for magnitude we can approximate

        // More accurate: compute complex response
        // At frequency w, the normalized s is: s = j*tan(w/2) / g_design
        // where g_design is the g computed for the filter's center frequency

        // Simplified magnitude calculation based on filter type
        FilterType type = getFilterType(f);
        float filterResponse = 1.0f;

        switch (type)
        {
            case FilterType::HighPass:
            {
                // HPF response rolls off below cutoff
                float fc = denormalizeFreq(currentParams_[f * N_PARAMS_PER_FILTER].load(), HPF_MIN_FREQ, HPF_MAX_FREQ);
                float ratio = freqHz / fc;
                // Second-order highpass approximation
                filterResponse = ratio * ratio / std::sqrt(1.0f + ratio * ratio * ratio * ratio);
                filterResponse = std::min(filterResponse, 1.0f);
                break;
            }

            case FilterType::LowPass:
            {
                // LPF response rolls off above cutoff
                float fc = denormalizeFreq(currentParams_[f * N_PARAMS_PER_FILTER].load(), LPF_MIN_FREQ, LPF_MAX_FREQ);
                float ratio = freqHz / fc;
                // Second-order lowpass approximation
                filterResponse = 1.0f / std::sqrt(1.0f + ratio * ratio * ratio * ratio);
                break;
            }

            case FilterType::LowShelf:
            {
                float fc = denormalizeFreq(currentParams_[f * N_PARAMS_PER_FILTER].load(), SHELF_MIN_FREQ, SHELF_MAX_FREQ);
                float gainDb = denormalizeGain(currentParams_[f * N_PARAMS_PER_FILTER + 1].load(), FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                gainDb *= sensitivity_.load();
                gainDb = std::clamp(gainDb, -12.0f, 0.0f);  // Cuts only, max -12dB
                float A = std::pow(10.0f, gainDb / 20.0f);

                // Low shelf transition
                float ratio = freqHz / fc;
                float t = ratio * ratio;
                filterResponse = std::sqrt((A * A + t) / (1.0f + t));
                if (freqHz < fc)
                    filterResponse = std::sqrt(A * A * (1.0f + 1.0f / t) / (A * A / t + 1.0f));
                break;
            }

            case FilterType::HighShelf:
            {
                float fc = denormalizeFreq(currentParams_[f * N_PARAMS_PER_FILTER].load(), SHELF_MIN_FREQ, SHELF_MAX_FREQ);
                float gainDb = denormalizeGain(currentParams_[f * N_PARAMS_PER_FILTER + 1].load(), FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                gainDb *= sensitivity_.load();
                gainDb = std::clamp(gainDb, -12.0f, 0.0f);  // Cuts only, max -12dB
                float A = std::pow(10.0f, gainDb / 20.0f);

                // High shelf transition
                float ratio = freqHz / fc;
                float t = ratio * ratio;
                filterResponse = std::sqrt((1.0f + A * A * t) / (1.0f + t));
                break;
            }

            case FilterType::Peak:
            {
                float fc = denormalizeFreq(currentParams_[f * N_PARAMS_PER_FILTER].load(), PEAK_MIN_FREQ, PEAK_MAX_FREQ);
                float gainDb = denormalizeGain(currentParams_[f * N_PARAMS_PER_FILTER + 1].load(), FILTER_MIN_GAIN, FILTER_MAX_GAIN);
                gainDb *= sensitivity_.load();
                gainDb = std::clamp(gainDb, -12.0f, 0.0f);  // Cuts only, max -12dB
                float q = denormalizeQ(currentParams_[f * N_PARAMS_PER_FILTER + 2].load());

                float A = std::pow(10.0f, gainDb / 20.0f);
                float ratio = freqHz / fc;
                float bw = 1.0f / q;

                // Peaking EQ response
                float x = ratio - 1.0f / ratio;
                float denom = x * x + bw * bw;
                filterResponse = std::sqrt((A * A * bw * bw + x * x) / denom);
                if (denom < 1e-10f) filterResponse = A;
                break;
            }

            default:
                filterResponse = 1.0f;
                break;
        }

        totalResponse *= std::complex<float>(filterResponse, 0.0f);
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

// Phase 3: User control override methods

void DifferentiableBiquadChain::setHPFOverride(float freqHz)
{
    hpfOverride_.store(freqHz);
}

void DifferentiableBiquadChain::setLPFOverride(float freqHz)
{
    lpfOverride_.store(freqHz);
}

void DifferentiableBiquadChain::setOutputGainOffset(float gainDb)
{
    outputGainOffset_.store(gainDb);
}

void DifferentiableBiquadChain::setSmoothingTime(float timeMs)
{
    smoothingTimeMs_ = std::clamp(timeMs, 1.0f, 200.0f);

    // Update smoothed values with new ramp time
    if (sampleRate_ > 0.0)
    {
        float rampSecs = smoothingTimeMs_ / 1000.0f;

        for (auto& sv : smoothedParams_)
            sv.reset(sampleRate_, rampSecs);

        // Gain smoothing uses scaled-up time to prevent pops from large swings
        float gainRampSecs = std::max(smoothingTimeMs_ * 10.0f, 200.0f) / 1000.0f;
        inputGainLinear_.reset(sampleRate_, gainRampSecs);
        outputGainLinear_.reset(sampleRate_, gainRampSecs);
    }
}

void DifferentiableBiquadChain::setSensitivity(float sens)
{
    sensitivity_.store(std::clamp(sens, 0.0f, 1.0f));
}
