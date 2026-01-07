#include "DynamicEQ.h"
#include <cmath>

DynamicEQ::DynamicEQ()
{
    // Initialize bands targeting common stage bleed frequencies
    // These cut ONLY when VAD says there's no vocal (silence between phrases)
    //
    // Band 0: 250Hz  - Kick drum fundamental, floor tom bleed
    // Band 1: 500Hz  - Snare body resonance, low guitar bleed
    // Band 2: 1000Hz - Midrange mud, guitar amp bleed
    // Band 3: 3000Hz - Snare crack, cymbal attack transients
    // Band 4: 6000Hz - Cymbal brightness, hi-hat
    // Band 5: 10000Hz - Hi-hat sizzle, cymbal air
    //
    // Moderate Q (1.5-2.5) for broad cuts, gentle max cuts (-6 to -9dB)
    bandParams_[0] = {250.0f,  1.5f, -6.0f};   // Kick/low bleed
    bandParams_[1] = {500.0f,  2.0f, -6.0f};   // Snare body
    bandParams_[2] = {1000.0f, 2.0f, -4.0f};   // Midrange (careful - vocal lives here too)
    bandParams_[3] = {3000.0f, 2.0f, -6.0f};   // Snare crack
    bandParams_[4] = {6000.0f, 1.5f, -8.0f};   // Cymbal brightness
    bandParams_[5] = {10000.0f, 1.2f, -9.0f};  // Hi-hat sizzle (can cut more aggressively)

    // Initialize gains to 0dB (transparent)
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        currentGainDb_[i] = 0.0f;
        targetGainDb_[i] = 0.0f;
        displayGainDb_[i].store(0.0f);
    }
}

void DynamicEQ::prepare(double sampleRate, int /*maxBlockSize*/)
{
    sampleRate_ = sampleRate;

    // Default smoothing: 50ms
    setSmoothingMs(50.0f);

    reset();
}

void DynamicEQ::reset()
{
    // Reset all filter states
    for (auto& bandStates : filterStates_)
    {
        for (auto& state : bandStates)
        {
            state.x1 = state.x2 = 0.0f;
            state.y1 = state.y2 = 0.0f;
        }
    }

    // Reset gains to 0dB
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        currentGainDb_[i] = 0.0f;
        targetGainDb_[i] = 0.0f;
    }
}

DynamicEQ::BiquadCoeffs DynamicEQ::computePeakingCoeffs(float freqHz, float q, float gainDb) const
{
    BiquadCoeffs coeffs;

    // Clamp frequency to valid range
    float nyquist = static_cast<float>(sampleRate_) * 0.5f;
    freqHz = std::clamp(freqHz, 20.0f, nyquist * 0.95f);
    q = std::clamp(q, 0.1f, 20.0f);

    // If gain is essentially 0, return bypass coefficients
    if (std::abs(gainDb) < 0.01f)
    {
        coeffs.b0 = 1.0f;
        coeffs.b1 = 0.0f;
        coeffs.b2 = 0.0f;
        coeffs.a1 = 0.0f;
        coeffs.a2 = 0.0f;
        return coeffs;
    }

    float A = std::pow(10.0f, gainDb / 40.0f);  // sqrt of linear gain
    float w0 = 2.0f * juce::MathConstants<float>::pi * freqHz / static_cast<float>(sampleRate_);
    float cosw0 = std::cos(w0);
    float sinw0 = std::sin(w0);
    float alpha = sinw0 / (2.0f * q);

    float b0 = 1.0f + alpha * A;
    float b1 = -2.0f * cosw0;
    float b2 = 1.0f - alpha * A;
    float a0 = 1.0f + alpha / A;
    float a1 = -2.0f * cosw0;
    float a2 = 1.0f - alpha / A;

    // Normalize
    coeffs.b0 = b0 / a0;
    coeffs.b1 = b1 / a0;
    coeffs.b2 = b2 / a0;
    coeffs.a1 = a1 / a0;
    coeffs.a2 = a2 / a0;

    return coeffs;
}

float DynamicEQ::processBiquad(float input, BiquadState& state, const BiquadCoeffs& coeffs)
{
    float output = coeffs.b0 * input + coeffs.b1 * state.x1 + coeffs.b2 * state.x2
                   - coeffs.a1 * state.y1 - coeffs.a2 * state.y2;

    // Update state
    state.x2 = state.x1;
    state.x1 = input;
    state.y2 = state.y1;
    state.y1 = output;

    // Denormal protection
    if (!std::isfinite(output))
        output = 0.0f;

    return output;
}

float DynamicEQ::processSample(float sample, float vadConfidence)
{
    float output = sample;

    // Process each band
    for (int band = 0; band < NUM_BANDS; ++band)
    {
        const auto& params = bandParams_[band];

        // Compute target gain: lerp between maxCut (when silent) and 0dB (when vocal)
        // vadConfidence = 1 means vocal present -> gain = 0dB
        // vadConfidence = 0 means silence -> gain = maxCut
        targetGainDb_[band] = (1.0f - vadConfidence) * params.maxCutDb;

        // Smooth the gain transition
        currentGainDb_[band] += gainSmoothingCoeff_ * (targetGainDb_[band] - currentGainDb_[band]);

        // Update display value
        displayGainDb_[band].store(currentGainDb_[band]);

        // Compute coefficients for current gain
        auto coeffs = computePeakingCoeffs(params.freqHz, params.q, currentGainDb_[band]);

        // Apply filter (mono - use channel 0)
        output = processBiquad(output, filterStates_[band][0], coeffs);
    }

    return output;
}

void DynamicEQ::processBlock(float* audio, const float* vadConfidence, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        audio[i] = processSample(audio[i], vadConfidence[i]);
    }
}

void DynamicEQ::setBandParams(int bandIndex, const BandParams& params)
{
    if (bandIndex >= 0 && bandIndex < NUM_BANDS)
    {
        bandParams_[bandIndex] = params;
    }
}

DynamicEQ::BandParams DynamicEQ::getBandParams(int bandIndex) const
{
    if (bandIndex >= 0 && bandIndex < NUM_BANDS)
    {
        return bandParams_[bandIndex];
    }
    return {};
}

void DynamicEQ::loadParams(const std::array<BandParams, NUM_BANDS>& params)
{
    bandParams_ = params;
}

void DynamicEQ::setSmoothingMs(float smoothingMs)
{
    if (sampleRate_ > 0.0)
    {
        float tau = smoothingMs / 1000.0f;
        gainSmoothingCoeff_ = 1.0f - std::exp(-1.0f / (tau * static_cast<float>(sampleRate_)));
    }
}

float DynamicEQ::getCurrentGainDb(int bandIndex) const
{
    if (bandIndex >= 0 && bandIndex < NUM_BANDS)
    {
        return displayGainDb_[bandIndex].load();
    }
    return 0.0f;
}

float DynamicEQ::getFrequencyResponse(float freqHz) const
{
    // Compute combined magnitude response of all bands at current gains
    float totalMagnitude = 1.0f;

    for (int band = 0; band < NUM_BANDS; ++band)
    {
        const auto& params = bandParams_[band];
        float gainDb = displayGainDb_[band].load();

        // Simple approximation of peaking filter response
        float ratio = freqHz / params.freqHz;
        float bw = 1.0f / params.q;

        // Distance from center in octaves-ish
        float x = ratio - 1.0f / ratio;
        float denom = x * x + bw * bw;

        float A = std::pow(10.0f, gainDb / 20.0f);

        // Approximate magnitude at this frequency
        float bandMag = std::sqrt((A * A * bw * bw + x * x) / denom);
        if (denom < 1e-10f) bandMag = A;

        totalMagnitude *= bandMag;
    }

    return totalMagnitude;
}
