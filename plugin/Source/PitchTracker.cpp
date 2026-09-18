#include "PitchTracker.h"
#include <algorithm>
#include <cmath>

double PitchTracker::Biquad::process(double input)
{
    const double output = b0 * input + z1;
    z1 = b1 * input - a1 * output + z2;
    z2 = b2 * input - a2 * output;
    return output;
}

void PitchTracker::prepare(double sampleRate)
{
    sampleRate_ = std::max(1.0, sampleRate);
    decimation_ = std::max(1, static_cast<int>(std::round(sampleRate_ / 24000.0)));
    const double decimatedRate = sampleRate_ / decimation_;
    minLag_ = std::clamp(static_cast<int>(std::floor(decimatedRate / 1000.0)), 1, windowSize - 3);
    maxLag_ = std::clamp(static_cast<int>(std::ceil(decimatedRate / 80.0)), minLag_, windowSize - 2);

    const double omega = 2.0 * std::acos(-1.0) * std::min(8000.0, sampleRate_ * 0.45) / sampleRate_;
    // The two sections of a fourth-order Butterworth low-pass.
    const double sectionQ[] = { 0.541196100146197, 1.306562964876377 };
    for (int stage = 0; stage < 2; ++stage)
    {
        auto& filter = lowpass_[stage];
        const double alpha = std::sin(omega) / (2.0 * sectionQ[stage]);
        const double a0 = 1.0 + alpha;
        filter.b0 = filter.b2 = (1.0 - std::cos(omega)) / (2.0 * a0);
        filter.b1 = 2.0 * filter.b0;
        filter.a1 = -2.0 * std::cos(omega) / a0;
        filter.a2 = (1.0 - alpha) / a0;
    }
    ring_.resize(windowSize);
    window_.resize(windowSize);
    nsdf_.resize(windowSize);
    reset();
}

void PitchTracker::reset()
{
    std::fill(ring_.begin(), ring_.end(), 0.0f);
    std::fill(window_.begin(), window_.end(), 0.0f);
    std::fill(nsdf_.begin(), nsdf_.end(), 0.0f);
    for (auto& filter : lowpass_)
        filter.z1 = filter.z2 = 0.0;
    decimationCounter_ = writeIndex_ = samplesFilled_ = updateCounter_ = 0;
    periodSamples_ = confidence_ = 0.0f;
}

void PitchTracker::pushSample(float input)
{
    if (ring_.empty())
        return;
    const float filtered = static_cast<float>(lowpass_[1].process(lowpass_[0].process(input)));
    if (++decimationCounter_ < decimation_)
        return;
    decimationCounter_ = 0;
    ring_[writeIndex_] = filtered;
    writeIndex_ = (writeIndex_ + 1) % windowSize;
    samplesFilled_ = std::min(samplesFilled_ + 1, windowSize);
    if (++updateCounter_ >= 256)
    {
        updateCounter_ = 0;
        if (samplesFilled_ == windowSize)
            analyse();
    }
}

void PitchTracker::analyse()
{
    double energy = 0.0;
    for (int i = 0; i < windowSize; ++i)
    {
        window_[i] = ring_[(writeIndex_ + i) % windowSize];
        energy += static_cast<double>(window_[i]) * window_[i];
    }
    periodSamples_ = confidence_ = 0.0f;
    if (energy < 1.0e-10)
        return;

    for (int lag = minLag_ - 1; lag <= maxLag_ + 1; ++lag)
    {
        double correlation = 0.0, denominator = 0.0;
        for (int i = 0; i < windowSize - lag; ++i)
        {
            const double a = window_[i], b = window_[i + lag];
            correlation += a * b;
            denominator += a * a + b * b;
        }
        nsdf_[lag] = denominator > 1.0e-12
                   ? static_cast<float>(2.0 * correlation / denominator) : 0.0f;
    }

    float globalPeak = 0.0f;
    for (int lag = minLag_; lag <= maxLag_; ++lag)
        if (nsdf_[lag] > nsdf_[lag - 1] && nsdf_[lag] >= nsdf_[lag + 1])
            globalPeak = std::max(globalPeak, nsdf_[lag]);
    if (globalPeak <= 0.0f)
        return;

    for (int lag = minLag_; lag <= maxLag_; ++lag)
    {
        const float left = nsdf_[lag - 1], peak = nsdf_[lag], right = nsdf_[lag + 1];
        // 0.95 (McLeod's k): a sub-octave miss still passes every true harmonic, an
        // octave-up miss notches the odd ones, so lean toward the global peak.
        if (peak <= left || peak < right || peak < 0.95f * globalPeak)
            continue;
        const float curvature = left - 2.0f * peak + right;
        const float offset = std::abs(curvature) > 1.0e-12f
                           ? std::clamp(0.5f * (left - right) / curvature, -0.5f, 0.5f) : 0.0f;
        periodSamples_ = (static_cast<float>(lag) + offset) * static_cast<float>(decimation_);
        confidence_ = std::clamp(peak - 0.25f * (left - right) * offset, 0.0f, 1.0f);
        return;
    }
}

float PitchTracker::getFrequencyHz() const
{
    return periodSamples_ > 0.0f ? static_cast<float>(sampleRate_ / periodSamples_) : 0.0f;
}
