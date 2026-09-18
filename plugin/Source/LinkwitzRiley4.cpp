#include "LinkwitzRiley4.h"
#include <algorithm>
#include <cmath>

double LinkwitzRiley4::Biquad::process(double input)
{
    const double output = b0 * input + z1;
    z1 = b1 * input - a1 * output + z2;
    z2 = b2 * input - a2 * output;
    return output;
}

void LinkwitzRiley4::prepare(double sampleRate, double cutoffHz)
{
    sampleRate = std::max(1.0, sampleRate);
    const double omega = 2.0 * std::acos(-1.0)
                       * std::clamp(cutoffHz, 0.01, sampleRate * 0.45) / sampleRate;
    const double cosine = std::cos(omega);
    const double alpha = std::sin(omega) / std::sqrt(2.0);
    const double a0 = 1.0 + alpha;
    for (int stage = 0; stage < 2; ++stage)
    {
        auto& low = low_[stage];
        auto& high = high_[stage];
        low.b0 = low.b2 = (1.0 - cosine) / (2.0 * a0);
        low.b1 = 2.0 * low.b0;
        high.b0 = high.b2 = (1.0 + cosine) / (2.0 * a0);
        high.b1 = -2.0 * high.b0;
        low.a1 = high.a1 = -2.0 * cosine / a0;
        low.a2 = high.a2 = (1.0 - alpha) / a0;
    }
    reset();
}

void LinkwitzRiley4::reset()
{
    for (auto& filter : low_)
        filter.z1 = filter.z2 = 0.0;
    for (auto& filter : high_)
        filter.z1 = filter.z2 = 0.0;
}

void LinkwitzRiley4::processSplit(float input, float& low, float& high)
{
    low = static_cast<float>(low_[1].process(low_[0].process(input)));
    high = processHigh(input);
}

float LinkwitzRiley4::processHigh(float input)
{
    return static_cast<float>(high_[1].process(high_[0].process(input)));
}
