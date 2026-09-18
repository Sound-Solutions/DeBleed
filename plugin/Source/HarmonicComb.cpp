#include "HarmonicComb.h"
#include <algorithm>
#include <cmath>

void HarmonicComb::prepare(double sampleRate)
{
    sampleRate_ = std::max(1.0, sampleRate);
    delay_.resize(static_cast<int>(std::ceil(sampleRate_ / 60.0)) + 4);
    riseCoeff_ = static_cast<float>(1.0 - std::exp(-1.0 / (0.025 * sampleRate_)));
    fallCoeff_ = periodCoeff_ = static_cast<float>(1.0 - std::exp(-1.0 / (0.005 * sampleRate_)));
    setDepthDb(9.0f);
    reset();
}

void HarmonicComb::reset()
{
    std::fill(delay_.begin(), delay_.end(), 0.0f);
    writeIndex_ = 0;
    envelope_ = feedbackGain_ = period_ = targetPeriod_ = 0.0f;
    confident_ = vocalPresent_ = forced_ = false;
}

void HarmonicComb::setDepthDb(float depthDb)
{
    const float ratio = std::pow(10.0f, -std::clamp(depthDb, 6.0f, 12.0f) / 20.0f);
    depthGain_ = (1.0f - ratio) / (1.0f + ratio);
}

void HarmonicComb::setTarget(float periodSamples, bool confident)
{
    const double frequency = periodSamples > 0.0f ? sampleRate_ / periodSamples : 0.0;
    confident_ = confident && std::isfinite(periodSamples) && frequency >= 80.0 && frequency <= 1000.0;
    if (confident_)
    {
        targetPeriod_ = periodSamples;
        // No preceding pitch to slew from on the first acquisition (the comb is dry).
        if (period_ == 0.0f)
            period_ = targetPeriod_;
    }
}

void HarmonicComb::setForcedState(float periodSamples, float gain)
{
    if (delay_.empty())
        return;
    forced_ = true;
    period_ = targetPeriod_ = std::clamp(periodSamples, 1.0f, static_cast<float>(delay_.size() - 2));
    envelope_ = 1.0f;
    feedbackGain_ = std::clamp(gain, 0.0f, 0.999f);
}

float HarmonicComb::process(float input)
{
    if (delay_.empty())
        return input;
    if (!forced_)
    {
        const bool active = confident_ && vocalPresent_;
        const float coefficient = active ? riseCoeff_ : fallCoeff_;
        envelope_ += coefficient * ((active ? 1.0f : 0.0f) - envelope_);
        if (active)
            period_ += periodCoeff_ * (targetPeriod_ - period_);
        feedbackGain_ = depthGain_ * envelope_;
    }

    const int length = static_cast<int>(delay_.size());
    const float delaySamples = std::clamp(period_, 1.0f, static_cast<float>(length - 2));
    const int whole = static_cast<int>(delaySamples);
    const float fraction = delaySamples - static_cast<float>(whole);
    const int newer = (writeIndex_ - whole + length) % length;
    const int older = (newer + length - 1) % length;
    const float delayed = delay_[newer] + fraction * (delay_[older] - delay_[newer]);
    const float output = (1.0f - feedbackGain_) * input + feedbackGain_ * delayed;
    delay_[writeIndex_] = output;
    writeIndex_ = (writeIndex_ + 1) % length;
    return output;
}
