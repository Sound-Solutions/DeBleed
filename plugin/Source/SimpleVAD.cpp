#include "SimpleVAD.h"
#include <cmath>

SimpleVAD::SimpleVAD()
{
    updateCoefficients();
}

void SimpleVAD::prepare(double sampleRate)
{
    sampleRate_ = sampleRate;
    updateCoefficients();
    reset();
}

void SimpleVAD::reset()
{
    rmsEnvelope_ = 0.0f;
    smoothedConfidence_ = 0.0f;
    currentConfidence_.store(0.0f);
}

void SimpleVAD::updateCoefficients()
{
    if (sampleRate_ <= 0.0)
        return;

    // RMS envelope follower coefficients
    float attackMs = attackMs_.load();
    float releaseMs = releaseMs_.load();

    // Time constant: coeff = 1 - exp(-1 / (tau * sampleRate))
    // where tau = timeMs / 1000
    float attackTau = attackMs / 1000.0f;
    float releaseTau = releaseMs / 1000.0f;

    rmsAttackCoeff_ = 1.0f - std::exp(-1.0f / (attackTau * static_cast<float>(sampleRate_)));
    rmsReleaseCoeff_ = 1.0f - std::exp(-1.0f / (releaseTau * static_cast<float>(sampleRate_)));

    // Confidence smoothing - faster attack, slower release
    // This makes VAD "open" quickly but "close" slowly (safer for vocals)
    float confAttackTau = 0.005f;   // 5ms attack
    float confReleaseTau = 0.050f;  // 50ms release

    confAttackCoeff_ = 1.0f - std::exp(-1.0f / (confAttackTau * static_cast<float>(sampleRate_)));
    confReleaseCoeff_ = 1.0f - std::exp(-1.0f / (confReleaseTau * static_cast<float>(sampleRate_)));
}

float SimpleVAD::processSample(float sample)
{
    // Update RMS envelope (peak-ish detector using squared signal)
    float squared = sample * sample;

    // Attack/release envelope follower
    float coeff = (squared > rmsEnvelope_) ? rmsAttackCoeff_ : rmsReleaseCoeff_;
    rmsEnvelope_ = coeff * squared + (1.0f - coeff) * rmsEnvelope_;

    // Convert to dB
    float rmsDb = 10.0f * std::log10(rmsEnvelope_ + 1e-10f);

    // Soft threshold with sigmoid-like curve
    float threshold = thresholdDb_.load();
    float knee = kneeDb_.load();

    // Compute raw confidence using soft knee
    // confidence = 0.5 + 0.5 * tanh((rmsDb - threshold) / knee)
    float rawConfidence = 0.5f + 0.5f * std::tanh((rmsDb - threshold) / (knee + 0.1f));

    // Clamp to 0-1
    rawConfidence = std::clamp(rawConfidence, 0.0f, 1.0f);

    // Smooth the confidence (fast attack, slow release for vocal safety)
    float confCoeff = (rawConfidence > smoothedConfidence_) ? confAttackCoeff_ : confReleaseCoeff_;
    smoothedConfidence_ = confCoeff * rawConfidence + (1.0f - confCoeff) * smoothedConfidence_;

    // Store for UI
    currentConfidence_.store(smoothedConfidence_);

    return smoothedConfidence_;
}

void SimpleVAD::processBlock(const float* input, float* confidence, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        confidence[i] = processSample(input[i]);
    }
}

void SimpleVAD::setThresholdDb(float thresholdDb)
{
    thresholdDb_.store(std::clamp(thresholdDb, -80.0f, 0.0f));
}

void SimpleVAD::setKneeDb(float kneeDb)
{
    kneeDb_.store(std::clamp(kneeDb, 1.0f, 30.0f));
}

void SimpleVAD::setAttackMs(float attackMs)
{
    attackMs_.store(std::clamp(attackMs, 0.1f, 100.0f));
    updateCoefficients();
}

void SimpleVAD::setReleaseMs(float releaseMs)
{
    releaseMs_.store(std::clamp(releaseMs, 1.0f, 500.0f));
    updateCoefficients();
}
