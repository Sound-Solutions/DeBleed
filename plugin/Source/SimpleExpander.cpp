#include "SimpleExpander.h"
#include <cmath>

SimpleExpander::SimpleExpander()
{
    updateCoefficients();
}

void SimpleExpander::prepare(double sampleRate)
{
    sampleRate_ = sampleRate;
    updateCoefficients();
    reset();
}

void SimpleExpander::reset()
{
    envelope_ = 0.0f;
    gainReduction_ = 1.0f;
    gainReductionDb_.store(0.0f);
}

void SimpleExpander::updateCoefficients()
{
    if (sampleRate_ <= 0.0)
        return;

    float attackMs = attackMs_.load();
    float releaseMs = releaseMs_.load();

    // Time constant coefficients
    float attackTau = attackMs / 1000.0f;
    float releaseTau = releaseMs / 1000.0f;

    attackCoeff_ = 1.0f - std::exp(-1.0f / (attackTau * static_cast<float>(sampleRate_)));
    releaseCoeff_ = 1.0f - std::exp(-1.0f / (releaseTau * static_cast<float>(sampleRate_)));
}

float SimpleExpander::processSample(float sample, float vadConfidence)
{
    // Get parameters
    float thresholdDb = thresholdDb_.load();
    float ratio = ratio_.load();
    float rangeDb = rangeDb_.load();
    bool vadGating = vadGating_.load();

    // VAD gating: modulate threshold based on vocal confidence
    // When vocal is present (confidence=1), raise the threshold (harder to trigger expansion)
    // When silence (confidence=0), use normal threshold
    // This makes the expander "open up" more when vocals are present
    if (vadGating && vadConfidence > 0.0f)
    {
        // Raise threshold by up to 20dB when vocal is clearly present
        float thresholdBoost = vadConfidence * 20.0f;
        thresholdDb -= thresholdBoost;  // Lower effective threshold = less expansion
    }

    // Compute input level (rectified)
    float inputLevel = std::abs(sample);

    // Envelope follower (peak detector with attack/release)
    float coeff = (inputLevel > envelope_) ? attackCoeff_ : releaseCoeff_;
    envelope_ = coeff * inputLevel + (1.0f - coeff) * envelope_;

    // Convert to dB
    float envelopeDb = 20.0f * std::log10(envelope_ + 1e-10f);

    // Compute gain reduction
    float gainReductionDb = 0.0f;

    if (envelopeDb < thresholdDb)
    {
        // Below threshold - apply expansion
        float belowThreshold = thresholdDb - envelopeDb;  // Positive value

        // Expansion: for every 1dB below threshold, reduce by (1 - 1/ratio) dB more
        // At ratio = 2:1, reduce by 0.5dB per dB below threshold
        // At ratio = inf:1 (gate), reduce by 1dB per dB below threshold
        float expansionFactor = 1.0f - (1.0f / ratio);
        gainReductionDb = -belowThreshold * expansionFactor;

        // Limit to range
        gainReductionDb = std::max(gainReductionDb, rangeDb);
    }

    // Convert to linear gain
    float targetGain = std::pow(10.0f, gainReductionDb / 20.0f);

    // Smooth the gain change (separate from envelope to avoid pumping)
    float gainCoeff = (targetGain < gainReduction_) ? attackCoeff_ : releaseCoeff_;
    gainReduction_ = gainCoeff * targetGain + (1.0f - gainCoeff) * gainReduction_;

    // Clamp
    gainReduction_ = std::clamp(gainReduction_, 0.0f, 1.0f);

    // Update meter
    gainReductionDb_.store(20.0f * std::log10(gainReduction_ + 1e-10f));

    // Apply gain
    return sample * gainReduction_;
}

void SimpleExpander::processBlock(float* audio, const float* vadConfidence, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        float vad = (vadConfidence != nullptr) ? vadConfidence[i] : 0.0f;
        audio[i] = processSample(audio[i], vad);
    }
}

void SimpleExpander::setThresholdDb(float thresholdDb)
{
    thresholdDb_.store(std::clamp(thresholdDb, -80.0f, 0.0f));
}

void SimpleExpander::setRatio(float ratio)
{
    ratio_.store(std::clamp(ratio, 1.0f, 100.0f));  // 100:1 is effectively a gate
}

void SimpleExpander::setAttackMs(float attackMs)
{
    attackMs_.store(std::clamp(attackMs, 0.01f, 100.0f));
    updateCoefficients();
}

void SimpleExpander::setReleaseMs(float releaseMs)
{
    releaseMs_.store(std::clamp(releaseMs, 1.0f, 2000.0f));
    updateCoefficients();
}

void SimpleExpander::setRangeDb(float rangeDb)
{
    rangeDb_.store(std::clamp(rangeDb, -80.0f, 0.0f));
}

void SimpleExpander::setVadGating(bool enabled)
{
    vadGating_.store(enabled);
}
