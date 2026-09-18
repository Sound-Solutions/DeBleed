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
    open_ = false;
    gainReduction_ = 1.0f;
    gainReductionDb_.store(0.0f);
}

void SimpleExpander::updateCoefficients()
{
    if (sampleRate_ <= 0.0)
        return;

    const auto coefficientForMs = [this](float ms)
    {
        return 1.0f - std::exp(-1.0f / (ms * 0.001f * static_cast<float>(sampleRate_)));
    };
    detectorRiseCoeff_ = coefficientForMs(0.1f);
    detectorFallCoeff_ = coefficientForMs(20.0f);
    openCoeff_ = coefficientForMs(openMs_.load());
    closeCoeff_ = coefficientForMs(closeMs_.load());
}

float SimpleExpander::processSample(float sample, float vadConfidence)
{
    return sample * computeGain(sample, vadConfidence);
}

float SimpleExpander::computeGain(float sidechainSample, float vadConfidence)
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

    // Fixed fast peak detector; the knobs time the gain, not the level
    const float inputLevel = std::abs(sidechainSample);
    const float detectorCoeff = (inputLevel > envelope_) ? detectorRiseCoeff_ : detectorFallCoeff_;
    envelope_ = detectorCoeff * inputLevel + (1.0f - detectorCoeff) * envelope_;
    const float envelopeDb = 20.0f * std::log10(envelope_ + 1e-10f);

    // Hysteresis: open at the threshold, close only once the level is 3 dB under it
    if (open_)
        open_ = envelopeDb >= thresholdDb - hysteresisDb;
    else
        open_ = envelopeDb >= thresholdDb;

    // While open the gain sits at unity; closed, expand against the open threshold so the
    // curve meets 0 dB exactly where the gate opens
    float gainReductionDb = 0.0f;
    if (!open_ && envelopeDb < thresholdDb)
    {
        const float belowThreshold = thresholdDb - envelopeDb;
        const float expansionFactor = 1.0f - (1.0f / ratio);
        gainReductionDb = std::max(-belowThreshold * expansionFactor, rangeDb);
    }

    const float targetGain = std::pow(10.0f, gainReductionDb / 20.0f);

    // OPEN times the gain coming up, CLOSE times it going down
    const float gainCoeff = (targetGain > gainReduction_) ? openCoeff_ : closeCoeff_;
    gainReduction_ = gainCoeff * targetGain + (1.0f - gainCoeff) * gainReduction_;

    // Clamp
    gainReduction_ = std::clamp(gainReduction_, 0.0f, 1.0f);

    // Update meter
    gainReductionDb_.store(20.0f * std::log10(gainReduction_ + 1e-10f));

    return gainReduction_;
}

void SimpleExpander::applyGains(float* audio, const float* gains, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
        audio[i] *= gains[i];
}

void SimpleExpander::setThresholdDb(float thresholdDb)
{
    thresholdDb_.store(std::clamp(thresholdDb, -80.0f, 0.0f));
}

void SimpleExpander::setRatio(float ratio)
{
    ratio_.store(std::clamp(ratio, 1.0f, 100.0f));  // 100:1 is effectively a gate
}

void SimpleExpander::setOpenMs(float openMs)
{
    openMs_.store(std::clamp(openMs, 0.01f, 100.0f));
    updateCoefficients();
}

void SimpleExpander::setCloseMs(float closeMs)
{
    closeMs_.store(std::clamp(closeMs, 1.0f, 2000.0f));
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
