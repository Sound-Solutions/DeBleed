#include "EnvelopeFollower.h"

// ============================================================================
// EnvelopeFollower (single band)
// ============================================================================

void EnvelopeFollower::prepare(double newSampleRate)
{
    sampleRate = newSampleRate;
    updateCoefficients();
    reset();
}

void EnvelopeFollower::reset()
{
    currentValue = 1.0f;  // Start at unity (no attenuation)
}

void EnvelopeFollower::setAttack(float newAttackMs)
{
    attackMs = juce::jlimit(0.1f, 1000.0f, newAttackMs);
    updateCoefficients();
}

void EnvelopeFollower::setRelease(float newReleaseMs)
{
    releaseMs = juce::jlimit(1.0f, 5000.0f, newReleaseMs);
    updateCoefficients();
}

float EnvelopeFollower::process(float target)
{
    // Choose coefficient based on direction
    // Attack = gain going DOWN (attenuation increasing)
    // Release = gain going UP (attenuation decreasing)
    float coeff = (target < currentValue) ? attackCoeff : releaseCoeff;

    // First-order IIR smoothing
    currentValue += coeff * (target - currentValue);

    return currentValue;
}

float EnvelopeFollower::processBlock(float target, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        process(target);
    }
    return currentValue;
}

void EnvelopeFollower::updateCoefficients()
{
    // Convert time constants to per-sample coefficients
    // Using 1 - exp(-1 / (time * sampleRate)) for smooth exponential decay
    // Time constant = time to reach ~63% of target

    float attackSamples = (attackMs / 1000.0f) * static_cast<float>(sampleRate);
    float releaseSamples = (releaseMs / 1000.0f) * static_cast<float>(sampleRate);

    // Prevent division by zero
    attackSamples = std::max(1.0f, attackSamples);
    releaseSamples = std::max(1.0f, releaseSamples);

    attackCoeff = 1.0f - std::exp(-1.0f / attackSamples);
    releaseCoeff = 1.0f - std::exp(-1.0f / releaseSamples);
}


// ============================================================================
// EnvelopeFollowerBank (64 bands)
// ============================================================================

void EnvelopeFollowerBank::prepare(double sampleRate)
{
    currentValues.fill(1.0f);

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        followers[i].prepare(sampleRate);
    }
}

void EnvelopeFollowerBank::reset()
{
    currentValues.fill(1.0f);

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        followers[i].reset();
    }
}

void EnvelopeFollowerBank::setAttack(float attackMs)
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        followers[i].setAttack(attackMs);
    }
}

void EnvelopeFollowerBank::setRelease(float releaseMs)
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        followers[i].setRelease(releaseMs);
    }
}

void EnvelopeFollowerBank::processBlock(const std::array<float, NUM_BANDS>& targetGains,
                                         std::array<float, NUM_BANDS>& smoothedGains,
                                         int numSamples)
{
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        smoothedGains[i] = followers[i].processBlock(targetGains[i], numSamples);
        currentValues[i] = smoothedGains[i];
    }
}
