#pragma once

#include <JuceHeader.h>
#include <array>
#include <cmath>

/**
 * EnvelopeFollower - Single-band envelope follower with attack/release ballistics.
 *
 * Used to smooth gain changes from the neural analysis to prevent clicks
 * and provide natural-sounding dynamics.
 */
class EnvelopeFollower
{
public:
    EnvelopeFollower() = default;
    ~EnvelopeFollower() = default;

    /**
     * Initialize the envelope follower.
     * @param sampleRate Sample rate in Hz
     */
    void prepare(double sampleRate);

    /**
     * Reset the envelope state.
     */
    void reset();

    /**
     * Set attack time.
     * @param attackMs Attack time in milliseconds
     */
    void setAttack(float attackMs);

    /**
     * Set release time.
     * @param releaseMs Release time in milliseconds
     */
    void setRelease(float releaseMs);

    /**
     * Process a single target value and return smoothed output.
     * Call once per block with the target gain value.
     * @param target Target gain value (0.0 to 1.0)
     * @return Smoothed gain value
     */
    float process(float target);

    /**
     * Process multiple samples towards a target.
     * Updates internal state numSamples times.
     * @param target Target gain value
     * @param numSamples Number of samples to process
     * @return Final smoothed value after all samples
     */
    float processBlock(float target, int numSamples);

    /**
     * Get current envelope value without processing.
     */
    float getCurrentValue() const { return currentValue; }

private:
    void updateCoefficients();

    double sampleRate = 48000.0;
    float attackMs = 5.0f;
    float releaseMs = 50.0f;

    float attackCoeff = 0.0f;
    float releaseCoeff = 0.0f;

    float currentValue = 1.0f;  // Start at unity
};


/**
 * EnvelopeFollowerBank - Collection of 192 envelope followers for the filter bank.
 * Hybrid 32+160 topology: 32 low bands (20-500Hz) + 160 high bands (500Hz-20kHz)
 */
class EnvelopeFollowerBank
{
public:
    static constexpr int NUM_BANDS = 192;

    EnvelopeFollowerBank() = default;
    ~EnvelopeFollowerBank() = default;

    /**
     * Initialize all envelope followers.
     */
    void prepare(double sampleRate);

    /**
     * Reset all envelope states.
     */
    void reset();

    /**
     * Set attack time for all bands.
     */
    void setAttack(float attackMs);

    /**
     * Set release time for all bands.
     */
    void setRelease(float releaseMs);

    /**
     * Process a block of target gains and return smoothed gains.
     * @param targetGains Input target gains (64 values)
     * @param smoothedGains Output smoothed gains (64 values)
     * @param numSamples Number of samples in the block
     */
    void processBlock(const std::array<float, NUM_BANDS>& targetGains,
                      std::array<float, NUM_BANDS>& smoothedGains,
                      int numSamples);

    /**
     * Get current smoothed values for all bands.
     */
    const std::array<float, NUM_BANDS>& getCurrentValues() const { return currentValues; }

private:
    std::array<EnvelopeFollower, NUM_BANDS> followers;
    std::array<float, NUM_BANDS> currentValues;
};
