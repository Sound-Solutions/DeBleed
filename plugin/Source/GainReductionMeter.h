#pragma once

#include <JuceHeader.h>
#include <atomic>

/**
 * GainReductionMeter - Vertical meter showing average gain reduction.
 *
 * Displays suppression amount in dB with smoothed visual response.
 */
class GainReductionMeter : public juce::Component
{
public:
    GainReductionMeter();
    ~GainReductionMeter() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Set the current reduction level (called from timer)
    void setReductionLevel(float reductionDb);

private:
    float currentReductionDb = 0.0f;
    float smoothedReduction = 0.0f;

    // Meter smoothing coefficients
    static constexpr float METER_ATTACK = 0.3f;
    static constexpr float METER_RELEASE = 0.15f;  // Faster recovery for accuracy

    // Draw dB scale markers
    void drawScale(juce::Graphics& g, juce::Rectangle<int> bounds);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GainReductionMeter)
};
