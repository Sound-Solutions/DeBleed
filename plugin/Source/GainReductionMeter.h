#pragma once

#include <JuceHeader.h>

/**
 * GainReductionMeter - Shows expander gain reduction.
 *
 * - Purple fill from top: amount of gain reduction
 */
class GainReductionMeter : public juce::Component
{
public:
    GainReductionMeter();
    ~GainReductionMeter() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Set the current reduction level in dB (negative values)
    void setReductionLevel(float reductionDb);

private:
    float currentReductionDb = 0.0f;
    float smoothedReduction = 0.0f;

    static constexpr float METER_ATTACK = 0.3f;
    static constexpr float METER_RELEASE = 0.15f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GainReductionMeter)
};
