#pragma once

#include <JuceHeader.h>
#include <atomic>

/**
 * GainReductionMeter - Vertical meter showing gate status.
 *
 * Shows:
 * - Input level (cyan bar)
 * - Threshold line (orange horizontal line)
 * - Gain reduction fill (purple, fills down from threshold when gating)
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

    // Set gate visualization data
    void setGateInfo(float detectedLevelDb, float thresholdDb, bool gateOpen);

private:
    float currentReductionDb = 0.0f;
    float smoothedReduction = 0.0f;

    // Gate visualization
    float detectedLevel = -60.0f;
    float smoothedLevel = -60.0f;
    float threshold = -40.0f;
    bool isGateOpen = true;

    // Meter smoothing coefficients
    static constexpr float METER_ATTACK = 0.3f;
    static constexpr float METER_RELEASE = 0.15f;
    static constexpr float LEVEL_ATTACK = 0.4f;
    static constexpr float LEVEL_RELEASE = 0.1f;

    // Draw dB scale markers
    void drawScale(juce::Graphics& g, juce::Rectangle<int> bounds);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GainReductionMeter)
};
