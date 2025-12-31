#pragma once

#include <JuceHeader.h>

/**
 * ConfidenceMeter - Shows neural confidence level and threshold.
 *
 * - Cyan bar: neural confidence (0-100%)
 * - Orange line: threshold setting from knob
 */
class ConfidenceMeter : public juce::Component
{
public:
    ConfidenceMeter();
    ~ConfidenceMeter() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Set confidence level (0.0 - 1.0)
    void setConfidence(float conf);

    // Set threshold directly from parameter (0.0 - 1.0)
    void setThreshold(float thresh);

    // Set gate open state
    void setGateOpen(bool open);

private:
    float confidence = 1.0f;
    float smoothedConfidence = 1.0f;
    float threshold = 0.5f;
    bool isGateOpen = true;

    static constexpr float ATTACK_COEFF = 0.4f;
    static constexpr float RELEASE_COEFF = 0.1f;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConfidenceMeter)
};
