#pragma once

#include <JuceHeader.h>

/**
 * ArcMeter - Split butterfly meter for gain reduction and signal level
 *
 * Thin glowing arcs and an inset disc, with spring-based smooth animation.
 */
class ArcMeter : public juce::Component
{
public:
    ArcMeter();
    ~ArcMeter() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    void setGainReduction(float grDb);
    void setOutputLevel(float levelDb);
    void setVADConfidence(float confidence);
    void setRange(float rangeDb);

private:
    // Current values
    float gainReductionDb_ = 0.0f;
    float outputLevelDb_ = -60.0f;
    float vadConfidence_ = 0.0f;
    float rangeDb_ = -60.0f;

    // Spring animation state (position + velocity)
    float smoothedGR_ = 0.0f;
    float grVelocity_ = 0.0f;
    float smoothedLevel_ = 0.0f;
    float levelVelocity_ = 0.0f;
    float smoothedVAD_ = 0.0f;
    float vadVelocity_ = 0.0f;

    // Animation timing
    float animTime_ = 0.0f;

    // Visual parameters
    static constexpr float ARC_SWEEP = 135.0f;

    // Spring constants
    static constexpr float SPRING_STIFFNESS = 180.0f;
    static constexpr float SPRING_DAMPING = 12.0f;

    void drawArcWithGlow(juce::Graphics& g, juce::Point<float> centre,
                         float radius, float startAngle, float endAngle,
                         juce::Colour colour);
    void updateSpring(float& position, float& velocity, float target, float dt);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ArcMeter)
};
