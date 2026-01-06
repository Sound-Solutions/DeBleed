#pragma once

#include <JuceHeader.h>

/**
 * ArcMeter - Split butterfly meter for gain reduction and signal level
 *
 * Enhanced visual design with:
 * - Gradient arcs with outer glow
 * - Pulsing VAD ring with confidence intensity
 * - Spring-based smooth animation
 * - Subtle background depth
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
    static constexpr float ARC_THICKNESS = 14.0f;
    static constexpr float ARC_SWEEP = 135.0f;
    static constexpr float GLOW_RADIUS = 8.0f;

    // Spring constants
    static constexpr float SPRING_STIFFNESS = 180.0f;
    static constexpr float SPRING_DAMPING = 12.0f;

    // Base colors
    juce::Colour bgColor_{0xff0d0d0f};
    juce::Colour arcBgColor_{0xff1a1a1e};
    juce::Colour textColor_{0xffffffff};

    // GR arc - Orange (matches expander knobs)
    juce::Colour grColor_{0xffff8800};

    // Signal arc - Cyan (matches output knobs)
    juce::Colour signalColor_{0xff00d4ff};

    // VAD ring colors
    juce::Colour vadRingActive_{0xff00ffaa};
    juce::Colour vadRingInactive_{0xff333333};

    // Helper methods
    void drawArcWithGlow(juce::Graphics& g, juce::Point<float> center,
                         float radius, float innerRadius,
                         float startAngle, float endAngle,
                         juce::Colour baseColor, juce::Colour glowColor,
                         float intensity);
    void drawVADRing(juce::Graphics& g, juce::Point<float> center, float confidence);
    void updateSpring(float& position, float& velocity, float target, float dt);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ArcMeter)
};
