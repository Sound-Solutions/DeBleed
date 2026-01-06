#include "ArcMeter.h"

ArcMeter::ArcMeter()
{
}

void ArcMeter::updateSpring(float& position, float& velocity, float target, float dt)
{
    // Spring physics for smooth, organic movement
    float displacement = target - position;
    float springForce = SPRING_STIFFNESS * displacement;
    float dampingForce = -SPRING_DAMPING * velocity;
    float acceleration = springForce + dampingForce;

    velocity += acceleration * dt;
    position += velocity * dt;

    // Clamp to prevent overshoot beyond reasonable bounds
    position = std::clamp(position, 0.0f, 1.0f);
}

void ArcMeter::drawArcWithGlow(juce::Graphics& g, juce::Point<float> center,
                                float radius, float innerRadius,
                                float startAngle, float endAngle,
                                juce::Colour baseColor, juce::Colour glowColor,
                                float intensity)
{
    if (std::abs(endAngle - startAngle) < 0.001f)
        return;

    // Create the main arc path
    juce::Path arc;
    arc.addCentredArc(center.x, center.y, radius, radius,
                      0.0f, startAngle, endAngle, true);
    arc.addCentredArc(center.x, center.y, innerRadius, innerRadius,
                      0.0f, endAngle, startAngle, false);
    arc.closeSubPath();

    // Fill with the glow color (matches the label text color)
    g.setColour(glowColor);
    g.fillPath(arc);
}

void ArcMeter::drawVADRing(juce::Graphics& g, juce::Point<float> center, float confidence)
{
    float ringRadius = 18.0f;
    float ringThickness = 3.0f;

    // Pulsing effect based on animation time
    float pulse = 0.5f + 0.5f * std::sin(animTime_ * 4.0f);
    float pulseScale = 1.0f + pulse * 0.08f * confidence;
    float actualRadius = ringRadius * pulseScale;

    // Background ring
    g.setColour(vadRingInactive_);
    g.drawEllipse(center.x - actualRadius, center.y - actualRadius,
                  actualRadius * 2, actualRadius * 2, ringThickness);

    // Active ring overlay (partial arc based on confidence)
    if (confidence > 0.01f)
    {
        // Draw full ring with intensity based on confidence
        juce::Colour ringColor = vadRingActive_.withAlpha(confidence * 0.9f);

        // Outer glow
        for (int i = 2; i >= 0; --i)
        {
            float glowAlpha = confidence * 0.2f * (1.0f - i * 0.25f);
            float glowRadius = actualRadius + (i + 1) * 2.0f;
            g.setColour(vadRingActive_.withAlpha(glowAlpha));
            g.drawEllipse(center.x - glowRadius, center.y - glowRadius,
                         glowRadius * 2, glowRadius * 2, 2.0f);
        }

        // Main ring
        g.setColour(ringColor);
        g.drawEllipse(center.x - actualRadius, center.y - actualRadius,
                      actualRadius * 2, actualRadius * 2, ringThickness);

        // Inner bright core when high confidence
        if (confidence > 0.5f)
        {
            float coreAlpha = (confidence - 0.5f) * 2.0f * pulse;
            float coreRadius = actualRadius * 0.4f;
            g.setColour(vadRingActive_.withAlpha(coreAlpha * 0.6f));
            g.fillEllipse(center.x - coreRadius, center.y - coreRadius,
                         coreRadius * 2, coreRadius * 2);
        }
    }
}

void ArcMeter::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat().reduced(4.0f);
    float size = std::min(bounds.getWidth(), bounds.getHeight());
    auto center = bounds.getCentre();
    float radius = size * 0.44f;
    float innerRadius = radius - ARC_THICKNESS;

    // Update animation time
    animTime_ += 0.016f;  // ~60fps
    if (animTime_ > 1000.0f) animTime_ = 0.0f;

    // Update spring animations
    float dt = 0.016f;
    float normalizedGR = std::clamp(-gainReductionDb_ / -rangeDb_, 0.0f, 1.0f);
    float normalizedLevel = std::clamp((outputLevelDb_ + 60.0f) / 60.0f, 0.0f, 1.0f);

    updateSpring(smoothedGR_, grVelocity_, normalizedGR, dt);
    updateSpring(smoothedLevel_, levelVelocity_, normalizedLevel, dt);
    updateSpring(smoothedVAD_, vadVelocity_, vadConfidence_, dt);

    // =========================================================================
    // Background with subtle radial gradient
    // =========================================================================
    juce::ColourGradient bgGradient(
        juce::Colour(0xff151518), center.x, center.y,
        bgColor_, center.x, center.y + radius * 1.2f,
        true
    );
    g.setGradientFill(bgGradient);
    g.fillRect(getLocalBounds());

    // Subtle vignette effect
    juce::ColourGradient vignette(
        juce::Colours::transparentBlack, center.x, center.y,
        juce::Colour(0x20000000), 0.0f, 0.0f,
        true
    );
    vignette.addColour(0.7, juce::Colours::transparentBlack);
    g.setGradientFill(vignette);
    g.fillRect(getLocalBounds());

    // Convert sweep to radians
    float sweepRad = juce::degreesToRadians(ARC_SWEEP);
    float topAngle = 0.0f;  // 0 = 12 o'clock (top center)

    // =========================================================================
    // Draw arc backgrounds with subtle inner shadow
    // =========================================================================

    // Left arc background
    juce::Path leftArcBg;
    leftArcBg.addCentredArc(center.x, center.y, radius, radius,
                            0.0f, topAngle, topAngle - sweepRad, true);
    leftArcBg.addCentredArc(center.x, center.y, innerRadius, innerRadius,
                            0.0f, topAngle - sweepRad, topAngle, false);
    leftArcBg.closeSubPath();

    // Inner shadow gradient
    juce::ColourGradient leftBgGrad(
        arcBgColor_.darker(0.3f), center.x, center.y,
        arcBgColor_, center.x - radius * 0.5f, center.y + radius * 0.5f,
        true
    );
    g.setGradientFill(leftBgGrad);
    g.fillPath(leftArcBg);

    // Right arc background
    juce::Path rightArcBg;
    rightArcBg.addCentredArc(center.x, center.y, radius, radius,
                             0.0f, topAngle, topAngle + sweepRad, true);
    rightArcBg.addCentredArc(center.x, center.y, innerRadius, innerRadius,
                             0.0f, topAngle + sweepRad, topAngle, false);
    rightArcBg.closeSubPath();

    juce::ColourGradient rightBgGrad(
        arcBgColor_.darker(0.3f), center.x, center.y,
        arcBgColor_, center.x + radius * 0.5f, center.y + radius * 0.5f,
        true
    );
    g.setGradientFill(rightBgGrad);
    g.fillPath(rightArcBg);

    // =========================================================================
    // Draw active arcs with glow
    // =========================================================================

    // Left arc - Gain Reduction (Orange)
    if (smoothedGR_ > 0.005f)
    {
        float grSweep = sweepRad * smoothedGR_;
        drawArcWithGlow(g, center, radius, innerRadius,
                        topAngle, topAngle - grSweep,
                        grColor_, grColor_, smoothedGR_);
    }

    // Right arc - Signal Level (Cyan)
    if (smoothedLevel_ > 0.005f)
    {
        float levelSweep = sweepRad * smoothedLevel_;
        drawArcWithGlow(g, center, radius, innerRadius,
                        topAngle, topAngle + levelSweep,
                        signalColor_, signalColor_, smoothedLevel_);
    }

    // =========================================================================
    // Center highlight ring
    // =========================================================================
    float centerRingRadius = innerRadius - 8.0f;
    g.setColour(juce::Colour(0x15ffffff));
    g.drawEllipse(center.x - centerRingRadius, center.y - centerRingRadius,
                  centerRingRadius * 2, centerRingRadius * 2, 1.0f);

    // =========================================================================
    // Labels for each side
    // =========================================================================
    g.setFont(juce::Font(10.0f).boldened());

    // GR label (orange)
    g.setColour(grColor_);
    float labelY = center.y - radius * 0.25f;
    g.drawText("GR",
               static_cast<int>(center.x - radius - 28),
               static_cast<int>(labelY),
               24, 14, juce::Justification::centredRight);

    // SIGNAL label (cyan)
    g.setColour(signalColor_);
    g.drawText("SIGNAL",
               static_cast<int>(center.x + radius + 4),
               static_cast<int>(labelY),
               42, 14, juce::Justification::centredLeft);

    // =========================================================================
    // Center text - GR value with glow effect
    // =========================================================================
    juce::String grText;
    if (gainReductionDb_ > -0.1f)
        grText = "0.0";
    else
        grText = juce::String(gainReductionDb_, 1);

    // Text glow when GR is active
    if (smoothedGR_ > 0.1f)
    {
        g.setColour(grColor_.withAlpha(smoothedGR_ * 0.3f));
        g.setFont(juce::Font(28.0f).boldened());
        auto glowBounds = juce::Rectangle<float>(center.x - 47, center.y - 22, 94, 34);
        g.drawText(grText, glowBounds, juce::Justification::centred);
    }

    // Main text
    g.setColour(textColor_);
    g.setFont(juce::Font(26.0f).boldened());
    auto textBounds = juce::Rectangle<float>(center.x - 45, center.y - 20, 90, 30);
    g.drawText(grText, textBounds, juce::Justification::centred);

    // "dB" label
    g.setFont(juce::Font(11.0f));
    g.setColour(textColor_.withAlpha(0.5f));
    auto dbBounds = juce::Rectangle<float>(center.x - 15, center.y + 10, 30, 16);
    g.drawText("dB", dbBounds, juce::Justification::centred);

    // =========================================================================
    // Tick marks
    // =========================================================================
    float tickInner = radius + 4.0f;
    float tickOuter = radius + 10.0f;

    // Top tick (brighter)
    g.setColour(textColor_.withAlpha(0.4f));
    g.drawLine(center.x, center.y - tickInner, center.x, center.y - tickOuter, 2.0f);

    // Calculate end tick positions
    float leftEndAngle = -sweepRad - juce::MathConstants<float>::halfPi;
    float rightEndAngle = sweepRad - juce::MathConstants<float>::halfPi;

    // Left bottom tick
    g.setColour(textColor_.withAlpha(0.25f));
    float lx1 = center.x + tickInner * std::cos(leftEndAngle);
    float ly1 = center.y + tickInner * std::sin(leftEndAngle);
    float lx2 = center.x + tickOuter * std::cos(leftEndAngle);
    float ly2 = center.y + tickOuter * std::sin(leftEndAngle);
    g.drawLine(lx1, ly1, lx2, ly2, 1.5f);

    // Right bottom tick
    float rx1 = center.x + tickInner * std::cos(rightEndAngle);
    float ry1 = center.y + tickInner * std::sin(rightEndAngle);
    float rx2 = center.x + tickOuter * std::cos(rightEndAngle);
    float ry2 = center.y + tickOuter * std::sin(rightEndAngle);
    g.drawLine(rx1, ry1, rx2, ry2, 1.5f);
}

void ArcMeter::resized()
{
}

void ArcMeter::setGainReduction(float grDb)
{
    if (std::abs(gainReductionDb_ - grDb) > 0.05f)
    {
        gainReductionDb_ = grDb;
        repaint();
    }
}

void ArcMeter::setOutputLevel(float levelDb)
{
    if (std::abs(outputLevelDb_ - levelDb) > 0.3f)
    {
        outputLevelDb_ = levelDb;
        repaint();
    }
}

void ArcMeter::setVADConfidence(float confidence)
{
    float clamped = std::clamp(confidence, 0.0f, 1.0f);
    if (std::abs(vadConfidence_ - clamped) > 0.01f)
    {
        vadConfidence_ = clamped;
        repaint();
    }
}

void ArcMeter::setRange(float rangeDb)
{
    rangeDb_ = std::min(rangeDb, -1.0f);
}
