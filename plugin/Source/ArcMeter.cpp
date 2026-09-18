#include "ArcMeter.h"
#include "DeBleedLookAndFeel.h"

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

void ArcMeter::drawArcWithGlow(juce::Graphics& g, juce::Point<float> centre,
                                float radius, float startAngle, float endAngle,
                                juce::Colour colour)
{
    if (std::abs(endAngle - startAngle) < 0.001f)
        return;

    juce::Path arc;
    arc.addCentredArc(centre.x, centre.y, radius, radius, 0.0f, startAngle, endAngle, true);
    g.setColour(colour.withAlpha(0.35f));
    g.strokePath(arc, juce::PathStrokeType(5.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
    g.setColour(colour);
    g.strokePath(arc, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
}

void ArcMeter::paint(juce::Graphics& g)
{
    const juce::Point<float> centre(134.0f, 144.0f);
    const float radius = 104.0f;

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

    const float sweepRad = juce::degreesToRadians(ARC_SWEEP);
    juce::Path tracks;
    tracks.addCentredArc(centre.x, centre.y, radius, radius, 0.0f, -sweepRad, 0.0f, true);
    tracks.addCentredArc(centre.x, centre.y, radius, radius, 0.0f, 0.0f, sweepRad, true);
    g.setColour(juce::Colour(DeBleedLookAndFeel::arcBackground));
    g.strokePath(tracks, juce::PathStrokeType(3.0f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    const auto grColour = juce::Colour(DeBleedLookAndFeel::orangeAccent);
    const auto signalColour = juce::Colour(DeBleedLookAndFeel::cyanAccent);
    if (smoothedGR_ > 0.005f)
        drawArcWithGlow(g, centre, radius, 0.0f, -sweepRad * smoothedGR_, grColour);
    if (smoothedLevel_ > 0.005f)
        drawArcWithGlow(g, centre, radius, 0.0f, sweepRad * smoothedLevel_, signalColour);

    DeBleedLookAndFeel::drawBody(g, centre, radius - 12.0f, true);

    const float footY = centre.y - radius * std::cos(sweepRad);
    const float footOffset = radius * std::sin(sweepRad);
    const auto labelFont = DeBleedLookAndFeel::font(8.0f, "Bold")
                               .withExtraKerningFactor(1.2f / 8.0f);
    g.setColour(grColour);
    DeBleedLookAndFeel::drawTextAtBaseline(g, "GR", labelFont, centre.x - footOffset, footY + 14.0f);
    g.setColour(signalColour);
    DeBleedLookAndFeel::drawTextAtBaseline(g, "SIGNAL", labelFont, centre.x + footOffset, footY + 14.0f);

    juce::String grText;
    if (gainReductionDb_ > -0.1f)
        grText = "0.0";
    else
        grText = juce::String(gainReductionDb_, 1);

    g.setColour(juce::Colour(DeBleedLookAndFeel::valueText));
    DeBleedLookAndFeel::drawTextAtBaseline(g, grText, DeBleedLookAndFeel::font(30.0f, "Medium"),
                                          centre.x, centre.y + 7.0f);
    g.setColour(juce::Colour(DeBleedLookAndFeel::dimText));
    DeBleedLookAndFeel::drawTextAtBaseline(g, "DB", labelFont, centre.x, centre.y + 24.0f);

    juce::Path topTick;
    topTick.startNewSubPath(centre.x, centre.y - radius - 5.0f);
    topTick.lineTo(centre.x, centre.y - radius - 10.0f);
    g.strokePath(topTick, juce::PathStrokeType(1.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));
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
