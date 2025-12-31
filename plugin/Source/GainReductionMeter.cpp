#include "GainReductionMeter.h"

GainReductionMeter::GainReductionMeter()
{
}

void GainReductionMeter::setReductionLevel(float reductionDb)
{
    currentReductionDb = reductionDb;

    // Apply smoothing
    float targetSmoothed = std::clamp(reductionDb, -60.0f, 0.0f);
    float coeff = (targetSmoothed < smoothedReduction) ? METER_ATTACK : METER_RELEASE;
    smoothedReduction += coeff * (targetSmoothed - smoothedReduction);
}

void GainReductionMeter::setGateInfo(float detectedLevelDb, float thresholdDb, bool gateOpen)
{
    detectedLevel = detectedLevelDb;
    threshold = thresholdDb;
    isGateOpen = gateOpen;

    // Smooth the level meter
    float coeff = (detectedLevel > smoothedLevel) ? LEVEL_ATTACK : LEVEL_RELEASE;
    smoothedLevel += coeff * (detectedLevel - smoothedLevel);
    smoothedLevel = std::clamp(smoothedLevel, -60.0f, 0.0f);
}

void GainReductionMeter::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();

    // Background
    g.setColour(juce::Colour(0xff0c0c0e));
    g.fillRect(bounds);

    // Scale area on left
    int scaleWidth = 22;
    auto scaleBounds = bounds.removeFromLeft(scaleWidth);

    // Meter area
    auto meterArea = bounds.reduced(2, 2);

    // Draw meter background
    g.setColour(juce::Colour(0xff141618));
    g.fillRect(meterArea);

    // dB range: 0 at top, -60 at bottom
    auto dbToY = [&](float db) -> float {
        float normalized = juce::jmap(db, -60.0f, 0.0f, 1.0f, 0.0f);
        return meterArea.getY() + normalized * meterArea.getHeight();
    };

    // === Draw input level bar (cyan) ===
    float levelY = dbToY(smoothedLevel);
    auto levelBar = meterArea.toFloat();
    levelBar.setTop(levelY);

    // Gradient from bright cyan at level to darker below
    juce::ColourGradient levelGradient(
        juce::Colour(0xff00d4ff).withAlpha(0.8f),
        levelBar.getCentreX(), levelBar.getY(),
        juce::Colour(0xff006688).withAlpha(0.4f),
        levelBar.getCentreX(), levelBar.getBottom(),
        false
    );
    g.setGradientFill(levelGradient);
    g.fillRect(levelBar);

    // === Draw gain reduction overlay (purple, from top) ===
    if (smoothedReduction < -0.5f)  // Only show if actually reducing
    {
        float grNormalized = juce::jmap(smoothedReduction, -60.0f, 0.0f, 0.0f, 1.0f);
        int grHeight = static_cast<int>((1.0f - grNormalized) * meterArea.getHeight());
        auto grArea = meterArea.removeFromTop(grHeight);

        juce::ColourGradient grGradient(
            juce::Colour(0xffcc66ff).withAlpha(0.9f),
            static_cast<float>(grArea.getCentreX()), static_cast<float>(grArea.getY()),
            juce::Colour(0xff800080).withAlpha(0.6f),
            static_cast<float>(grArea.getCentreX()), static_cast<float>(grArea.getBottom()),
            false
        );
        g.setGradientFill(grGradient);
        g.fillRect(grArea);
    }

    // === Draw threshold line (orange) ===
    float thresholdY = dbToY(threshold);
    g.setColour(juce::Colour(0xffff8800));  // Orange
    g.drawHorizontalLine(static_cast<int>(thresholdY),
                         static_cast<float>(meterArea.getX()),
                         static_cast<float>(meterArea.getRight()));

    // Draw threshold triangle marker on left
    juce::Path triangle;
    float triSize = 6.0f;
    triangle.addTriangle(
        meterArea.getX(), thresholdY,
        meterArea.getX() - triSize, thresholdY - triSize/2,
        meterArea.getX() - triSize, thresholdY + triSize/2
    );
    g.fillPath(triangle);

    // === Draw gate state indicator ===
    auto indicatorArea = meterArea.removeFromBottom(12).reduced(2, 2);
    g.setColour(isGateOpen ? juce::Colour(0xff00cc66) : juce::Colour(0xffcc3333));
    g.fillRoundedRectangle(indicatorArea.toFloat(), 2.0f);

    g.setColour(juce::Colours::white);
    g.setFont(8.0f);
    g.drawText(isGateOpen ? "OPEN" : "GATE", indicatorArea, juce::Justification::centred);

    // Draw border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRect(bounds, 1);

    // Draw scale
    drawScale(g, scaleBounds);
}

void GainReductionMeter::drawScale(juce::Graphics& g, juce::Rectangle<int> bounds)
{
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.setFont(9.0f);

    // dB markers
    int dbValues[] = {0, -12, -24, -36, -48, -60};

    for (int db : dbValues)
    {
        float normalized = juce::jmap(static_cast<float>(db), -60.0f, 0.0f, 1.0f, 0.0f);
        int y = bounds.getY() + 2 + static_cast<int>(normalized * (bounds.getHeight() - 16));

        // Draw tick
        g.drawHorizontalLine(y, static_cast<float>(bounds.getRight() - 4), static_cast<float>(bounds.getRight()));

        // Draw label
        juce::String label = (db == 0) ? "0" : juce::String(db);
        g.drawText(label, bounds.getX(), y - 5, bounds.getWidth() - 5, 10,
                  juce::Justification::right, false);
    }
}

void GainReductionMeter::resized()
{
}
