#include "ConfidenceMeter.h"

ConfidenceMeter::ConfidenceMeter()
{
}

void ConfidenceMeter::setConfidence(float conf)
{
    confidence = juce::jlimit(0.0f, 1.0f, conf);

    // Smooth the display
    float coeff = (confidence > smoothedConfidence) ? ATTACK_COEFF : RELEASE_COEFF;
    smoothedConfidence += coeff * (confidence - smoothedConfidence);
}

void ConfidenceMeter::setThreshold(float thresh)
{
    threshold = juce::jlimit(0.0f, 1.0f, thresh);
}

void ConfidenceMeter::setGateOpen(bool open)
{
    isGateOpen = open;
}

void ConfidenceMeter::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();

    // Background
    g.setColour(juce::Colour(0xff0c0c0e));
    g.fillRect(bounds);

    // Scale area on left
    int scaleWidth = 28;
    auto scaleBounds = bounds.removeFromLeft(scaleWidth);

    // Meter area
    auto meterArea = bounds.reduced(2, 2);

    // Draw meter background
    g.setColour(juce::Colour(0xff141618));
    g.fillRect(meterArea);

    // Percentage scale: 100% at top, 0% at bottom
    auto pctToY = [&](float pct) -> float {
        return meterArea.getY() + (1.0f - pct) * meterArea.getHeight();
    };

    // === Draw confidence bar (cyan) ===
    float confY = pctToY(smoothedConfidence);
    auto confBar = meterArea.toFloat();
    confBar.setTop(confY);

    // Gradient from bright cyan at top to darker below
    juce::ColourGradient confGradient(
        juce::Colour(0xff00d4ff).withAlpha(0.9f),
        confBar.getCentreX(), confBar.getY(),
        juce::Colour(0xff006688).withAlpha(0.5f),
        confBar.getCentreX(), confBar.getBottom(),
        false
    );
    g.setGradientFill(confGradient);
    g.fillRect(confBar);

    // === Draw threshold line (orange) - directly from parameter ===
    float thresholdY = pctToY(threshold);
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

    // === Draw gate state indicator at bottom ===
    auto indicatorArea = meterArea.removeFromBottom(14).reduced(2, 2);
    g.setColour(isGateOpen ? juce::Colour(0xff00cc66) : juce::Colour(0xffcc3333));
    g.fillRoundedRectangle(indicatorArea.toFloat(), 2.0f);

    g.setColour(juce::Colours::white);
    g.setFont(8.0f);
    g.drawText(isGateOpen ? "OPEN" : "GATE", indicatorArea, juce::Justification::centred);

    // Draw border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRect(bounds, 1);

    // Draw scale
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.setFont(9.0f);

    int pctValues[] = {100, 75, 50, 25, 0};
    for (int pct : pctValues)
    {
        float y = pctToY(pct / 100.0f);
        int yInt = static_cast<int>(y);

        // Draw tick
        g.drawHorizontalLine(yInt, static_cast<float>(scaleBounds.getRight() - 4),
                            static_cast<float>(scaleBounds.getRight()));

        // Draw label
        juce::String label = juce::String(pct) + "%";
        g.drawText(label, scaleBounds.getX(), yInt - 5, scaleBounds.getWidth() - 5, 10,
                  juce::Justification::right, false);
    }

    // Label at top
    g.setFont(10.0f);
    g.setColour(juce::Colours::white.withAlpha(0.8f));
    g.drawText("CONF", meterArea.removeFromTop(14), juce::Justification::centred);
}

void ConfidenceMeter::resized()
{
}
