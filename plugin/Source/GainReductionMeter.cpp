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

void GainReductionMeter::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();

    // Background - match DeBleed dark theme
    g.setColour(juce::Colour(0xff0c0c0e));  // visualizerBackground
    g.fillRect(bounds);

    // Meter area (leave space for scale on left)
    int scaleWidth = 20;
    auto meterArea = bounds.removeFromRight(bounds.getWidth() - scaleWidth);
    meterArea.reduce(2, 2);

    // Draw meter background
    g.setColour(juce::Colour(0xff141618));  // panelBackground
    g.fillRect(meterArea);

    // Calculate meter fill (0 dB at top, -60 dB at bottom)
    // Gain reduction is negative, so -30 dB means 50% of the meter
    float normalized = juce::jmap(smoothedReduction, -60.0f, 0.0f, 0.0f, 1.0f);
    normalized = std::clamp(normalized, 0.0f, 1.0f);

    // Fill from top down (showing how much is being reduced)
    int fillHeight = static_cast<int>((1.0f - normalized) * meterArea.getHeight());
    auto fillArea = meterArea.removeFromTop(fillHeight);

    // Gradient fill - bright purple at top (most reduction) to darker purple at bottom
    juce::ColourGradient gradient(
        juce::Colour(0xffcc66ff),  // Bright purple/magenta at top (most reduction)
        static_cast<float>(fillArea.getCentreX()), static_cast<float>(fillArea.getY()),
        juce::Colour(0xff800080),  // Purple accent at bottom (matches Range knob)
        static_cast<float>(fillArea.getCentreX()), static_cast<float>(fillArea.getBottom()),
        false
    );
    g.setGradientFill(gradient);
    g.fillRect(fillArea);

    // Draw border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRect(bounds.removeFromRight(bounds.getWidth()), 1);

    // Draw scale
    drawScale(g, bounds);
}

void GainReductionMeter::drawScale(juce::Graphics& g, juce::Rectangle<int> bounds)
{
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.setFont(9.0f);

    // dB markers
    int dbValues[] = {0, -6, -12, -18, -24, -30, -40, -50, -60};

    for (int db : dbValues)
    {
        float normalized = juce::jmap(static_cast<float>(db), -60.0f, 0.0f, 0.0f, 1.0f);
        int y = bounds.getY() + static_cast<int>((1.0f - normalized) * bounds.getHeight());

        // Draw tick
        g.drawHorizontalLine(y, static_cast<float>(bounds.getRight() - 4), static_cast<float>(bounds.getRight()));

        // Draw label (skip some for space)
        if (db == 0 || db == -12 || db == -24 || db == -40 || db == -60)
        {
            juce::String label = (db == 0) ? "0" : juce::String(db);
            g.drawText(label, bounds.getX(), y - 5, bounds.getWidth() - 5, 10,
                      juce::Justification::right, false);
        }
    }
}

void GainReductionMeter::resized()
{
    // Nothing special needed
}
