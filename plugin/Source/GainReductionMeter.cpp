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

    // === Draw gain reduction (purple, fills from top) ===
    if (smoothedReduction < -0.5f)
    {
        // GR fills from top down
        float grNormalized = juce::jmap(smoothedReduction, -60.0f, 0.0f, 0.0f, 1.0f);
        int grHeight = static_cast<int>((1.0f - grNormalized) * meterArea.getHeight());
        auto grArea = meterArea.withHeight(grHeight);

        juce::ColourGradient grGradient(
            juce::Colour(0xffcc66ff).withAlpha(0.9f),
            static_cast<float>(grArea.getCentreX()), static_cast<float>(grArea.getY()),
            juce::Colour(0xff800080).withAlpha(0.7f),
            static_cast<float>(grArea.getCentreX()), static_cast<float>(grArea.getBottom()),
            false
        );
        g.setGradientFill(grGradient);
        g.fillRect(grArea);
    }

    // Draw border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRect(bounds, 1);

    // Draw scale
    g.setColour(juce::Colours::white.withAlpha(0.6f));
    g.setFont(9.0f);

    int dbValues[] = {0, -12, -24, -36, -48, -60};
    for (int db : dbValues)
    {
        float normalized = juce::jmap(static_cast<float>(db), -60.0f, 0.0f, 1.0f, 0.0f);
        int y = meterArea.getY() + static_cast<int>(normalized * meterArea.getHeight());

        // Draw tick
        g.drawHorizontalLine(y, static_cast<float>(scaleBounds.getRight() - 4),
                            static_cast<float>(scaleBounds.getRight()));

        // Draw label
        juce::String label = (db == 0) ? "0" : juce::String(db);
        g.drawText(label, scaleBounds.getX(), y - 5, scaleBounds.getWidth() - 5, 10,
                  juce::Justification::right, false);
    }

    // Label at top
    g.setFont(10.0f);
    g.setColour(juce::Colours::white.withAlpha(0.8f));
    auto labelArea = meterArea.removeFromTop(14);
    g.drawText("GR", labelArea, juce::Justification::centred);
}

void GainReductionMeter::resized()
{
}
