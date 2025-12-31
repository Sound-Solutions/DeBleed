#include "EQCurveDisplay.h"

EQCurveDisplay::EQCurveDisplay()
{
    setOpaque(false);
}

void EQCurveDisplay::resized()
{
    auto bounds = getLocalBounds().toFloat();

    // Calculate plot area (with margins for labels)
    plotArea_ = bounds;
    plotArea_.removeFromLeft(static_cast<float>(leftMargin_));
    plotArea_.removeFromRight(static_cast<float>(rightMargin_));
    plotArea_.removeFromTop(static_cast<float>(topMargin_));
    plotArea_.removeFromBottom(static_cast<float>(bottomMargin_));

    // Resize magnitude response cache
    int numPoints = std::max(1, static_cast<int>(plotArea_.getWidth()));
    magnitudeResponseDb_.resize(numPoints, 0.0f);
}

void EQCurveDisplay::paint(juce::Graphics& g)
{
    // Background
    g.setColour(juce::Colour(0xff1a1a1a));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 4.0f);

    if (plotArea_.isEmpty())
        return;

    // Draw components
    if (showGrid_)
        drawGrid(g);

    drawCurve(g);
    drawLabels(g);
}

void EQCurveDisplay::updateFromChain(const DifferentiableBiquadChain& chain)
{
    if (plotArea_.isEmpty())
        return;

    int numPoints = static_cast<int>(magnitudeResponseDb_.size());

    for (int i = 0; i < numPoints; ++i)
    {
        float x = plotArea_.getX() + static_cast<float>(i);
        float freqHz = xToFrequency(x);

        // Get magnitude response from the chain
        float magnitude = chain.getFrequencyResponse(freqHz);

        // Convert to dB, with protection for zero/negative values
        if (magnitude > 0.0f)
            magnitudeResponseDb_[i] = 20.0f * std::log10(magnitude);
        else
            magnitudeResponseDb_[i] = minDb_;

        // Clamp to display range
        magnitudeResponseDb_[i] = juce::jlimit(minDb_, maxDb_, magnitudeResponseDb_[i]);
    }

    repaint();
}

void EQCurveDisplay::setDecibelRange(float minDb, float maxDb)
{
    minDb_ = minDb;
    maxDb_ = maxDb;
    repaint();
}

float EQCurveDisplay::frequencyToX(float freqHz) const
{
    // Log scale: x = (log(f) - log(fmin)) / (log(fmax) - log(fmin)) * width
    float logMin = std::log10(minFreqHz_);
    float logMax = std::log10(maxFreqHz_);
    float logFreq = std::log10(std::max(freqHz, minFreqHz_));

    float normalized = (logFreq - logMin) / (logMax - logMin);
    return plotArea_.getX() + normalized * plotArea_.getWidth();
}

float EQCurveDisplay::decibelToY(float dB) const
{
    // Linear scale: y = (1 - (dB - minDb) / (maxDb - minDb)) * height
    float normalized = (dB - minDb_) / (maxDb_ - minDb_);
    return plotArea_.getBottom() - normalized * plotArea_.getHeight();
}

float EQCurveDisplay::xToFrequency(float x) const
{
    // Inverse of frequencyToX
    float logMin = std::log10(minFreqHz_);
    float logMax = std::log10(maxFreqHz_);

    float normalized = (x - plotArea_.getX()) / plotArea_.getWidth();
    float logFreq = logMin + normalized * (logMax - logMin);

    return std::pow(10.0f, logFreq);
}

void EQCurveDisplay::drawGrid(juce::Graphics& g)
{
    g.setColour(gridColor_);

    // Vertical lines at decade frequencies: 20, 50, 100, 200, 500, 1k, 2k, 5k, 10k, 20k
    std::array<float, 10> frequencies = {20.0f, 50.0f, 100.0f, 200.0f, 500.0f,
                                          1000.0f, 2000.0f, 5000.0f, 10000.0f, 20000.0f};

    for (float freq : frequencies)
    {
        if (freq >= minFreqHz_ && freq <= maxFreqHz_)
        {
            float x = frequencyToX(freq);
            g.drawVerticalLine(static_cast<int>(x), plotArea_.getY(), plotArea_.getBottom());
        }
    }

    // Horizontal lines at 6dB intervals
    for (float db = minDb_; db <= maxDb_; db += 6.0f)
    {
        float y = decibelToY(db);

        if (std::abs(db) < 0.1f)
        {
            // 0dB line is brighter
            g.setColour(zeroLineColor_);
            g.drawHorizontalLine(static_cast<int>(y), plotArea_.getX(), plotArea_.getRight());
            g.setColour(gridColor_);
        }
        else
        {
            g.drawHorizontalLine(static_cast<int>(y), plotArea_.getX(), plotArea_.getRight());
        }
    }
}

void EQCurveDisplay::drawCurve(juce::Graphics& g)
{
    if (magnitudeResponseDb_.empty())
        return;

    juce::Path curvePath;
    juce::Path fillPath;

    // Start the paths
    float startY = decibelToY(magnitudeResponseDb_[0]);
    curvePath.startNewSubPath(plotArea_.getX(), startY);

    fillPath.startNewSubPath(plotArea_.getX(), plotArea_.getBottom());
    fillPath.lineTo(plotArea_.getX(), startY);

    // Add points for each pixel
    for (size_t i = 1; i < magnitudeResponseDb_.size(); ++i)
    {
        float x = plotArea_.getX() + static_cast<float>(i);
        float y = decibelToY(magnitudeResponseDb_[i]);

        curvePath.lineTo(x, y);
        fillPath.lineTo(x, y);
    }

    // Close fill path
    fillPath.lineTo(plotArea_.getRight(), plotArea_.getBottom());
    fillPath.closeSubPath();

    // Draw fill
    g.setColour(fillColor_);
    g.fillPath(fillPath);

    // Draw curve
    g.setColour(curveColor_);
    g.strokePath(curvePath, juce::PathStrokeType(2.0f));
}

void EQCurveDisplay::drawLabels(juce::Graphics& g)
{
    g.setColour(labelColor_);
    g.setFont(juce::Font(10.0f));

    // Frequency labels (bottom)
    std::array<std::pair<float, const char*>, 6> freqLabels = {{
        {100.0f, "100"},
        {1000.0f, "1k"},
        {10000.0f, "10k"},
        {20.0f, "20"},
        {200.0f, "200"},
        {2000.0f, "2k"}
    }};

    for (const auto& [freq, label] : freqLabels)
    {
        if (freq >= minFreqHz_ && freq <= maxFreqHz_)
        {
            float x = frequencyToX(freq);
            auto textBounds = juce::Rectangle<float>(x - 20.0f, plotArea_.getBottom() + 4.0f, 40.0f, 16.0f);
            g.drawText(label, textBounds, juce::Justification::centredTop);
        }
    }

    // dB labels (left side)
    std::array<float, 5> dbLabels = {-24.0f, -12.0f, 0.0f, 12.0f, 24.0f};

    for (float db : dbLabels)
    {
        if (db >= minDb_ && db <= maxDb_)
        {
            float y = decibelToY(db);
            juce::String text = (db >= 0 && db != 0) ? "+" + juce::String(static_cast<int>(db))
                                                      : juce::String(static_cast<int>(db));
            auto textBounds = juce::Rectangle<float>(2.0f, y - 8.0f, 30.0f, 16.0f);
            g.drawText(text, textBounds, juce::Justification::centredRight);
        }
    }

    // Hz label
    g.setFont(juce::Font(9.0f));
    g.drawText("Hz", juce::Rectangle<float>(plotArea_.getRight() - 15.0f, plotArea_.getBottom() + 8.0f, 20.0f, 12.0f),
               juce::Justification::centredLeft);

    // dB label
    g.drawText("dB", juce::Rectangle<float>(2.0f, plotArea_.getY() - 2.0f, 20.0f, 12.0f),
               juce::Justification::centredLeft);
}
