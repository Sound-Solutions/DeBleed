#include "SpectrogramVisualization.h"

SpectrogramVisualization::SpectrogramVisualization(DeBleedAudioProcessor& processor)
    : audioProcessor(processor)
{
    // Initialize history buffers to zero
    for (auto& frame : magnitudeHistory)
        frame.fill(0.0f);
    for (auto& frame : maskHistory)
        frame.fill(1.0f);  // Default mask = 1.0 (pass-through)

    // Create initial image
    spectrogramImage = juce::Image(juce::Image::RGB, NUM_HISTORY_FRAMES, N_FREQ_BINS, true);
}

void SpectrogramVisualization::updateFromQueue()
{
    auto& vizData = audioProcessor.getVisualizationData();
    DeBleedAudioProcessor::VisualizationData::FrameData frame;

    bool gotNewFrames = false;

    // Consume all available frames from FIFO
    while (vizData.popFrame(frame))
    {
        // Store in rolling buffer
        std::memcpy(magnitudeHistory[writeIndex].data(), frame.magnitude.data(), N_FREQ_BINS * sizeof(float));
        std::memcpy(maskHistory[writeIndex].data(), frame.mask.data(), N_FREQ_BINS * sizeof(float));

        writeIndex = (writeIndex + 1) % NUM_HISTORY_FRAMES;
        gotNewFrames = true;
    }

    if (gotNewFrames)
    {
        needsFullRedraw = true;
        repaint();
    }
}

juce::Colour SpectrogramVisualization::getMagnitudeColor(float magnitude)
{
    // Convert to dB, clamp to range
    float db = juce::Decibels::gainToDecibels(magnitude + 1e-10f);
    float normalized = juce::jmap(db, -80.0f, 0.0f, 0.0f, 1.0f);
    normalized = std::clamp(normalized, 0.0f, 1.0f);

    // Heat colormap: dark blue -> cyan -> green -> yellow -> red
    if (normalized < 0.2f)
    {
        // Dark blue to blue
        float t = normalized / 0.2f;
        return juce::Colour::fromHSV(0.66f, 1.0f, 0.2f + t * 0.6f, 1.0f);
    }
    else if (normalized < 0.4f)
    {
        // Blue to cyan
        float t = (normalized - 0.2f) / 0.2f;
        return juce::Colour::fromHSV(0.66f - t * 0.16f, 1.0f, 0.8f + t * 0.2f, 1.0f);
    }
    else if (normalized < 0.6f)
    {
        // Cyan to green
        float t = (normalized - 0.4f) / 0.2f;
        return juce::Colour::fromHSV(0.5f - t * 0.17f, 1.0f, 1.0f, 1.0f);
    }
    else if (normalized < 0.8f)
    {
        // Green to yellow
        float t = (normalized - 0.6f) / 0.2f;
        return juce::Colour::fromHSV(0.33f - t * 0.16f, 1.0f, 1.0f, 1.0f);
    }
    else
    {
        // Yellow to red
        float t = (normalized - 0.8f) / 0.2f;
        return juce::Colour::fromHSV(0.17f - t * 0.17f, 1.0f, 1.0f, 1.0f);
    }
}

juce::Colour SpectrogramVisualization::getMaskOverlayColor(float maskValue)
{
    // mask=1.0 means pass-through (no overlay)
    // mask=0.0 means full suppression (bright red overlay)
    float suppression = 1.0f - maskValue;
    return juce::Colour(255, 60, 60).withAlpha(suppression * 0.6f);
}

void SpectrogramVisualization::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds();

    // Background
    g.setColour(juce::Colour(0xff0a1520));
    g.fillRect(bounds);

    if (needsFullRedraw)
    {
        // Redraw entire spectrogram to image
        juce::Image::BitmapData bitmapData(spectrogramImage, juce::Image::BitmapData::writeOnly);

        for (int x = 0; x < NUM_HISTORY_FRAMES; ++x)
        {
            int historyIdx = (writeIndex + x) % NUM_HISTORY_FRAMES;

            for (int y = 0; y < N_FREQ_BINS; ++y)
            {
                // Flip Y so low frequencies at bottom
                int binIdx = N_FREQ_BINS - 1 - y;

                // Get magnitude color
                juce::Colour magColor = getMagnitudeColor(magnitudeHistory[historyIdx][binIdx]);

                // Get mask overlay
                juce::Colour maskColor = getMaskOverlayColor(maskHistory[historyIdx][binIdx]);

                // Blend magnitude with mask overlay
                juce::Colour finalColor = magColor.overlaidWith(maskColor);

                bitmapData.setPixelColour(x, y, finalColor);
            }
        }

        needsFullRedraw = false;
    }

    // Draw scaled image to component bounds (leave space for scale)
    int scaleWidth = 35;
    auto imageArea = bounds.removeFromLeft(bounds.getWidth() - scaleWidth);
    g.drawImage(spectrogramImage, imageArea.toFloat(), juce::RectanglePlacement::stretchToFit);

    // Draw border
    g.setColour(juce::Colour(0xff3a4a5c));
    g.drawRect(imageArea, 1);

    // Draw frequency scale
    drawFrequencyScale(g);
}

void SpectrogramVisualization::drawFrequencyScale(juce::Graphics& g)
{
    auto bounds = getLocalBounds();
    int scaleWidth = 35;
    auto scaleArea = bounds.removeFromRight(scaleWidth);

    g.setColour(juce::Colours::white.withAlpha(0.7f));
    g.setFont(10.0f);

    // Frequency markers (approximate for 48kHz, 256-point FFT)
    // Bin spacing = 48000 / 256 = 187.5 Hz per bin
    // 129 bins covers 0 to 24kHz
    struct FreqMarker { float freq; const char* label; };
    FreqMarker markers[] = {
        {100, "100"},
        {500, "500"},
        {1000, "1k"},
        {2000, "2k"},
        {5000, "5k"},
        {10000, "10k"},
        {20000, "20k"}
    };

    float binSpacing = 24000.0f / N_FREQ_BINS;  // Hz per bin

    for (const auto& marker : markers)
    {
        int bin = static_cast<int>(marker.freq / binSpacing);
        if (bin >= 0 && bin < N_FREQ_BINS)
        {
            // Convert bin to Y position (flip because low freq at bottom)
            float yNorm = 1.0f - (static_cast<float>(bin) / N_FREQ_BINS);
            int y = static_cast<int>(scaleArea.getY() + yNorm * scaleArea.getHeight());

            // Draw tick
            g.drawHorizontalLine(y, static_cast<float>(scaleArea.getX()), static_cast<float>(scaleArea.getX() + 5));

            // Draw label
            g.drawText(marker.label, scaleArea.getX() + 6, y - 6, scaleWidth - 8, 12,
                      juce::Justification::left, false);
        }
    }
}

void SpectrogramVisualization::resized()
{
    // Recreate image if size changed significantly
    if (getWidth() > 0 && getHeight() > 0)
    {
        needsFullRedraw = true;
    }
}
