#pragma once

#include <JuceHeader.h>
#include "DifferentiableBiquadChain.h"

/**
 * EQCurveDisplay - Real-time frequency response visualization for Neural 5045.
 *
 * Displays the combined magnitude response of the 16-biquad filter chain:
 * - Log frequency axis (20Hz to 20kHz)
 * - Linear dB axis (-24dB to +24dB)
 * - Grid lines at decade frequencies and 6dB intervals
 * - Real-time curve updates from DifferentiableBiquadChain
 *
 * Call updateFromChain() periodically (e.g., from timer) to refresh the display.
 */
class EQCurveDisplay : public juce::Component
{
public:
    EQCurveDisplay();
    ~EQCurveDisplay() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    /**
     * Update the display from a biquad chain's frequency response.
     * Call this periodically (e.g., every 50-100ms) to update the visualization.
     */
    void updateFromChain(const DifferentiableBiquadChain& chain);

    /**
     * Set whether to show the grid lines.
     */
    void setShowGrid(bool show) { showGrid_ = show; repaint(); }

    /**
     * Set the display range in dB.
     */
    void setDecibelRange(float minDb, float maxDb);

private:
    // Convert frequency (Hz) to x position
    float frequencyToX(float freqHz) const;

    // Convert dB to y position
    float decibelToY(float dB) const;

    // Convert x position to frequency (Hz)
    float xToFrequency(float x) const;

    // Draw the background grid
    void drawGrid(juce::Graphics& g);

    // Draw the frequency response curve
    void drawCurve(juce::Graphics& g);

    // Draw frequency and dB labels
    void drawLabels(juce::Graphics& g);

    // Cached magnitude response (in dB) for each pixel
    std::vector<float> magnitudeResponseDb_;

    // Display settings
    float minFreqHz_ = 20.0f;
    float maxFreqHz_ = 20000.0f;
    float minDb_ = -24.0f;
    float maxDb_ = 24.0f;
    bool showGrid_ = true;

    // Plot area (excluding labels)
    juce::Rectangle<float> plotArea_;

    // Margins for labels
    static constexpr int leftMargin_ = 35;
    static constexpr int rightMargin_ = 10;
    static constexpr int topMargin_ = 10;
    static constexpr int bottomMargin_ = 25;

    // Colors
    juce::Colour gridColor_{0x33ffffff};
    juce::Colour curveColor_{0xff5aaa6c};
    juce::Colour fillColor_{0x305aaa6c};
    juce::Colour labelColor_{0x99ffffff};
    juce::Colour zeroLineColor_{0x66ffffff};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EQCurveDisplay)
};
