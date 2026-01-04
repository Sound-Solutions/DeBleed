#pragma once

#include <JuceHeader.h>
#include "DifferentiableBiquadChain.h"

/**
 * EQCurveDisplay - Interactive frequency response visualization for Neural 5045.
 *
 * Displays the combined magnitude response of the 16-biquad filter chain:
 * - Log frequency axis (20Hz to 20kHz)
 * - Linear dB axis (-24dB to +24dB)
 * - Grid lines at decade frequencies and 6dB intervals
 * - Real-time curve updates from DifferentiableBiquadChain
 * - Draggable HPF/LPF handles for user control
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

    // Mouse interaction
    void mouseMove(const juce::MouseEvent& e) override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    void mouseExit(const juce::MouseEvent& e) override;

    /**
     * Update the display from a biquad chain's frequency response.
     * Call this periodically (e.g., every 50-100ms) to update the visualization.
     */
    void updateFromChain(const DifferentiableBiquadChain& chain);

    /**
     * Set callback for when HPF frequency is changed via dragging.
     */
    void setHPFChangedCallback(std::function<void(float)> callback) { hpfChangedCallback_ = callback; }

    /**
     * Set callback for when LPF frequency is changed via dragging.
     */
    void setLPFChangedCallback(std::function<void(float)> callback) { lpfChangedCallback_ = callback; }

    /**
     * Set the current HPF/LPF frequencies for handle positions.
     */
    void setFilterFrequencies(float hpfHz, float lpfHz);

    /**
     * Set whether to show the grid lines.
     */
    void setShowGrid(bool show) { showGrid_ = show; repaint(); }

    /**
     * Set the display range in dB.
     */
    void setDecibelRange(float minDb, float maxDb);

private:
    // Handle types
    enum class HandleType { None, HPF, LPF };

    // Convert frequency (Hz) to x position
    float frequencyToX(float freqHz) const;

    // Convert dB to y position
    float decibelToY(float dB) const;

    // Convert x position to frequency (Hz)
    float xToFrequency(float x) const;

    // Get handle hit area
    juce::Rectangle<float> getHandleBounds(HandleType type) const;

    // Check if point is over a handle
    HandleType hitTestHandle(juce::Point<float> point) const;

    // Draw the background grid
    void drawGrid(juce::Graphics& g);

    // Draw the frequency response curve
    void drawCurve(juce::Graphics& g);

    // Draw frequency and dB labels
    void drawLabels(juce::Graphics& g);

    // Draw interactive handles
    void drawHandles(juce::Graphics& g);

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

    // Filter handle positions
    float hpfFreqHz_ = 20.0f;
    float lpfFreqHz_ = 20000.0f;

    // Interaction state
    HandleType hoveredHandle_ = HandleType::None;
    HandleType draggedHandle_ = HandleType::None;
    float dragStartFreq_ = 0.0f;

    // Callbacks
    std::function<void(float)> hpfChangedCallback_;
    std::function<void(float)> lpfChangedCallback_;

    // Margins for labels
    static constexpr int leftMargin_ = 35;
    static constexpr int rightMargin_ = 10;
    static constexpr int topMargin_ = 10;
    static constexpr int bottomMargin_ = 25;

    // Handle appearance
    static constexpr float handleRadius_ = 8.0f;
    static constexpr float handleHitRadius_ = 12.0f;

    // Colors
    juce::Colour gridColor_{0x33ffffff};
    juce::Colour curveColor_{0xff00ffff};        // Cyan
    juce::Colour fillColor_{0x2000ffff};         // Cyan with alpha
    juce::Colour labelColor_{0x99ffffff};
    juce::Colour zeroLineColor_{0x66ffffff};
    juce::Colour hpfHandleColor_{0xffffa500};    // Orange
    juce::Colour lpfHandleColor_{0xff00ffff};    // Cyan
    juce::Colour handleHoverColor_{0xffffffff};  // White

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(EQCurveDisplay)
};
