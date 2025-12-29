#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include <array>

/**
 * SpectrogramVisualization - Displays a scrolling spectrogram with mask overlay.
 *
 * Shows:
 * - Input magnitude spectrogram (heat colormap: blue -> cyan -> green -> yellow -> red)
 * - Neural mask overlay (red tint where suppression is happening)
 * - Scrolls left-to-right as new frames arrive
 */
class SpectrogramVisualization : public juce::Component
{
public:
    static constexpr int NUM_HISTORY_FRAMES = 128;  // ~340ms at 128-hop, 48kHz
    static constexpr int N_FREQ_BINS = 129;

    SpectrogramVisualization(DeBleedAudioProcessor& processor);
    ~SpectrogramVisualization() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Called from timer to consume queued frames and trigger repaint
    void updateFromQueue();

private:
    DeBleedAudioProcessor& audioProcessor;

    // Rolling buffer for spectrogram history
    std::array<std::array<float, N_FREQ_BINS>, NUM_HISTORY_FRAMES> magnitudeHistory;
    std::array<std::array<float, N_FREQ_BINS>, NUM_HISTORY_FRAMES> maskHistory;
    int writeIndex = 0;

    // Pre-rendered image for efficiency
    juce::Image spectrogramImage;
    bool needsFullRedraw = true;

    // Color mapping
    juce::Colour getMagnitudeColor(float magnitude);
    juce::Colour getMaskOverlayColor(float maskValue);

    // Frequency scale drawing
    void drawFrequencyScale(juce::Graphics& g);

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectrogramVisualization)
};
