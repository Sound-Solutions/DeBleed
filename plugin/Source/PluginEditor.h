#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "DeBleedLookAndFeel.h"
#include "ControlPanel.h"
#include "ArcMeter.h"

/**
 * DeBleedAudioProcessorEditor - Clean, minimal expander UI.
 *
 * Layout:
 * - Header: Title and power button
 * - Center: Arc meter showing gain reduction + VAD
 * - Bottom: Control panel with 7 knobs
 */
class DeBleedAudioProcessorEditor : public juce::AudioProcessorEditor,
                                     public juce::Timer
{
public:
    explicit DeBleedAudioProcessorEditor(DeBleedAudioProcessor&);
    ~DeBleedAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    DeBleedAudioProcessor& audioProcessor;
    DeBleedLookAndFeel customLookAndFeel;

    // Header components
    juce::Label titleLabel;
    juce::ToggleButton bypassButton;

    // Arc meter for GR visualization
    ArcMeter arcMeter_;

    // Bottom control panel
    ControlPanel controlPanel_;

    // Parameter attachments for header controls
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;

    // Layout constants
    static constexpr int headerHeight = 36;
    static constexpr int controlPanelHeight = 200;  // Two rows of knobs

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessorEditor)
};
