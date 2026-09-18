#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "DeBleedLookAndFeel.h"
#include "Section.h"
#include "ArcMeter.h"

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
    ArcMeter arcMeter_;
    ExpanderSection expanderSection_;
    VocalSection vocalSection_;
    OutputSection outputSection_;
    juce::ToggleButton powerButton_, chevronButton_;
    std::unique_ptr<juce::ParameterAttachment> bypassAttachment_;

    static constexpr int expandedWidth = 860;
    static constexpr int collapsedWidth = 268;
    static constexpr int editorHeight = 268;
    static constexpr double slideDurationMs = 180.0;
    bool collapsed_ = false;
    bool sliding_ = false;
    int slideStartWidth_ = expandedWidth;
    double slideStartTime_ = 0.0;

    void toggleCollapsed();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessorEditor)
};
