#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "DeBleedLookAndFeel.h"

/**
 * GateBandSelector - Band selector with dynamic knobs for 6-band gate.
 *
 * Shows 6 clickable tabs (SUB, LOW, L-MID, MID, H-MID, HIGH) and a set of
 * knobs that dynamically bind to the selected band's parameters.
 */
class GateBandSelector : public juce::Component,
                          public juce::Button::Listener
{
public:
    static constexpr int NUM_BANDS = 6;

    GateBandSelector(DeBleedAudioProcessor& processor);
    ~GateBandSelector() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void buttonClicked(juce::Button* button) override;

    // Select a band programmatically
    void selectBand(int band);
    int getSelectedBand() const { return selectedBand; }

private:
    DeBleedAudioProcessor& audioProcessor;
    int selectedBand = 3;  // Default to MID band

    // Band tab buttons
    std::array<std::unique_ptr<juce::TextButton>, NUM_BANDS> bandButtons;
    static constexpr std::array<const char*, NUM_BANDS> bandNames = {
        {"SUB", "LOW", "L-M", "MID", "H-M", "HIGH"}
    };

    // Parameter knobs (shared, rebind on band selection)
    juce::Slider thresholdSlider;
    juce::Label thresholdLabel;
    juce::Slider attackSlider;
    juce::Label attackLabel;
    juce::Slider releaseSlider;
    juce::Label releaseLabel;
    juce::Slider holdSlider;
    juce::Label holdLabel;
    juce::Slider rangeSlider;
    juce::Label rangeLabel;
    juce::ToggleButton enabledButton;

    // Dynamic parameter attachments (recreated on band selection)
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> releaseAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> holdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> rangeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> enabledAttachment;

    void setupKnob(juce::Slider& slider, juce::Label& label, const juce::String& name, juce::Colour colour);
    void rebindToBand(int band);
    void updateButtonStates();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(GateBandSelector)
};
