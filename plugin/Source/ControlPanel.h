/*
  ==============================================================================
    ControlPanel.h
    Two-row control panel for DeBleed V2 expander
  ==============================================================================
*/
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

/**
 * ControlPanel - Two-row knob layout
 *
 * Row 1 (top):    [Mix] [Output]  - Cyan
 * Row 2 (bottom): [Thresh] [Ratio] [Attack] [Release] [Range] - Orange
 */
class ControlPanel : public juce::Component
{
public:
    ControlPanel(DeBleedAudioProcessor& processor);
    ~ControlPanel() override;

    void paint(juce::Graphics& g) override;
    void resized() override;

private:
    DeBleedAudioProcessor& audioProcessor;

    // Knob helper
    struct LabeledKnob
    {
        juce::Slider slider;
        juce::Label label;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;

        void setup(juce::Component* parent, const juce::String& labelText,
                   juce::AudioProcessorValueTreeState& apvts, const juce::String& paramId)
        {
            slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
            slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 65, 16);
            parent->addAndMakeVisible(slider);

            label.setText(labelText, juce::dontSendNotification);
            label.setJustificationType(juce::Justification::centred);
            label.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
            label.setFont(juce::Font(10.0f));
            parent->addAndMakeVisible(label);

            attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                apvts, paramId, slider);
        }

        void setBounds(juce::Rectangle<int> area)
        {
            auto labelArea = area.removeFromTop(14);
            label.setBounds(labelArea);
            slider.setBounds(area);
        }

        void setKnobColor(juce::uint32 color)
        {
            slider.getProperties().set("knobColor", (juce::int64)color);
        }
    };

    // Row 1: Output controls (cyan)
    LabeledKnob mixKnob;
    LabeledKnob outputGainKnob;

    // Row 2: Expander controls (orange)
    LabeledKnob thresholdKnob;
    LabeledKnob ratioKnob;
    LabeledKnob attackKnob;
    LabeledKnob releaseKnob;
    LabeledKnob rangeKnob;

    // Colors matching the mix knob gradient
    static constexpr juce::uint32 cyanColor = 0xff00d4ff;   // Cyan from mix gradient
    static constexpr juce::uint32 orangeColor = 0xffff8800; // Orange from mix gradient

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ControlPanel)
};
