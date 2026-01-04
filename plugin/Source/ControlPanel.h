/*
  ==============================================================================
    ControlPanel.h
    Bottom panel with rotary knobs for DeBleed parameters
    Style: Kinetics-inspired dark theme with cyan accent knobs
  ==============================================================================
*/
#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"

/**
 * ControlPanel - Bottom panel containing all parameter knobs.
 *
 * Layout:
 *   [Mix] [Output Gain] [HPF] [LPF] [Sensitivity] [Smoothing]
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
            slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 70, 18);
            parent->addAndMakeVisible(slider);

            label.setText(labelText, juce::dontSendNotification);
            label.setJustificationType(juce::Justification::centred);
            label.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
            label.setFont(juce::Font(11.0f));
            parent->addAndMakeVisible(label);

            attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
                apvts, paramId, slider);
        }

        void setBounds(juce::Rectangle<int> area)
        {
            auto labelArea = area.removeFromTop(18);
            label.setBounds(labelArea);
            slider.setBounds(area);
        }

        void setKnobColor(juce::uint32 color)
        {
            slider.getProperties().set("knobColor", (juce::int64)color);
        }

        void setDualColor(bool dual)
        {
            if (dual)
                slider.getProperties().set("isDualColor", true);
        }
    };

    LabeledKnob mixKnob;
    LabeledKnob outputGainKnob;
    LabeledKnob hpfKnob;
    LabeledKnob lpfKnob;
    LabeledKnob sensitivityKnob;
    LabeledKnob smoothingKnob;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ControlPanel)
};
