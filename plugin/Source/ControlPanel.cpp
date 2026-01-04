/*
  ==============================================================================
    ControlPanel.cpp
    Bottom panel with rotary knobs for DeBleed parameters
  ==============================================================================
*/

#include "ControlPanel.h"

ControlPanel::ControlPanel(DeBleedAudioProcessor& processor)
    : audioProcessor(processor)
{
    auto& apvts = processor.getParameters();

    // Set up knobs with their parameter attachments
    mixKnob.setup(this, "MIX", apvts, DeBleedAudioProcessor::PARAM_MIX);
    mixKnob.setDualColor(true);  // Orange to cyan gradient

    outputGainKnob.setup(this, "OUTPUT", apvts, DeBleedAudioProcessor::PARAM_OUTPUT_GAIN);
    outputGainKnob.setKnobColor(juce::Colours::white.getARGB());

    hpfKnob.setup(this, "HPF", apvts, DeBleedAudioProcessor::PARAM_HPF_FREQ);
    hpfKnob.setKnobColor(juce::Colours::orange.getARGB());

    lpfKnob.setup(this, "LPF", apvts, DeBleedAudioProcessor::PARAM_LPF_FREQ);
    lpfKnob.setKnobColor(juce::Colours::cyan.getARGB());

    sensitivityKnob.setup(this, "SENSITIVITY", apvts, DeBleedAudioProcessor::PARAM_SENSITIVITY);
    sensitivityKnob.setKnobColor(juce::Colours::cyan.getARGB());

    smoothingKnob.setup(this, "SMOOTH", apvts, DeBleedAudioProcessor::PARAM_SMOOTHING);
    smoothingKnob.setKnobColor(juce::Colours::grey.getARGB());
}

ControlPanel::~ControlPanel()
{
}

void ControlPanel::paint(juce::Graphics& g)
{
    // Dark panel background
    g.setColour(juce::Colour::fromRGB(10, 11, 13));
    g.fillRect(getLocalBounds());

    // Top border line
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));
}

void ControlPanel::resized()
{
    auto bounds = getLocalBounds().reduced(20, 15);

    // Calculate knob size - evenly distribute across width
    int numKnobs = 6;
    int knobWidth = bounds.getWidth() / numKnobs;
    int knobHeight = bounds.getHeight();

    // Minimum size for knobs
    knobWidth = std::min(knobWidth, 120);
    knobHeight = std::min(knobHeight, 120);

    // Center the knobs
    int totalWidth = knobWidth * numKnobs;
    int startX = bounds.getX() + (bounds.getWidth() - totalWidth) / 2;

    auto getKnobBounds = [&](int index) {
        return juce::Rectangle<int>(startX + index * knobWidth, bounds.getY(), knobWidth, knobHeight);
    };

    mixKnob.setBounds(getKnobBounds(0));
    outputGainKnob.setBounds(getKnobBounds(1));
    hpfKnob.setBounds(getKnobBounds(2));
    lpfKnob.setBounds(getKnobBounds(3));
    sensitivityKnob.setBounds(getKnobBounds(4));
    smoothingKnob.setBounds(getKnobBounds(5));
}
