/*
  ==============================================================================
    ControlPanel.cpp
    Two-row control panel for DeBleed V2 expander
  ==============================================================================
*/

#include "ControlPanel.h"

ControlPanel::ControlPanel(DeBleedAudioProcessor& processor)
    : audioProcessor(processor)
{
    auto& apvts = processor.getParameters();

    // =========================================================================
    // Row 1: Output controls (Cyan)
    // =========================================================================
    mixKnob.setup(this, "MIX", apvts, DeBleedAudioProcessor::PARAM_MIX);
    mixKnob.setKnobColor(cyanColor);

    outputGainKnob.setup(this, "OUTPUT", apvts, DeBleedAudioProcessor::PARAM_OUTPUT_GAIN);
    outputGainKnob.setKnobColor(cyanColor);

    // =========================================================================
    // Row 2: Expander controls (Orange)
    // =========================================================================
    thresholdKnob.setup(this, "THRESH", apvts, DeBleedAudioProcessor::PARAM_EXP_THRESHOLD);
    thresholdKnob.setKnobColor(orangeColor);

    ratioKnob.setup(this, "RATIO", apvts, DeBleedAudioProcessor::PARAM_EXP_RATIO);
    ratioKnob.setKnobColor(orangeColor);

    attackKnob.setup(this, "ATTACK", apvts, DeBleedAudioProcessor::PARAM_EXP_ATTACK);
    attackKnob.setKnobColor(orangeColor);

    releaseKnob.setup(this, "RELEASE", apvts, DeBleedAudioProcessor::PARAM_EXP_RELEASE);
    releaseKnob.setKnobColor(orangeColor);

    rangeKnob.setup(this, "RANGE", apvts, DeBleedAudioProcessor::PARAM_EXP_RANGE);
    rangeKnob.setKnobColor(orangeColor);
}

ControlPanel::~ControlPanel()
{
}

void ControlPanel::paint(juce::Graphics& g)
{
    // Dark panel background
    g.setColour(juce::Colour::fromRGB(18, 18, 20));
    g.fillRect(getLocalBounds());

    // Top border line
    g.setColour(juce::Colours::white.withAlpha(0.08f));
    g.drawHorizontalLine(0, 0.0f, static_cast<float>(getWidth()));

    // Row separator
    int rowHeight = getHeight() / 2;
    g.setColour(juce::Colours::white.withAlpha(0.04f));
    g.drawHorizontalLine(rowHeight, 20.0f, static_cast<float>(getWidth() - 20));
}

void ControlPanel::resized()
{
    auto bounds = getLocalBounds().reduced(15, 8);
    int rowHeight = bounds.getHeight() / 2;

    // =========================================================================
    // Row 1: Mix and Output (centered)
    // =========================================================================
    auto row1 = bounds.removeFromTop(rowHeight);
    int knobWidth1 = 90;
    int totalWidth1 = knobWidth1 * 2 + 30;
    int startX1 = row1.getCentreX() - totalWidth1 / 2;

    mixKnob.setBounds(juce::Rectangle<int>(startX1, row1.getY(), knobWidth1, rowHeight - 5));
    outputGainKnob.setBounds(juce::Rectangle<int>(startX1 + knobWidth1 + 30, row1.getY(), knobWidth1, rowHeight - 5));

    // =========================================================================
    // Row 2: Expander controls (5 knobs, evenly spaced)
    // =========================================================================
    auto row2 = bounds;
    int numKnobs = 5;
    int knobWidth2 = std::min(row2.getWidth() / numKnobs, 90);
    int totalWidth2 = knobWidth2 * numKnobs;
    int startX2 = row2.getCentreX() - totalWidth2 / 2;

    thresholdKnob.setBounds(juce::Rectangle<int>(startX2 + knobWidth2 * 0, row2.getY(), knobWidth2, rowHeight - 5));
    ratioKnob.setBounds(juce::Rectangle<int>(startX2 + knobWidth2 * 1, row2.getY(), knobWidth2, rowHeight - 5));
    attackKnob.setBounds(juce::Rectangle<int>(startX2 + knobWidth2 * 2, row2.getY(), knobWidth2, rowHeight - 5));
    releaseKnob.setBounds(juce::Rectangle<int>(startX2 + knobWidth2 * 3, row2.getY(), knobWidth2, rowHeight - 5));
    rangeKnob.setBounds(juce::Rectangle<int>(startX2 + knobWidth2 * 4, row2.getY(), knobWidth2, rowHeight - 5));
}
