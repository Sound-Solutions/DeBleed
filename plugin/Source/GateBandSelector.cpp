#include "GateBandSelector.h"

GateBandSelector::GateBandSelector(DeBleedAudioProcessor& processor)
    : audioProcessor(processor)
{
    // Create band buttons
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        bandButtons[b] = std::make_unique<juce::TextButton>(bandNames[b]);
        bandButtons[b]->addListener(this);
        bandButtons[b]->setClickingTogglesState(false);
        addAndMakeVisible(bandButtons[b].get());
    }

    // Setup knobs with orange accent (gate color)
    juce::Colour gateOrange = juce::Colour(0xFFFF8C00);  // Dark orange

    setupKnob(thresholdSlider, thresholdLabel, "Thresh", gateOrange);
    setupKnob(attackSlider, attackLabel, "Atk", gateOrange);
    setupKnob(releaseSlider, releaseLabel, "Rel", gateOrange);
    setupKnob(holdSlider, holdLabel, "Hold", gateOrange);
    setupKnob(rangeSlider, rangeLabel, "Range", gateOrange);

    // Enable button
    enabledButton.setButtonText("EN");
    enabledButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white.withAlpha(0.8f));
    addAndMakeVisible(enabledButton);

    // Bind to initial band
    rebindToBand(selectedBand);
    updateButtonStates();
}

GateBandSelector::~GateBandSelector()
{
    for (auto& btn : bandButtons)
    {
        if (btn)
            btn->removeListener(this);
    }
}

void GateBandSelector::setupKnob(juce::Slider& slider, juce::Label& label, const juce::String& name, juce::Colour colour)
{
    slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 50, 14);
    slider.setColour(juce::Slider::rotarySliderFillColourId, colour);
    slider.setColour(juce::Slider::thumbColourId, colour);
    addAndMakeVisible(slider);

    label.setText(name, juce::dontSendNotification);
    label.setJustificationType(juce::Justification::centred);
    label.setFont(juce::Font(10.0f, juce::Font::bold));
    label.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    addAndMakeVisible(label);
}

void GateBandSelector::rebindToBand(int band)
{
    if (band < 0 || band >= NUM_BANDS)
        return;

    // Clear existing attachments first
    thresholdAttachment.reset();
    attackAttachment.reset();
    releaseAttachment.reset();
    holdAttachment.reset();
    rangeAttachment.reset();
    enabledAttachment.reset();

    // Create new attachments for selected band
    thresholdAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(),
        DeBleedAudioProcessor::PARAM_GATE_THRESHOLD[band],
        thresholdSlider);

    attackAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(),
        DeBleedAudioProcessor::PARAM_GATE_ATTACK[band],
        attackSlider);

    releaseAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(),
        DeBleedAudioProcessor::PARAM_GATE_RELEASE[band],
        releaseSlider);

    holdAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(),
        DeBleedAudioProcessor::PARAM_GATE_HOLD[band],
        holdSlider);

    rangeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(),
        DeBleedAudioProcessor::PARAM_GATE_RANGE[band],
        rangeSlider);

    enabledAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(),
        DeBleedAudioProcessor::PARAM_GATE_ENABLED[band],
        enabledButton);
}

void GateBandSelector::updateButtonStates()
{
    juce::Colour selectedColor = juce::Colour(0xFFFF8C00);  // Orange
    juce::Colour normalColor = juce::Colour(0xFF404040);

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        bool isSelected = (b == selectedBand);
        bandButtons[b]->setColour(juce::TextButton::buttonColourId,
                                   isSelected ? selectedColor : normalColor);
        bandButtons[b]->setColour(juce::TextButton::textColourOffId,
                                   isSelected ? juce::Colours::black : juce::Colours::white.withAlpha(0.8f));
    }
}

void GateBandSelector::selectBand(int band)
{
    if (band >= 0 && band < NUM_BANDS && band != selectedBand)
    {
        selectedBand = band;
        rebindToBand(band);
        updateButtonStates();
        repaint();
    }
}

void GateBandSelector::buttonClicked(juce::Button* button)
{
    for (int b = 0; b < NUM_BANDS; ++b)
    {
        if (button == bandButtons[b].get())
        {
            selectBand(b);
            break;
        }
    }
}

void GateBandSelector::paint(juce::Graphics& g)
{
    // Background
    g.setColour(juce::Colour(0xFF2A2A2A));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 4.0f);

    // Border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRoundedRectangle(getLocalBounds().toFloat().reduced(0.5f), 4.0f, 1.0f);
}

void GateBandSelector::resized()
{
    auto bounds = getLocalBounds().reduced(4);

    // Top row: band selector buttons
    auto buttonRow = bounds.removeFromTop(24);
    int buttonWidth = buttonRow.getWidth() / NUM_BANDS;

    for (int b = 0; b < NUM_BANDS; ++b)
    {
        bandButtons[b]->setBounds(buttonRow.removeFromLeft(buttonWidth).reduced(1));
    }

    bounds.removeFromTop(4);  // Spacing

    // Bottom section: knobs
    auto knobArea = bounds;
    int knobWidth = 50;
    int knobSpacing = 2;

    // Helper to place a knob with label
    auto placeKnob = [&](juce::Slider& slider, juce::Label& label)
    {
        auto knobBounds = knobArea.removeFromLeft(knobWidth);
        auto labelBounds = knobBounds.removeFromTop(14);
        label.setBounds(labelBounds);
        slider.setBounds(knobBounds.reduced(2, 0));
        knobArea.removeFromLeft(knobSpacing);
    };

    placeKnob(thresholdSlider, thresholdLabel);
    placeKnob(attackSlider, attackLabel);
    placeKnob(releaseSlider, releaseLabel);
    placeKnob(holdSlider, holdLabel);
    placeKnob(rangeSlider, rangeLabel);

    // Enable button at the end
    auto enableBounds = knobArea.removeFromLeft(30);
    enabledButton.setBounds(enableBounds.withSizeKeepingCentre(28, 20).translated(0, 10));
}
