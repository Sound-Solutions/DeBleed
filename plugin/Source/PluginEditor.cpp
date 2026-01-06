#include "PluginEditor.h"

#if DEBUG || JUCE_DEBUG
#include "BuildTimestamp.h"
#endif

DeBleedAudioProcessorEditor::DeBleedAudioProcessorEditor(DeBleedAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      controlPanel_(p)
{
    setLookAndFeel(&customLookAndFeel);

    // Title
    titleLabel.setText("DeBleed", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(18.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    // Bypass button (power button in header)
    bypassButton.setButtonText("");
    bypassButton.getProperties().set("invertColors", true);
    addAndMakeVisible(bypassButton);

    // Parameter attachment
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_BYPASS, bypassButton);

    // Arc meter for GR visualization
    addAndMakeVisible(arcMeter_);

    // Set the range to match expander
    arcMeter_.setRange(-80.0f);

    // Control panel (always visible at bottom)
    addAndMakeVisible(controlPanel_);

    // Start timer for UI updates
    startTimer(50);

    // Set window size - narrower and compact
    setSize(440, 460);
}

DeBleedAudioProcessorEditor::~DeBleedAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

void DeBleedAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Dark background
    g.fillAll(juce::Colour::fromRGB(26, 26, 26));

    // Header background
    auto headerBounds = getLocalBounds().removeFromTop(headerHeight);
    g.setColour(juce::Colours::black.withAlpha(0.4f));
    g.fillRect(headerBounds);

    // Header bottom line
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawHorizontalLine(headerHeight - 1, 0, static_cast<float>(getWidth()));

    // Subtle gradient on main area
    auto mainArea = getLocalBounds();
    mainArea.removeFromTop(headerHeight);
    mainArea.removeFromBottom(controlPanelHeight);

    juce::ColourGradient gradient(
        juce::Colour::fromRGB(30, 30, 32), 0.0f, static_cast<float>(mainArea.getY()),
        juce::Colour::fromRGB(20, 20, 22), 0.0f, static_cast<float>(mainArea.getBottom()),
        false);
    g.setGradientFill(gradient);
    g.fillRect(mainArea);

#if DEBUG || JUCE_DEBUG
    // Draw timestamp in corner
    g.setColour(juce::Colours::white.withAlpha(0.3f));
    g.setFont(9.0f);
    g.drawText("Build: " BUILD_TIMESTAMP,
               getWidth() - 150, getHeight() - 14, 145, 12,
               juce::Justification::centredRight);
#endif
}

void DeBleedAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();

    // Header - 40px
    auto header = bounds.removeFromTop(headerHeight).reduced(12, 8);

    // Title on left
    titleLabel.setBounds(header.removeFromLeft(100));

    // Bypass button on right
    bypassButton.setBounds(header.removeFromRight(28).withSizeKeepingCentre(28, 28));

    // Control panel at bottom
    controlPanel_.setBounds(bounds.removeFromBottom(controlPanelHeight));

    // Arc meter in center (remaining space)
    arcMeter_.setBounds(bounds.reduced(20));
}

void DeBleedAudioProcessorEditor::timerCallback()
{
    // Safety check - don't update if component is being destroyed
    if (!isShowing())
        return;

    // Update arc meter with current GR, output level, and VAD values
    float gr = audioProcessor.getExpander().getGainReduction();
    float level = audioProcessor.getOutputLevelDb();
    float vad = audioProcessor.getVADConfidence();

    arcMeter_.setGainReduction(gr);
    arcMeter_.setOutputLevel(level);
    arcMeter_.setVADConfidence(vad);
}
