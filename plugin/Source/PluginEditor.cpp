#include "PluginEditor.h"

#if DEBUG || JUCE_DEBUG
#include "BuildTimestamp.h"
#endif

DeBleedAudioProcessorEditor::DeBleedAudioProcessorEditor(DeBleedAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      expanderSection_(p.getParameters()),
      vocalSection_(p.getParameters()),
      outputSection_(p.getParameters())
{
    setLookAndFeel(&customLookAndFeel);
    setOpaque(true);
    setResizable(false, false);

    addAndMakeVisible(arcMeter_);
    arcMeter_.setRange(-80.0f);
    addAndMakeVisible(expanderSection_);
    addAndMakeVisible(vocalSection_);
    addAndMakeVisible(outputSection_);

    powerButton_.setTitle("Power");
    powerButton_.getProperties().set("ringStyle", true);
    powerButton_.getProperties().set("ringColour", static_cast<juce::int64>(DeBleedLookAndFeel::orangeAccent));
    addAndMakeVisible(powerButton_);

    // Power is on when the existing bypass parameter is off, including host automation.
    auto& parameters = audioProcessor.getParameters();
    bypassAttachment_ = std::make_unique<juce::ParameterAttachment>(
        *parameters.getParameter(DeBleedAudioProcessor::PARAM_BYPASS),
        [this](float bypass)
        {
            powerButton_.setToggleState(bypass < 0.5f, juce::dontSendNotification);
        }, parameters.undoManager);
    powerButton_.onClick = [this]
    {
        bypassAttachment_->setValueAsCompleteGesture(powerButton_.getToggleState() ? 0.0f : 1.0f);
    };
    bypassAttachment_->sendInitialUpdate();

    collapsed_ = static_cast<bool>(parameters.state.getProperty("editorCollapsed", false));
    chevronButton_.setTitle("Collapse controls");
    chevronButton_.getProperties().set("chevron", true);
    chevronButton_.getProperties().set("pointsLeft", !collapsed_);
    chevronButton_.setToggleState(collapsed_, juce::dontSendNotification);
    chevronButton_.onClick = [this] { toggleCollapsed(); };
    addAndMakeVisible(chevronButton_);

    slideTimer_.onTick = [this] { slideFrame(); };
    setSize(collapsed_ ? collapsedWidth : expandedWidth, editorHeight);
    startTimer(50);  // meter feed, unchanged from v1.1.0 so the spring moves exactly as before
}

DeBleedAudioProcessorEditor::~DeBleedAudioProcessorEditor()
{
    slideTimer_.stopTimer();
    stopTimer();
    setLookAndFeel(nullptr);
}

void DeBleedAudioProcessorEditor::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(DeBleedLookAndFeel::bodyBackground));
    g.setGradientFill(juce::ColourGradient(juce::Colour(DeBleedLookAndFeel::meterBackground), 134.0f, 134.0f,
                                         juce::Colour(DeBleedLookAndFeel::mainBackground), 335.0f, 134.0f, true));
    g.fillRect(0, 0, collapsedWidth, editorHeight);

    g.setColour(juce::Colour(DeBleedLookAndFeel::groove));
    g.fillRect(268, 0, 1, 268);
    g.fillRect(270, 130, 590, 1);
    g.fillRect(596, 132, 1, 136);
    g.setColour(juce::Colours::white.withAlpha(0.05f));
    g.fillRect(269, 0, 1, 268);
    g.fillRect(270, 131, 590, 1);
    g.fillRect(597, 132, 1, 136);

    g.setColour(juce::Colour(DeBleedLookAndFeel::labelText));
    const auto titleFont = DeBleedLookAndFeel::font(10.0f, "Bold")
                               .withExtraKerningFactor(2.6f / 10.0f);
    DeBleedLookAndFeel::drawTextAtBaseline(g, "DEBLEED", titleFont, 16.0f, 21.0f, false);

#if DEBUG || JUCE_DEBUG
    g.setColour(juce::Colour(DeBleedLookAndFeel::inactiveRing));
    DeBleedLookAndFeel::drawTextAtBaseline(g, "Build: " BUILD_TIMESTAMP,
                                          DeBleedLookAndFeel::font(8.0f, "Regular"), 16.0f, 260.0f, false);
#endif
}

void DeBleedAudioProcessorEditor::resized()
{
    // Fixed expanded coordinates: resizing the window reveals or covers the controls.
    arcMeter_.setBounds(0, 0, 268, 268);
    powerButton_.setBounds(228, 0, 36, 36);      // centre (246, 18), room for the glow
    chevronButton_.setBounds(228, 232, 36, 36);  // centre (246, 250)
    expanderSection_.setBounds(270, 0, 590, 130);
    vocalSection_.setBounds(270, 132, 326, 136);
    outputSection_.setBounds(598, 132, 262, 136);
}

void DeBleedAudioProcessorEditor::toggleCollapsed()
{
    collapsed_ = chevronButton_.getToggleState();
    audioProcessor.getParameters().state.setProperty("editorCollapsed", collapsed_, nullptr);
    chevronButton_.getProperties().set("pointsLeft", !collapsed_);
    chevronButton_.repaint();
    slideStartWidth_ = getWidth();
    slideStartTime_ = juce::Time::getMillisecondCounterHiRes();
    sliding_ = true;
    slideTimer_.startTimer(16);
}

void DeBleedAudioProcessorEditor::slideFrame()
{
    // Finish the slide even if the host hides the window during the animation.
    if (!sliding_)
    {
        slideTimer_.stopTimer();
        return;
    }
    const int targetWidth = collapsed_ ? collapsedWidth : expandedWidth;
    const double progress = juce::jlimit(0.0, 1.0,
        (juce::Time::getMillisecondCounterHiRes() - slideStartTime_) / slideDurationMs);
    if (progress >= 1.0)
    {
        setSize(targetWidth, editorHeight);
        sliding_ = false;
        slideTimer_.stopTimer();
    }
    else
    {
        const double remaining = 1.0 - progress;
        const double eased = 1.0 - remaining * remaining * remaining;
        setSize(juce::roundToInt(slideStartWidth_ + (targetWidth - slideStartWidth_) * eased), editorHeight);
    }
}

void DeBleedAudioProcessorEditor::timerCallback()
{
    if (!isShowing())
        return;

    float gr = audioProcessor.getExpander().getGainReduction();
    float level = audioProcessor.getOutputLevelDb();
    float vad = audioProcessor.getVADConfidence();

    arcMeter_.setGainReduction(gr);
    arcMeter_.setOutputLevel(level);
    arcMeter_.setVADConfidence(vad);
}
