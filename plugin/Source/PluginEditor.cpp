#include "PluginEditor.h"

#if DEBUG || JUCE_DEBUG
#include "BuildTimestamp.h"
#endif

DeBleedAudioProcessorEditor::DeBleedAudioProcessorEditor(DeBleedAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      cleanDropZone("Target Vocals", true),
      noiseDropZone("Stage Noise", true),
      progressBar(progressValue)
{
    setLookAndFeel(&customLookAndFeel);

    // Title
    titleLabel.setText("DeBleed", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(20.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel.setJustificationType(juce::Justification::centredLeft);
    addAndMakeVisible(titleLabel);

    // Tab buttons
    trainingTabButton.setButtonText("TRAIN");
    trainingTabButton.getProperties().set("isTabButton", true);
    trainingTabButton.setClickingTogglesState(true);
    trainingTabButton.setToggleState(true, juce::dontSendNotification);
    trainingTabButton.onClick = [this]() { setActiveTab(Tab::Training); };
    addAndMakeVisible(trainingTabButton);

    visualizingTabButton.setButtonText("VISUALIZE");
    visualizingTabButton.getProperties().set("isTabButton", true);
    visualizingTabButton.setClickingTogglesState(true);
    visualizingTabButton.onClick = [this]() { setActiveTab(Tab::Visualizing); };
    addAndMakeVisible(visualizingTabButton);

    // Bypass button (power button in header)
    // invertColors: orange when OFF (active), grey when ON (bypassed)
    bypassButton.setButtonText("");
    bypassButton.getProperties().set("invertColors", true);
    addAndMakeVisible(bypassButton);

    // Live Mode toggle (disables training when in live mode)
    liveModeButton.setButtonText("LIVE");
    liveModeButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white.withAlpha(0.8f));
    // Handle both user clicks and parameter restore
    auto applyLiveModeState = [this]() {
        bool isLive = liveModeButton.getToggleState();
        trainingTabButton.setEnabled(!isLive);
        if (isLive && currentTab == Tab::Training)
        {
            setActiveTab(Tab::Visualizing);
        }
    };
    liveModeButton.onClick = applyLiveModeState;
    liveModeButton.onStateChange = applyLiveModeState;
    addAndMakeVisible(liveModeButton);

    // Drop zones
    cleanDropZone.setDirectorySelectedCallback([this](const juce::String& dir) {
        cleanAudioPath = dir;
        trainNewButton.setEnabled(cleanAudioPath.isNotEmpty() && noiseAudioPath.isNotEmpty());
        updateContinueButtonState();
    });
    addAndMakeVisible(cleanDropZone);

    noiseDropZone.setDirectorySelectedCallback([this](const juce::String& dir) {
        noiseAudioPath = dir;
        trainNewButton.setEnabled(cleanAudioPath.isNotEmpty() && noiseAudioPath.isNotEmpty());
        updateContinueButtonState();
    });
    addAndMakeVisible(noiseDropZone);

    // Progress bar
    addAndMakeVisible(progressBar);

    // Train New button
    trainNewButton.setButtonText("Train New");
    trainNewButton.setEnabled(false);
    trainNewButton.onClick = [this]() { startNewTraining(); };
    addAndMakeVisible(trainNewButton);

    // Continue Training button
    continueTrainingButton.setButtonText("Continue Training");
    continueTrainingButton.setEnabled(false);
    continueTrainingButton.onClick = [this]() {
        DBG("Continue Training button clicked!");
        continueTraining();
    };
    addAndMakeVisible(continueTrainingButton);

    // Load Model button
    loadModelButton.setButtonText("Load Model");
    loadModelButton.onClick = [this]() { loadModel(); };
    addAndMakeVisible(loadModelButton);

    // Status label
    statusLabel.setText("Select audio folders to begin", juce::dontSendNotification);
    statusLabel.setFont(juce::Font(12.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    addAndMakeVisible(statusLabel);

    // Model status
    modelStatusLabel.setFont(juce::Font(11.0f));
    modelStatusLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    addAndMakeVisible(modelStatusLabel);

    // Setup knobs with rotary style
    auto setupKnob = [this](juce::Slider& slider, juce::Label& label, const juce::String& name,
                            juce::uint32 color) {
        slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
        slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 60, 16);
        slider.getProperties().set("knobColor", static_cast<juce::int64>(color));
        addAndMakeVisible(slider);

        label.setText(name, juce::dontSendNotification);
        label.setFont(juce::Font(11.0f));
        label.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.8f));
        label.setJustificationType(juce::Justification::centred);
        addAndMakeVisible(label);
    };

    // Cyan knob
    setupKnob(strengthSlider, strengthLabel, "Strength", DeBleedLookAndFeel::cyanAccent);

    // Threshold - orange
    setupKnob(thresholdSlider, thresholdLabel, "Threshold", DeBleedLookAndFeel::orangeAccent);

    // Orange knobs
    setupKnob(attackSlider, attackLabel, "Attack", DeBleedLookAndFeel::orangeAccent);
    setupKnob(releaseSlider, releaseLabel, "Release", DeBleedLookAndFeel::orangeAccent);

    // Yellow knobs
    setupKnob(freqLowSlider, freqLowLabel, "Freq Lo", DeBleedLookAndFeel::yellowAccent);
    setupKnob(freqHighSlider, freqHighLabel, "Freq Hi", DeBleedLookAndFeel::yellowAccent);
    freqHighSlider.getProperties().set("invertArc", true);  // Fill from right to left

    // Purple knob
    setupKnob(floorSlider, floorLabel, "Range", DeBleedLookAndFeel::purpleAccent);

    // Mix slider (hidden)
    mixSlider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    mixSlider.setVisible(false);
    mixLabel.setVisible(false);

    // Low Latency button
    lowLatencyButton.setButtonText("Low Latency");
    lowLatencyButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white.withAlpha(0.8f));
    addAndMakeVisible(lowLatencyButton);

    // Latency label
    latencyLabel.setFont(juce::Font(10.0f));
    latencyLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.6f));
    addAndMakeVisible(latencyLabel);

    // Visualization components
    rtaView = std::make_unique<RTAVisualization>(audioProcessor);
    addAndMakeVisible(rtaView.get());

    gainReductionMeter = std::make_unique<GainReductionMeter>();
    addAndMakeVisible(gainReductionMeter.get());

    // Parameter attachments
    strengthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_STRENGTH, strengthSlider);
    mixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_MIX, mixSlider);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_BYPASS, bypassButton);
    lowLatencyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_LOW_LATENCY, lowLatencyButton);
    attackAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_ATTACK, attackSlider);
    releaseAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_RELEASE, releaseSlider);
    freqLowAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_FREQ_LOW, freqLowSlider);
    freqHighAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_FREQ_HIGH, freqHighSlider);
    thresholdAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_THRESHOLD, thresholdSlider);
    floorAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_FLOOR, floorSlider);
    liveModeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_LIVE_MODE, liveModeButton);

    // Log text box (hidden by default)
    logTextBox.setMultiLine(true);
    logTextBox.setReadOnly(true);
    logTextBox.setScrollbarsShown(true);
    logTextBox.setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain));

    // Set up trainer callbacks
    audioProcessor.getTrainerProcess().setProgressCallback(
        [this](int progress, const juce::String& status) {
            onTrainingProgress(progress, status);
        });

    audioProcessor.getTrainerProcess().setCompletionCallback(
        [this](bool success, const juce::String& modelPath, const juce::String& error) {
            onTrainingComplete(success, modelPath, error);
        });

    audioProcessor.getTrainerProcess().setLogCallback(
        [this](const juce::String& message) {
            juce::MessageManager::callAsync([this, message]() {
                logTextBox.moveCaretToEnd();
                logTextBox.insertTextAtCaret(message + "\n");
            });
        });

    // Update model status and latency
    updateModelStatus();
    updateLatencyLabel();
    updateContinueButtonState();

    // Start timer for UI updates
    startTimer(100);

    // Set to new larger size
    setSize(900, 600);

    // Initialize tab visibility
    setActiveTab(Tab::Training);
}

DeBleedAudioProcessorEditor::~DeBleedAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

void DeBleedAudioProcessorEditor::setActiveTab(Tab tab)
{
    currentTab = tab;

    trainingTabButton.setToggleState(tab == Tab::Training, juce::dontSendNotification);
    visualizingTabButton.setToggleState(tab == Tab::Visualizing, juce::dontSendNotification);

    // Training tab components
    cleanDropZone.setVisible(tab == Tab::Training);
    noiseDropZone.setVisible(tab == Tab::Training);
    progressBar.setVisible(tab == Tab::Training);
    trainNewButton.setVisible(tab == Tab::Training);
    continueTrainingButton.setVisible(tab == Tab::Training);
    loadModelButton.setVisible(tab == Tab::Training);
    statusLabel.setVisible(tab == Tab::Training);
    modelStatusLabel.setVisible(tab == Tab::Training);
    logTextBox.setVisible(tab == Tab::Training && showLog);

    // Visualizing tab components
    rtaView->setVisible(tab == Tab::Visualizing);
    gainReductionMeter->setVisible(tab == Tab::Visualizing);

    // Knobs (only on visualizing tab)
    strengthSlider.setVisible(tab == Tab::Visualizing);
    strengthLabel.setVisible(tab == Tab::Visualizing);
    attackSlider.setVisible(tab == Tab::Visualizing);
    attackLabel.setVisible(tab == Tab::Visualizing);
    releaseSlider.setVisible(tab == Tab::Visualizing);
    releaseLabel.setVisible(tab == Tab::Visualizing);
    thresholdSlider.setVisible(tab == Tab::Visualizing);
    thresholdLabel.setVisible(tab == Tab::Visualizing);
    freqLowSlider.setVisible(tab == Tab::Visualizing);
    freqLowLabel.setVisible(tab == Tab::Visualizing);
    freqHighSlider.setVisible(tab == Tab::Visualizing);
    freqHighLabel.setVisible(tab == Tab::Visualizing);
    floorSlider.setVisible(tab == Tab::Visualizing);
    floorLabel.setVisible(tab == Tab::Visualizing);
    lowLatencyButton.setVisible(tab == Tab::Visualizing);
    latencyLabel.setVisible(tab == Tab::Visualizing);

    resized();
    repaint();
}

void DeBleedAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Dark background
    g.fillAll(juce::Colour(DeBleedLookAndFeel::mainBackground));

    // Header background
    auto headerBounds = getLocalBounds().removeFromTop(50);
    g.setColour(juce::Colour(DeBleedLookAndFeel::panelBackground));
    g.fillRect(headerBounds);

    // Header bottom line
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawHorizontalLine(49, 0, static_cast<float>(getWidth()));

#if DEBUG || JUCE_DEBUG
    g.setColour(juce::Colours::grey.withAlpha(0.5f));
    g.setFont(9.0f);
    g.drawText("Build: " BUILD_TIMESTAMP,
               getLocalBounds().removeFromBottom(14),
               juce::Justification::centredRight);
#endif
}

void DeBleedAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();

    // Header - 50px
    auto header = bounds.removeFromTop(50);
    header = header.reduced(15, 10);

    // Title on left
    titleLabel.setBounds(header.removeFromLeft(100));

    // Bypass button on right (power icon - 30x30)
    bypassButton.setBounds(header.removeFromRight(30).withSizeKeepingCentre(30, 30));
    header.removeFromRight(8);

    // Live mode toggle
    liveModeButton.setBounds(header.removeFromRight(60).withSizeKeepingCentre(60, 24));
    header.removeFromRight(15);

    // Tab buttons in center
    auto tabArea = header.withSizeKeepingCentre(200, 30);
    trainingTabButton.setBounds(tabArea.removeFromLeft(95));
    tabArea.removeFromLeft(10);
    visualizingTabButton.setBounds(tabArea);

    // Content area
    auto content = bounds.reduced(15);

    if (currentTab == Tab::Training)
        layoutTrainingTab(content);
    else
        layoutVisualizingTab(content);
}

void DeBleedAudioProcessorEditor::layoutTrainingTab(juce::Rectangle<int> bounds)
{
    // Drop zones - side by side, larger
    auto dropZoneArea = bounds.removeFromTop(180);
    int dropZoneWidth = (dropZoneArea.getWidth() - 20) / 2;

    cleanDropZone.setBounds(dropZoneArea.removeFromLeft(dropZoneWidth));
    dropZoneArea.removeFromLeft(20);
    noiseDropZone.setBounds(dropZoneArea);

    bounds.removeFromTop(20);

    // Progress bar
    progressBar.setBounds(bounds.removeFromTop(22));
    bounds.removeFromTop(10);

    // Status label
    statusLabel.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(15);

    // Buttons row
    auto buttonRow = bounds.removeFromTop(32);
    trainNewButton.setBounds(buttonRow.removeFromLeft(110));
    buttonRow.removeFromLeft(10);
    continueTrainingButton.setBounds(buttonRow.removeFromLeft(130));
    buttonRow.removeFromLeft(10);
    loadModelButton.setBounds(buttonRow.removeFromLeft(100));

    bounds.removeFromTop(15);

    // Model status
    modelStatusLabel.setBounds(bounds.removeFromTop(20));

    // Log (if shown)
    if (showLog && bounds.getHeight() > 100)
    {
        bounds.removeFromTop(15);
        logTextBox.setBounds(bounds);
        logTextBox.setVisible(true);
    }
}

void DeBleedAudioProcessorEditor::layoutVisualizingTab(juce::Rectangle<int> bounds)
{
    // Knob area at bottom - 120px for knobs + labels
    auto knobArea = bounds.removeFromBottom(120);

    // RTA and GR meter
    auto vizArea = bounds;
    int meterWidth = 40;

    rtaView->setBounds(vizArea.removeFromLeft(vizArea.getWidth() - meterWidth - 15));
    vizArea.removeFromLeft(15);
    gainReductionMeter->setBounds(vizArea);

    // Knob row layout - 7 knobs + toggles
    int knobSize = 70;
    int knobSpacing = 15;
    int totalKnobWidth = 7 * knobSize + 6 * knobSpacing;

    auto knobRow = knobArea.withSizeKeepingCentre(totalKnobWidth + 150, knobArea.getHeight());

    // Helper to place knob + label
    auto placeKnob = [knobSize](juce::Rectangle<int>& area, juce::Slider& slider, juce::Label& label) {
        auto knobBounds = area.removeFromLeft(knobSize);
        label.setBounds(knobBounds.removeFromTop(16));
        slider.setBounds(knobBounds);
    };

    placeKnob(knobRow, strengthSlider, strengthLabel);
    knobRow.removeFromLeft(knobSpacing);
    placeKnob(knobRow, attackSlider, attackLabel);
    knobRow.removeFromLeft(knobSpacing);
    placeKnob(knobRow, releaseSlider, releaseLabel);
    knobRow.removeFromLeft(knobSpacing);
    placeKnob(knobRow, thresholdSlider, thresholdLabel);
    knobRow.removeFromLeft(knobSpacing);
    placeKnob(knobRow, freqLowSlider, freqLowLabel);
    knobRow.removeFromLeft(knobSpacing);
    placeKnob(knobRow, freqHighSlider, freqHighLabel);
    knobRow.removeFromLeft(knobSpacing);
    placeKnob(knobRow, floorSlider, floorLabel);

    // Toggles on the right
    knobRow.removeFromLeft(20);
    auto toggleArea = knobRow.removeFromLeft(100);
    lowLatencyButton.setBounds(toggleArea.removeFromTop(25));
    toggleArea.removeFromTop(5);
    latencyLabel.setBounds(toggleArea.removeFromTop(16));
}

void DeBleedAudioProcessorEditor::timerCallback()
{
    // Update progress bar
    auto& trainer = audioProcessor.getTrainerProcess();

    if (trainer.isTraining())
    {
        progressValue = trainer.getProgress() / 100.0;
        progressBar.repaint();
    }

    // Update visualization (only when visible)
    if (currentTab == Tab::Visualizing)
    {
        if (rtaView)
            rtaView->updateFromQueue();

        if (gainReductionMeter)
        {
            float reductionDb = audioProcessor.getVisualizationData().averageGainReductionDb.load();
            gainReductionMeter->setReductionLevel(reductionDb);
            gainReductionMeter->repaint();
        }

        updateLatencyLabel();
    }

    updateModelStatus();
}

void DeBleedAudioProcessorEditor::startNewTraining()
{
    if (cleanAudioPath.isEmpty() || noiseAudioPath.isEmpty())
        return;

    // Ask for model name
    auto* nameDialog = new juce::AlertWindow("Name Your Model",
                                              "Enter a name for this trained model:",
                                              juce::AlertWindow::QuestionIcon);
    nameDialog->addTextEditor("modelName", "", "Model Name:");
    nameDialog->addButton("Train", 1, juce::KeyPress(juce::KeyPress::returnKey));
    nameDialog->addButton("Cancel", 0, juce::KeyPress(juce::KeyPress::escapeKey));

    nameDialog->enterModalState(true, juce::ModalCallbackFunction::create(
        [this, nameDialog](int result)
        {
            if (result == 1)
            {
                juce::String modelName = nameDialog->getTextEditorContents("modelName").trim();
                if (modelName.isEmpty())
                    modelName = juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S");

                modelName = modelName.replaceCharacters(" /\\:*?\"<>|", "___________");
                startTrainingWithName(modelName, false);
            }
            delete nameDialog;
        }), true);
}

void DeBleedAudioProcessorEditor::continueTraining()
{
    DBG("continueTraining() called");

    if (!audioProcessor.isModelLoaded())
    {
        DBG("No model loaded - showing alert");
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "No Model Loaded",
            "Please load a model first before continuing training.",
            "OK");
        return;
    }

    // Use existing model directory for continuation
    juce::File modelFile(audioProcessor.getModelPath());
    juce::String modelDir = modelFile.getParentDirectory().getFullPathName();

    // Check if checkpoint exists
    juce::File checkpointFile = modelFile.getParentDirectory().getChildFile("checkpoint.pt");
    if (!checkpointFile.exists())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "No Checkpoint Found",
            "This model doesn't have a checkpoint file for continuation.\n\n"
            "You can only continue training on models that were trained with checkpoint support.",
            "OK");
        return;
    }

    lastTrainedModelDir = modelDir;
    startTrainingWithName(modelFile.getParentDirectory().getFileName(), true);
}

void DeBleedAudioProcessorEditor::startTrainingWithName(const juce::String& modelName, bool isContinuation)
{
    juce::File outputDir;

    if (isContinuation)
    {
        outputDir = juce::File(lastTrainedModelDir);
    }
    else
    {
        outputDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                        .getChildFile("DeBleed")
                        .getChildFile("Models")
                        .getChildFile(modelName);

        if (outputDir.exists())
        {
            outputDir = outputDir.getSiblingFile(modelName + "_" +
                            juce::Time::getCurrentTime().formatted("%H%M%S"));
        }

        outputDir.createDirectory();
    }

    // Clear log
    logTextBox.clear();
    showLog = true;
    addAndMakeVisible(logTextBox);
    resized();

    // Start training
    trainNewButton.setEnabled(false);
    continueTrainingButton.setEnabled(false);
    trainNewButton.setButtonText(isContinuation ? "Continuing..." : "Training...");
    statusLabel.setText("Preparing training...", juce::dontSendNotification);
    progressValue = 0.0;

    bool started = audioProcessor.getTrainerProcess().startTraining(
        cleanAudioPath,
        noiseAudioPath,
        outputDir.getFullPathName(),
        modelName,  // Model file name (matches folder name)
        isContinuation ? 25 : 50,  // fewer epochs for continuation
        isContinuation  // pass continuation flag
    );

    if (!started)
    {
        statusLabel.setText("Failed to start training: " +
                           audioProcessor.getTrainerProcess().getLastError(),
                           juce::dontSendNotification);
        trainNewButton.setEnabled(true);
        trainNewButton.setButtonText("Train New");
        updateContinueButtonState();
    }
}

void DeBleedAudioProcessorEditor::loadModel()
{
    juce::File defaultDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                                .getChildFile("DeBleed")
                                .getChildFile("Models");

    if (!defaultDir.exists())
        defaultDir = juce::File::getSpecialLocation(juce::File::userHomeDirectory);

    auto chooser = std::make_shared<juce::FileChooser>(
        "Select ONNX Model",
        defaultDir,
        "*.onnx"
    );

    chooser->launchAsync(juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles,
        [this, chooser](const juce::FileChooser& fc)
        {
            auto result = fc.getResult();
            if (result.existsAsFile())
            {
                if (audioProcessor.loadModel(result.getFullPathName()))
                {
                    statusLabel.setText("Model loaded: " + result.getFileName(), juce::dontSendNotification);
                    lastTrainedModelDir = result.getParentDirectory().getFullPathName();
                    updateModelStatus();
                    updateContinueButtonState();
                }
                else
                {
                    statusLabel.setText("Failed to load model", juce::dontSendNotification);
                    juce::AlertWindow::showMessageBoxAsync(
                        juce::AlertWindow::WarningIcon,
                        "Load Failed",
                        "Failed to load the selected model file.\n\nMake sure it's a valid DeBleed ONNX model.",
                        "OK"
                    );
                }
            }
        });
}

void DeBleedAudioProcessorEditor::onTrainingProgress(int progress, const juce::String& status)
{
    juce::MessageManager::callAsync([this, progress, status]() {
        progressValue = progress / 100.0;
        statusLabel.setText(status, juce::dontSendNotification);
        progressBar.repaint();
    });
}

void DeBleedAudioProcessorEditor::onTrainingComplete(bool success, const juce::String& modelPath,
                                                      const juce::String& error)
{
    juce::MessageManager::callAsync([this, success, modelPath, error]() {
        trainNewButton.setEnabled(cleanAudioPath.isNotEmpty() && noiseAudioPath.isNotEmpty());
        trainNewButton.setButtonText("Train New");

        if (success)
        {
            statusLabel.setText("Training complete! Model loaded.", juce::dontSendNotification);
            progressValue = 1.0;

            // Load the new model
            audioProcessor.loadModel(modelPath);

            // Store the model directory for continue training
            juce::File modelFile(modelPath);
            lastTrainedModelDir = modelFile.getParentDirectory().getFullPathName();

            updateModelStatus();
            updateContinueButtonState();

            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::InfoIcon,
                "Training Complete",
                "Model trained successfully and loaded!\n\nModel saved to:\n" + modelPath,
                "OK"
            );
        }
        else
        {
            statusLabel.setText("Training failed: " + error, juce::dontSendNotification);
            progressValue = 0.0;

            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::WarningIcon,
                "Training Failed",
                error,
                "OK"
            );
        }

        progressBar.repaint();
    });
}

void DeBleedAudioProcessorEditor::updateModelStatus()
{
    if (audioProcessor.isModelLoaded())
    {
        juce::File modelFile(audioProcessor.getModelPath());
        modelStatusLabel.setText("Model: " + modelFile.getFileName(),
                                  juce::dontSendNotification);
        modelStatusLabel.setColour(juce::Label::textColourId, juce::Colour(0xff5aaa6c));
    }
    else
    {
        modelStatusLabel.setText("Model: Not loaded", juce::dontSendNotification);
        modelStatusLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.5f));
    }
}

void DeBleedAudioProcessorEditor::updateLatencyLabel()
{
    float latencyMs = audioProcessor.getLatencyMs();
    juce::String latencyText = juce::String(latencyMs, 1) + " ms";
    latencyLabel.setText(latencyText, juce::dontSendNotification);
}

void DeBleedAudioProcessorEditor::updateContinueButtonState()
{
    bool canContinue = audioProcessor.isModelLoaded();
    DBG("updateContinueButtonState: modelLoaded=" << (canContinue ? "true" : "false"));

    if (canContinue)
    {
        // Check if checkpoint exists
        juce::File modelFile(audioProcessor.getModelPath());
        juce::File checkpointFile = modelFile.getParentDirectory().getChildFile("checkpoint.pt");
        DBG("Looking for checkpoint at: " << checkpointFile.getFullPathName());
        canContinue = checkpointFile.exists();
        DBG("Checkpoint exists: " << (canContinue ? "true" : "false"));
    }

    DBG("Setting continueTrainingButton enabled: " << (canContinue ? "true" : "false"));
    continueTrainingButton.setEnabled(canContinue);
}
