#include "PluginEditor.h"

#if DEBUG || JUCE_DEBUG
#include "BuildTimestamp.h"
#endif

DeBleedAudioProcessorEditor::DeBleedAudioProcessorEditor(DeBleedAudioProcessor& p)
    : AudioProcessorEditor(&p),
      audioProcessor(p),
      cleanDropZone("Target Vocals", true),
      noiseDropZone("Stage Noise", true),
      progressBar(progressValue),
      controlPanel_(p)
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
    trainingTabButton.onClick = [this]() { setActiveTab(Tab::Training); };
    addAndMakeVisible(trainingTabButton);

    visualizingTabButton.setButtonText("VISUALIZE");
    visualizingTabButton.getProperties().set("isTabButton", true);
    visualizingTabButton.setClickingTogglesState(true);
    visualizingTabButton.setToggleState(true, juce::dontSendNotification);  // Default tab
    visualizingTabButton.onClick = [this]() { setActiveTab(Tab::Visualizing); };
    addAndMakeVisible(visualizingTabButton);

    // Bypass button (power button in header)
    bypassButton.setButtonText("");
    bypassButton.getProperties().set("invertColors", true);
    addAndMakeVisible(bypassButton);

    // Live Mode toggle
    liveModeButton.setButtonText("LIVE");
    liveModeButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white.withAlpha(0.8f));
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
    continueTrainingButton.onClick = [this]() { continueTraining(); };
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

    // Parameter attachments
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_BYPASS, bypassButton);
    liveModeAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_LIVE_MODE, liveModeButton);

    // Log text box (hidden by default)
    logTextBox.setMultiLine(true);
    logTextBox.setReadOnly(true);
    logTextBox.setScrollbarsShown(true);
    logTextBox.setFont(juce::Font(juce::Font::getDefaultMonospacedFontName(), 10.0f, juce::Font::plain));

    // EQ Curve display for visualizing tab
    addAndMakeVisible(eqCurveDisplay_);

    // Connect EQ curve handle dragging to parameters
    eqCurveDisplay_.setHPFChangedCallback([this](float freq) {
        if (auto* param = audioProcessor.getParameters().getParameter(DeBleedAudioProcessor::PARAM_HPF_FREQ))
        {
            param->setValueNotifyingHost(param->convertTo0to1(freq));
        }
    });

    eqCurveDisplay_.setLPFChangedCallback([this](float freq) {
        if (auto* param = audioProcessor.getParameters().getParameter(DeBleedAudioProcessor::PARAM_LPF_FREQ))
        {
            param->setValueNotifyingHost(param->convertTo0to1(freq));
        }
    });

    // Control panel (always visible at bottom)
    addAndMakeVisible(controlPanel_);

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

    // Update model status
    updateModelStatus();
    updateContinueButtonState();

    // Start timer for UI updates
    startTimer(100);

    // Set window size - Kinetics style
    setSize(1100, 700);

    // Initialize tab visibility
    setActiveTab(Tab::Visualizing);
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
    eqCurveDisplay_.setVisible(tab == Tab::Visualizing);

    // Control panel is always visible
    controlPanel_.setVisible(true);

    resized();
    repaint();
}

void DeBleedAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Dark background (Kinetics style)
    g.fillAll(juce::Colour::fromRGB(10, 11, 13));

    // Header background
    auto headerBounds = getLocalBounds().removeFromTop(headerHeight);
    g.setColour(juce::Colours::black.withAlpha(0.3f));
    g.fillRect(headerBounds);

    // Header bottom line
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawHorizontalLine(headerHeight - 1, 0, static_cast<float>(getWidth()));

#if DEBUG || JUCE_DEBUG
    g.setColour(juce::Colours::white.withAlpha(0.3f));
    g.setFont(9.0f);
    g.drawText("Build: " BUILD_TIMESTAMP,
               getWidth() - 140, 0, 130, headerHeight,
               juce::Justification::centredRight);
#endif
}

void DeBleedAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds();

    // Header - 50px
    auto header = bounds.removeFromTop(headerHeight).reduced(15, 10);

    // Title on left
    titleLabel.setBounds(header.removeFromLeft(100));

    // Bypass button on right
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

    // Control panel at bottom
    controlPanel_.setBounds(bounds.removeFromBottom(controlPanelHeight));

    // Main content area
    auto content = bounds.reduced(15);

    if (currentTab == Tab::Training)
        layoutTrainingTab(content);
    else
        layoutVisualizingTab(content);
}

void DeBleedAudioProcessorEditor::layoutTrainingTab(juce::Rectangle<int> bounds)
{
    // Drop zones - side by side
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
    // EQ curve display takes the full content area
    eqCurveDisplay_.setBounds(bounds);
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

    // Update EQ curve visualization from the biquad chain
    if (currentTab == Tab::Visualizing)
    {
        eqCurveDisplay_.updateFromChain(audioProcessor.getBiquadChain());

        // Sync handle positions with current parameter values
        float hpfFreq = *audioProcessor.getParameters().getRawParameterValue(DeBleedAudioProcessor::PARAM_HPF_FREQ);
        float lpfFreq = *audioProcessor.getParameters().getRawParameterValue(DeBleedAudioProcessor::PARAM_LPF_FREQ);
        eqCurveDisplay_.setFilterFrequencies(hpfFreq, lpfFreq);
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
    if (!audioProcessor.isModelLoaded())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "No Model Loaded",
            "Please load a model first before continuing training.",
            "OK");
        return;
    }

    juce::File modelFile(audioProcessor.getModelPath());
    juce::String modelDir = modelFile.getParentDirectory().getFullPathName();

    juce::File checkpointFile = modelFile.getParentDirectory().getChildFile("checkpoint.pt");
    if (!checkpointFile.exists())
    {
        juce::AlertWindow::showMessageBoxAsync(juce::AlertWindow::WarningIcon,
            "No Checkpoint Found",
            "This model doesn't have a checkpoint file for continuation.",
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

    // Pro quality: 150 epochs for new training, 75 for continuation
    bool started = audioProcessor.getTrainerProcess().startTraining(
        cleanAudioPath,
        noiseAudioPath,
        outputDir.getFullPathName(),
        modelName,
        isContinuation ? 75 : 150,
        isContinuation
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
                        "Failed to load the selected model file.",
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

            audioProcessor.loadModel(modelPath);

            juce::File modelFile(modelPath);
            lastTrainedModelDir = modelFile.getParentDirectory().getFullPathName();

            updateModelStatus();
            updateContinueButtonState();

            juce::AlertWindow::showMessageBoxAsync(
                juce::AlertWindow::InfoIcon,
                "Training Complete",
                "Model trained successfully!\n\nSaved to:\n" + modelPath,
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

void DeBleedAudioProcessorEditor::updateContinueButtonState()
{
    bool canContinue = audioProcessor.isModelLoaded();

    if (canContinue)
    {
        juce::File modelFile(audioProcessor.getModelPath());
        juce::File checkpointFile = modelFile.getParentDirectory().getChildFile("checkpoint.pt");
        canContinue = checkpointFile.exists();
    }

    continueTrainingButton.setEnabled(canContinue);
}
