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
    // Set up look and feel with custom colors
    lookAndFeel.setColour(juce::Slider::thumbColourId, juce::Colour(0xff5a9ad8));
    lookAndFeel.setColour(juce::Slider::trackColourId, juce::Colour(0xff3a5a7c));
    lookAndFeel.setColour(juce::Slider::backgroundColourId, juce::Colour(0xff2a3a4c));
    lookAndFeel.setColour(juce::ProgressBar::foregroundColourId, juce::Colour(0xff5a9ad8));
    lookAndFeel.setColour(juce::ProgressBar::backgroundColourId, juce::Colour(0xff2a3a4c));
    setLookAndFeel(&lookAndFeel);

    // Title
    titleLabel.setText("DeBleed - Neural Gate", juce::dontSendNotification);
    titleLabel.setFont(juce::Font(22.0f, juce::Font::bold));
    titleLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    titleLabel.setJustificationType(juce::Justification::centred);
    addAndMakeVisible(titleLabel);

    // Drop zones
    cleanDropZone.setDirectorySelectedCallback([this](const juce::String& dir) {
        cleanAudioPath = dir;
        trainButton.setEnabled(cleanAudioPath.isNotEmpty() && noiseAudioPath.isNotEmpty());
    });
    addAndMakeVisible(cleanDropZone);

    noiseDropZone.setDirectorySelectedCallback([this](const juce::String& dir) {
        noiseAudioPath = dir;
        trainButton.setEnabled(cleanAudioPath.isNotEmpty() && noiseAudioPath.isNotEmpty());
    });
    addAndMakeVisible(noiseDropZone);

    // Progress bar
    addAndMakeVisible(progressBar);

    // Train button
    trainButton.setButtonText("Train Model");
    trainButton.setEnabled(false);
    trainButton.onClick = [this]() { startTraining(); };
    trainButton.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff4a7a5c));
    trainButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff3a6a4c));
    addAndMakeVisible(trainButton);

    // Load Model button
    loadModelButton.setButtonText("Load Model");
    loadModelButton.onClick = [this]() { loadModel(); };
    loadModelButton.setColour(juce::TextButton::buttonColourId, juce::Colour(0xff5a7a9c));
    loadModelButton.setColour(juce::TextButton::buttonOnColourId, juce::Colour(0xff4a6a8c));
    addAndMakeVisible(loadModelButton);

    // Status label
    statusLabel.setText("Select audio folders to begin", juce::dontSendNotification);
    statusLabel.setFont(juce::Font(12.0f));
    statusLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    addAndMakeVisible(statusLabel);

    // Strength slider
    strengthSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    strengthSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(strengthSlider);

    strengthLabel.setText("Strength", juce::dontSendNotification);
    strengthLabel.setFont(juce::Font(12.0f));
    strengthLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    strengthLabel.attachToComponent(&strengthSlider, true);
    addAndMakeVisible(strengthLabel);

    // Mix slider
    mixSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    mixSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 50, 20);
    addAndMakeVisible(mixSlider);

    mixLabel.setText("Mix", juce::dontSendNotification);
    mixLabel.setFont(juce::Font(12.0f));
    mixLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    mixLabel.attachToComponent(&mixSlider, true);
    addAndMakeVisible(mixLabel);

    // Bypass button
    bypassButton.setButtonText("Bypass");
    bypassButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    addAndMakeVisible(bypassButton);

    // Low Latency button
    lowLatencyButton.setButtonText("Low Latency");
    lowLatencyButton.setColour(juce::ToggleButton::textColourId, juce::Colours::white);
    addAndMakeVisible(lowLatencyButton);

    // Latency label
    latencyLabel.setFont(juce::Font(11.0f));
    latencyLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    addAndMakeVisible(latencyLabel);

    // Model status
    modelStatusLabel.setFont(juce::Font(11.0f));
    modelStatusLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    modelStatusLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(modelStatusLabel);

    // New parameter sliders - Attack
    attackSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    attackSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 55, 20);
    addAndMakeVisible(attackSlider);
    attackLabel.setText("Attack", juce::dontSendNotification);
    attackLabel.setFont(juce::Font(11.0f));
    attackLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(attackLabel);

    // Release
    releaseSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    releaseSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 55, 20);
    addAndMakeVisible(releaseSlider);
    releaseLabel.setText("Release", juce::dontSendNotification);
    releaseLabel.setFont(juce::Font(11.0f));
    releaseLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(releaseLabel);

    // Freq Low
    freqLowSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    freqLowSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 55, 20);
    addAndMakeVisible(freqLowSlider);
    freqLowLabel.setText("Freq Lo", juce::dontSendNotification);
    freqLowLabel.setFont(juce::Font(11.0f));
    freqLowLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(freqLowLabel);

    // Freq High
    freqHighSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    freqHighSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 55, 20);
    addAndMakeVisible(freqHighSlider);
    freqHighLabel.setText("Freq Hi", juce::dontSendNotification);
    freqHighLabel.setFont(juce::Font(11.0f));
    freqHighLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(freqHighLabel);

    // Threshold
    thresholdSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    thresholdSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 55, 20);
    addAndMakeVisible(thresholdSlider);
    thresholdLabel.setText("Threshold", juce::dontSendNotification);
    thresholdLabel.setFont(juce::Font(11.0f));
    thresholdLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(thresholdLabel);

    // Range (depth of attenuation when gated)
    floorSlider.setSliderStyle(juce::Slider::LinearHorizontal);
    floorSlider.setTextBoxStyle(juce::Slider::TextBoxRight, false, 55, 20);
    addAndMakeVisible(floorSlider);
    floorLabel.setText("Range", juce::dontSendNotification);
    floorLabel.setFont(juce::Font(11.0f));
    floorLabel.setColour(juce::Label::textColourId, juce::Colours::white);
    addAndMakeVisible(floorLabel);

    // Visualization components
    rtaView = std::make_unique<RTAVisualization>(audioProcessor);
    addAndMakeVisible(rtaView.get());

    gainReductionMeter = std::make_unique<GainReductionMeter>();
    addAndMakeVisible(gainReductionMeter.get());

    // Parameter attachments - existing
    strengthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_STRENGTH, strengthSlider);
    mixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_MIX, mixSlider);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_BYPASS, bypassButton);
    lowLatencyAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_LOW_LATENCY, lowLatencyButton);

    // Parameter attachments - new
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

    // Log text box (hidden by default)
    logTextBox.setMultiLine(true);
    logTextBox.setReadOnly(true);
    logTextBox.setScrollbarsShown(true);
    logTextBox.setColour(juce::TextEditor::backgroundColourId, juce::Colour(0xff1a2a3c));
    logTextBox.setColour(juce::TextEditor::textColourId, juce::Colours::white.withAlpha(0.8f));
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

    // Start timer for UI updates
    startTimer(100);

    // Set size (expanded for visualization and new params)
    setSize(600, 580);
}

DeBleedAudioProcessorEditor::~DeBleedAudioProcessorEditor()
{
    stopTimer();
    setLookAndFeel(nullptr);
}

void DeBleedAudioProcessorEditor::paint(juce::Graphics& g)
{
    // Background gradient
    juce::ColourGradient gradient(
        juce::Colour(0xff1a2a3c), 0.0f, 0.0f,
        juce::Colour(0xff0a1a2c), 0.0f, static_cast<float>(getHeight()),
        false
    );
    g.setGradientFill(gradient);
    g.fillAll();

    // Section dividers
    g.setColour(juce::Colour(0xff3a4a5c));

    // Below title
    g.drawLine(10.0f, 50.0f, getWidth() - 10.0f, 50.0f, 1.0f);

    // Above controls
    g.drawLine(10.0f, 280.0f, getWidth() - 10.0f, 280.0f, 1.0f);

#if DEBUG || JUCE_DEBUG
    g.setColour(juce::Colours::grey);
    g.setFont(10.0f);
    g.drawText("Build: " BUILD_TIMESTAMP,
               getLocalBounds().removeFromBottom(16),
               juce::Justification::centredRight);
#endif
}

void DeBleedAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(15);

    // Title
    titleLabel.setBounds(bounds.removeFromTop(35));

    bounds.removeFromTop(10);

    // Drop zones (side by side)
    auto dropZoneArea = bounds.removeFromTop(100);
    int dropZoneWidth = (dropZoneArea.getWidth() - 15) / 2;

    cleanDropZone.setBounds(dropZoneArea.removeFromLeft(dropZoneWidth));
    dropZoneArea.removeFromLeft(15);
    noiseDropZone.setBounds(dropZoneArea);

    bounds.removeFromTop(10);

    // Progress and training
    progressBar.setBounds(bounds.removeFromTop(18));
    bounds.removeFromTop(8);

    auto trainRow = bounds.removeFromTop(28);
    trainButton.setBounds(trainRow.removeFromLeft(110));
    trainRow.removeFromLeft(8);
    loadModelButton.setBounds(trainRow.removeFromLeft(95));
    trainRow.removeFromLeft(8);
    statusLabel.setBounds(trainRow);

    bounds.removeFromTop(12);

    // Visualization area
    auto vizArea = bounds.removeFromTop(130);
    int meterWidth = 50;
    rtaView->setBounds(vizArea.removeFromLeft(vizArea.getWidth() - meterWidth - 10));
    vizArea.removeFromLeft(10);
    gainReductionMeter->setBounds(vizArea);

    bounds.removeFromTop(15);

    // Parameter controls - 3 rows, 3 columns
    int labelWidth = 55;
    int sliderWidth = 120;
    int columnWidth = labelWidth + sliderWidth + 10;
    int rowHeight = 24;
    int rowGap = 6;

    // Row 1: Strength, Attack, Release
    auto row1 = bounds.removeFromTop(rowHeight);
    strengthLabel.setBounds(row1.removeFromLeft(labelWidth));
    strengthSlider.setBounds(row1.removeFromLeft(sliderWidth));
    row1.removeFromLeft(15);
    attackLabel.setBounds(row1.removeFromLeft(labelWidth));
    attackSlider.setBounds(row1.removeFromLeft(sliderWidth));
    row1.removeFromLeft(15);
    releaseLabel.setBounds(row1.removeFromLeft(labelWidth));
    releaseSlider.setBounds(row1.removeFromLeft(sliderWidth));

    bounds.removeFromTop(rowGap);

    // Row 2: Threshold, Freq Lo, Freq Hi
    auto row2 = bounds.removeFromTop(rowHeight);
    thresholdLabel.setBounds(row2.removeFromLeft(labelWidth));
    thresholdSlider.setBounds(row2.removeFromLeft(sliderWidth));
    row2.removeFromLeft(15);
    freqLowLabel.setBounds(row2.removeFromLeft(labelWidth));
    freqLowSlider.setBounds(row2.removeFromLeft(sliderWidth));
    row2.removeFromLeft(15);
    freqHighLabel.setBounds(row2.removeFromLeft(labelWidth));
    freqHighSlider.setBounds(row2.removeFromLeft(sliderWidth));

    bounds.removeFromTop(rowGap);

    // Row 3: Range only
    auto row3 = bounds.removeFromTop(rowHeight);
    floorLabel.setBounds(row3.removeFromLeft(labelWidth));
    floorSlider.setBounds(row3.removeFromLeft(sliderWidth));

    bounds.removeFromTop(12);

    // Bottom row: bypass, low latency, latency display, and model status
    auto bottomRow = bounds.removeFromTop(25);
    bypassButton.setBounds(bottomRow.removeFromLeft(80));
    bottomRow.removeFromLeft(8);
    lowLatencyButton.setBounds(bottomRow.removeFromLeft(95));
    bottomRow.removeFromLeft(8);
    latencyLabel.setBounds(bottomRow.removeFromLeft(70));
    modelStatusLabel.setBounds(bottomRow);

    // Mix slider is removed from main UI (user said they don't need it)
    mixSlider.setVisible(false);
    mixLabel.setVisible(false);

    // Log (if shown)
    if (showLog && bounds.getHeight() > 50)
    {
        bounds.removeFromTop(10);
        logTextBox.setBounds(bounds);
        logTextBox.setVisible(true);
    }
    else
    {
        logTextBox.setVisible(false);
    }
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

    // Update visualization
    if (rtaView)
        rtaView->updateFromQueue();

    if (gainReductionMeter)
    {
        float reductionDb = audioProcessor.getVisualizationData().averageGainReductionDb.load();
        gainReductionMeter->setReductionLevel(reductionDb);
        gainReductionMeter->repaint();
    }

    updateModelStatus();
    updateLatencyLabel();
}

void DeBleedAudioProcessorEditor::startTraining()
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

                // Sanitize the name for filesystem
                modelName = modelName.replaceCharacters(" /\\:*?\"<>|", "___________");

                startTrainingWithName(modelName);
            }
            delete nameDialog;
        }), true);
}

void DeBleedAudioProcessorEditor::startTrainingWithName(const juce::String& modelName)
{
    // Create output directory in user's app data
    juce::File outputDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                               .getChildFile("DeBleed")
                               .getChildFile("Models")
                               .getChildFile(modelName);

    // If name already exists, append timestamp
    if (outputDir.exists())
    {
        outputDir = outputDir.getSiblingFile(modelName + "_" +
                        juce::Time::getCurrentTime().formatted("%H%M%S"));
    }

    outputDir.createDirectory();

    // Clear log
    logTextBox.clear();
    showLog = true;
    addAndMakeVisible(logTextBox);
    resized();

    // Start training
    trainButton.setEnabled(false);
    trainButton.setButtonText("Training...");
    statusLabel.setText("Preparing training...", juce::dontSendNotification);
    progressValue = 0.0;

    bool started = audioProcessor.getTrainerProcess().startTraining(
        cleanAudioPath,
        noiseAudioPath,
        outputDir.getFullPathName(),
        50  // epochs
    );

    if (!started)
    {
        statusLabel.setText("Failed to start training: " +
                           audioProcessor.getTrainerProcess().getLastError(),
                           juce::dontSendNotification);
        trainButton.setEnabled(true);
        trainButton.setButtonText("Train Model");
    }
}

void DeBleedAudioProcessorEditor::loadModel()
{
    // Default to the DeBleed models directory
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
                    updateModelStatus();
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
        trainButton.setEnabled(cleanAudioPath.isNotEmpty() && noiseAudioPath.isNotEmpty());
        trainButton.setButtonText("Train Model");

        if (success)
        {
            statusLabel.setText("Training complete! Model loaded.", juce::dontSendNotification);
            progressValue = 1.0;

            // Load the new model
            audioProcessor.loadModel(modelPath);
            updateModelStatus();

            // Show success message
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
        modelStatusLabel.setColour(juce::Label::textColourId, juce::Colour(0xff5a9a6c));
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
