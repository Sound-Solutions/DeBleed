#include "PluginEditor.h"

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

    // Model status
    modelStatusLabel.setFont(juce::Font(11.0f));
    modelStatusLabel.setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    modelStatusLabel.setJustificationType(juce::Justification::centredRight);
    addAndMakeVisible(modelStatusLabel);

    // Parameter attachments
    strengthAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_STRENGTH, strengthSlider);
    mixAttachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_MIX, mixSlider);
    bypassAttachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        audioProcessor.getParameters(), DeBleedAudioProcessor::PARAM_BYPASS, bypassButton);

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

    // Update model status
    updateModelStatus();

    // Start timer for UI updates
    startTimer(100);

    // Set size
    setSize(500, 400);
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
}

void DeBleedAudioProcessorEditor::resized()
{
    auto bounds = getLocalBounds().reduced(15);

    // Title
    titleLabel.setBounds(bounds.removeFromTop(40));

    bounds.removeFromTop(15);

    // Drop zones (side by side)
    auto dropZoneArea = bounds.removeFromTop(120);
    int dropZoneWidth = (dropZoneArea.getWidth() - 15) / 2;

    cleanDropZone.setBounds(dropZoneArea.removeFromLeft(dropZoneWidth));
    dropZoneArea.removeFromLeft(15);  // Gap
    noiseDropZone.setBounds(dropZoneArea);

    bounds.removeFromTop(15);

    // Progress and training
    progressBar.setBounds(bounds.removeFromTop(20));
    bounds.removeFromTop(10);

    auto trainRow = bounds.removeFromTop(30);
    trainButton.setBounds(trainRow.removeFromLeft(120));
    trainRow.removeFromLeft(15);
    statusLabel.setBounds(trainRow);

    bounds.removeFromTop(20);

    // Parameter controls
    auto controlArea = bounds.removeFromTop(80);

    // Strength slider
    auto strengthRow = controlArea.removeFromTop(30);
    strengthRow.removeFromLeft(70);  // Label space
    strengthSlider.setBounds(strengthRow.removeFromLeft(200));

    controlArea.removeFromTop(10);

    // Mix slider
    auto mixRow = controlArea.removeFromTop(30);
    mixRow.removeFromLeft(70);  // Label space
    mixSlider.setBounds(mixRow.removeFromLeft(200));

    bounds.removeFromTop(10);

    // Bottom row: bypass and model status
    auto bottomRow = bounds.removeFromTop(25);
    bypassButton.setBounds(bottomRow.removeFromLeft(100));
    modelStatusLabel.setBounds(bottomRow);

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

    updateModelStatus();
}

void DeBleedAudioProcessorEditor::startTraining()
{
    if (cleanAudioPath.isEmpty() || noiseAudioPath.isEmpty())
        return;

    // Create output directory in user's app data
    juce::File outputDir = juce::File::getSpecialLocation(juce::File::userApplicationDataDirectory)
                               .getChildFile("DeBleed")
                               .getChildFile("Models")
                               .getChildFile(juce::Time::getCurrentTime().formatted("%Y%m%d_%H%M%S"));

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
