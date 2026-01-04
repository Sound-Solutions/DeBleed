#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "AudioDropZone.h"
#include "DeBleedLookAndFeel.h"
#include "EQCurveDisplay.h"
#include "ControlPanel.h"

/**
 * DeBleedAudioProcessorEditor - GUI for the DeBleed Neural 5045 plugin.
 *
 * Kinetics-style layout:
 * - Header: Title, tab buttons, power button
 * - Main area: EQ curve visualization (interactive) or Training panel
 * - Bottom: Control panel with rotary knobs
 */
class DeBleedAudioProcessorEditor : public juce::AudioProcessorEditor,
                                     public juce::Timer
{
public:
    enum class Tab { Training, Visualizing };

    explicit DeBleedAudioProcessorEditor(DeBleedAudioProcessor&);
    ~DeBleedAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    void setActiveTab(Tab tab);
    void layoutTrainingTab(juce::Rectangle<int> bounds);
    void layoutVisualizingTab(juce::Rectangle<int> bounds);

    void startNewTraining();
    void continueTraining();
    void startTrainingWithName(const juce::String& modelName, bool isContinuation);
    void loadModel();
    void onTrainingProgress(int progress, const juce::String& status);
    void onTrainingComplete(bool success, const juce::String& modelPath, const juce::String& error);
    void updateModelStatus();
    void updateContinueButtonState();

    DeBleedAudioProcessor& audioProcessor;
    DeBleedLookAndFeel customLookAndFeel;

    // Current tab
    Tab currentTab = Tab::Visualizing;  // Default to visualizing

    // Header components
    juce::Label titleLabel;
    juce::TextButton trainingTabButton;
    juce::TextButton visualizingTabButton;
    juce::ToggleButton bypassButton;
    juce::ToggleButton liveModeButton;

    // Training tab components
    AudioDropZone cleanDropZone;
    AudioDropZone noiseDropZone;
    juce::ProgressBar progressBar;
    juce::TextButton trainNewButton;
    juce::TextButton continueTrainingButton;
    juce::TextButton loadModelButton;
    juce::Label statusLabel;
    juce::Label modelStatusLabel;
    double progressValue = 0.0;

    // Training log
    juce::TextEditor logTextBox;
    bool showLog = false;

    // Stored paths
    juce::String cleanAudioPath;
    juce::String noiseAudioPath;

    // Track last trained model for continue training
    juce::String lastTrainedModelDir;

    // Visualizing tab components
    EQCurveDisplay eqCurveDisplay_;

    // Bottom control panel (always visible)
    ControlPanel controlPanel_;

    // Parameter attachments for header controls
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> liveModeAttachment;

    // Layout constants
    static constexpr int headerHeight = 50;
    static constexpr int controlPanelHeight = 150;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessorEditor)
};
