#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "AudioDropZone.h"
#include "DeBleedLookAndFeel.h"
#include "EQCurveDisplay.h"

/**
 * DeBleedAudioProcessorEditor - GUI for the DeBleed Neural 5045 plugin.
 *
 * Two-tab layout:
 * - Training tab: Drop zones, progress, train buttons
 * - Visualizing tab: Real-time EQ curve from neural network
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
    Tab currentTab = Tab::Training;

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

    // Hidden mix slider (must be declared BEFORE attachment for correct destruction order)
    juce::Slider mixSlider;

    // Parameter attachments (destroyed before the components they attach to)
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> liveModeAttachment;

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

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessorEditor)
};
