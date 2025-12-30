#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "AudioDropZone.h"
#include "RTAVisualization.h"
#include "GainReductionMeter.h"
#include "DeBleedLookAndFeel.h"

/**
 * DeBleedAudioProcessorEditor - GUI for the DeBleed Neural Gate plugin.
 *
 * Two-tab layout with Kinetics-style knobs:
 * - Training tab: Drop zones, progress, train buttons
 * - Visualizing tab: Large RTA, knobs
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
    void updateLatencyLabel();
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
    juce::ToggleButton liveModeButton;  // Live/Train toggle

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

    // Visualization components
    std::unique_ptr<RTAVisualization> rtaView;
    std::unique_ptr<GainReductionMeter> gainReductionMeter;

    // Parameter knobs (rotary style)
    juce::Slider strengthSlider;
    juce::Label strengthLabel;
    juce::Slider attackSlider;
    juce::Label attackLabel;
    juce::Slider releaseSlider;
    juce::Label releaseLabel;
    juce::Slider thresholdSlider;
    juce::Label thresholdLabel;
    juce::Slider floorSlider;  // Range parameter
    juce::Label floorLabel;
    juce::Slider mixSlider;    // Hidden but kept for attachment
    juce::Label mixLabel;

    // Toggle buttons
    juce::ToggleButton lowLatencyButton;
    juce::Label latencyLabel;

    // Parameter attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> strengthAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lowLatencyAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> releaseAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> thresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> floorAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> liveModeAttachment;

    // Training log
    juce::TextEditor logTextBox;
    bool showLog = false;

    // Stored paths
    juce::String cleanAudioPath;
    juce::String noiseAudioPath;

    // Track last trained model for continue training
    juce::String lastTrainedModelDir;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessorEditor)
};
