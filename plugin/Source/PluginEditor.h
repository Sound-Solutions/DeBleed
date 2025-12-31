#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "AudioDropZone.h"
#include "RTAVisualization.h"
#include "GainReductionMeter.h"
#include "ConfidenceMeter.h"
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
    std::unique_ptr<ConfidenceMeter> confidenceMeter;

    // Parameter knobs (rotary style)
    // Row 1: Hunter controls (compressor-style, surgical)
    juce::Slider hunterAttackSlider;
    juce::Label hunterAttackLabel;
    juce::Slider hunterReleaseSlider;
    juce::Label hunterReleaseLabel;
    juce::Slider hunterHoldSlider;
    juce::Label hunterHoldLabel;
    juce::Slider hunterRangeSlider;
    juce::Label hunterRangeLabel;

    // Row 1 continued: Hunter frequency bounds
    juce::Slider hpfBoundSlider;
    juce::Label hpfBoundLabel;
    juce::Slider lpfBoundSlider;
    juce::Label lpfBoundLabel;
    juce::Slider tightnessSlider;
    juce::Label tightnessLabel;

    // Row 2: Expander controls (gate-style)
    juce::Slider expanderAttackSlider;
    juce::Label expanderAttackLabel;
    juce::Slider expanderReleaseSlider;
    juce::Label expanderReleaseLabel;
    juce::Slider expanderHoldSlider;
    juce::Label expanderHoldLabel;
    juce::Slider expanderRangeSlider;
    juce::Label expanderRangeLabel;
    juce::Slider expanderThresholdSlider;
    juce::Label expanderThresholdLabel;
    juce::ToggleButton lrEnabledButton;

    // Hidden but kept for attachment
    juce::Slider mixSlider;
    juce::Label mixLabel;

    // Toggle buttons
    juce::ToggleButton lowLatencyButton;
    juce::Label latencyLabel;

    // Parameter attachments - General
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lowLatencyAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> liveModeAttachment;

    // Hunter parameter attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hunterAttackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hunterReleaseAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hunterHoldAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hunterRangeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> hpfBoundAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> lpfBoundAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> tightnessAttachment;

    // Expander parameter attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> expanderAttackAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> expanderReleaseAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> expanderHoldAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> expanderRangeAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> expanderThresholdAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lrEnabledAttachment;

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
