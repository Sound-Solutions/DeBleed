#pragma once

#include <JuceHeader.h>
#include "PluginProcessor.h"
#include "AudioDropZone.h"

/**
 * DeBleedAudioProcessorEditor - GUI for the DeBleed Neural Gate plugin.
 *
 * Layout:
 * +------------------------------------------+
 * |  DeBleed - Neural Gate                   |
 * +------------------------------------------+
 * |  +------------------+  +---------------+ |
 * |  | Target Vocals    |  | Stage Noise   | |
 * |  | (Drop Zone)      |  | (Drop Zone)   | |
 * |  +------------------+  +---------------+ |
 * +------------------------------------------+
 * |  [    Progress Bar                     ] |
 * |  [ Train Model ]                         |
 * +------------------------------------------+
 * |  Strength: [========]  Mix: [========]   |
 * |  [ ] Bypass              Model: Loaded   |
 * +------------------------------------------+
 */
class DeBleedAudioProcessorEditor : public juce::AudioProcessorEditor,
                                     public juce::Timer
{
public:
    explicit DeBleedAudioProcessorEditor(DeBleedAudioProcessor&);
    ~DeBleedAudioProcessorEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;
    void timerCallback() override;

private:
    void startTraining();
    void startTrainingWithName(const juce::String& modelName);
    void loadModel();
    void onTrainingProgress(int progress, const juce::String& status);
    void onTrainingComplete(bool success, const juce::String& modelPath, const juce::String& error);
    void updateModelStatus();
    void updateLatencyLabel();

    DeBleedAudioProcessor& audioProcessor;

    // Header
    juce::Label titleLabel;

    // Drop zones
    AudioDropZone cleanDropZone;
    AudioDropZone noiseDropZone;

    // Training controls
    juce::ProgressBar progressBar;
    juce::TextButton trainButton;
    juce::TextButton loadModelButton;
    juce::Label statusLabel;
    double progressValue = 0.0;

    // Parameter controls
    juce::Slider strengthSlider;
    juce::Label strengthLabel;
    juce::Slider mixSlider;
    juce::Label mixLabel;
    juce::ToggleButton bypassButton;
    juce::ToggleButton lowLatencyButton;
    juce::Label latencyLabel;

    // Model status
    juce::Label modelStatusLabel;

    // Parameter attachments
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> strengthAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> mixAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> bypassAttachment;
    std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> lowLatencyAttachment;

    // Training log
    juce::TextEditor logTextBox;
    bool showLog = false;

    // Stored paths
    juce::String cleanAudioPath;
    juce::String noiseAudioPath;

    // Look and feel
    juce::LookAndFeel_V4 lookAndFeel;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeBleedAudioProcessorEditor)
};
