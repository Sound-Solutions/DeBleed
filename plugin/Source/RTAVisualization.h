#pragma once

#include <JuceHeader.h>
#include <array>

// Forward declaration
class DeBleedAudioProcessor;

/**
 * RTAVisualization - Real-Time Analyzer style display with mask reduction overlay.
 *
 * Layout:
 * - Top 60%: Input spectrum (RTA curve with cyan fill)
 * - 0dB reference line
 * - Bottom 40%: Reduction curve (red/orange showing attenuation)
 *
 * Uses logarithmic frequency scale (20Hz - 20kHz)
 */
class RTAVisualization : public juce::Component
{
public:
    static constexpr int N_FREQ_BINS = 129;
    static constexpr int NUM_IIR_BANDS = 64;
    static constexpr float MIN_FREQ = 20.0f;
    static constexpr float MAX_FREQ = 20000.0f;
    static constexpr float MIN_DB = -60.0f;
    static constexpr float MAX_DB = 12.0f;
    static constexpr float MIN_REDUCTION_DB = -40.0f;
    static constexpr float BANDPASS_Q = 2.0f;  // Must match IIRFilterBank

    explicit RTAVisualization(DeBleedAudioProcessor& processor);
    ~RTAVisualization() override = default;

    void paint(juce::Graphics& g) override;
    void resized() override;

    // Called from timer to update display data
    void updateFromQueue();

private:
    DeBleedAudioProcessor& audioProcessor;

    // Display buffers (smoothed for display)
    std::array<float, N_FREQ_BINS> displayMagnitudeDb;  // Input spectrum in dB
    std::array<float, N_FREQ_BINS> displayMask;         // Mask values 0-1

    // Smoothing parameters
    static constexpr float minDecayRate = 1.5f;   // dB per frame for bass
    static constexpr float maxDecayRate = 5.0f;   // dB per frame for treble
    static constexpr float maskSmoothCoeff = 0.2f;

    // Frequency mapping
    float binToFreq(int bin) const;
    float freqToNormX(float freq) const;
    float pixelToFreq(float x, float width) const;
    int freqToBin(float freq) const;
    float getInterpolatedMagnitude(float binIndex) const;
    float getInterpolatedMask(float binIndex) const;

    // Drawing functions
    void drawBackground(juce::Graphics& g);
    void drawGrid(juce::Graphics& g);
    void drawSpectrumCurve(juce::Graphics& g);
    void drawReductionCurve(juce::Graphics& g);
    void drawDividerLine(juce::Graphics& g);

    // IIR frequency response calculation
    float getBandpassMagnitude(float freq, float centerFreq, float Q) const;
    float getCombinedGainAtFreq(float freq) const;

    // Layout helpers
    float getSpectrumHeight() const { return getHeight() * 0.6f; }
    float getDividerY() const { return getSpectrumHeight(); }
    float getReductionHeight() const { return getHeight() * 0.4f; }

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(RTAVisualization)
};
