#pragma once

#include <JuceHeader.h>
#include <array>
#include <atomic>

/**
 * SpectralVAD - Voice Activity Detection with learned spectral weighting
 *
 * Uses a bank of bandpass filters to compute frequency-specific energy,
 * then applies learned weights to distinguish vocal timbre from bleed.
 *
 * The weights are trained to be high for vocal-dominant frequencies
 * (formants ~800Hz-3kHz) and low for bleed-dominant frequencies
 * (cymbals, snare bleed, etc).
 *
 * Zero latency - uses IIR bandpass filters (no FFT).
 */
class SpectralVAD
{
public:
    // 8 bands covering vocal and bleed frequency ranges
    static constexpr int NUM_BANDS = 8;

    // Default center frequencies (Hz) - log-spaced from 200Hz to 8kHz
    static constexpr std::array<float, NUM_BANDS> DEFAULT_FREQUENCIES = {
        200.0f, 400.0f, 800.0f, 1600.0f, 2500.0f, 4000.0f, 6000.0f, 8000.0f
    };

    // Default Q values (bandwidth control)
    static constexpr float DEFAULT_Q = 1.4f;  // ~1 octave bandwidth

    SpectralVAD();
    ~SpectralVAD() = default;

    /**
     * Prepare for playback.
     */
    void prepare(double sampleRate);

    /**
     * Reset all filter and envelope states.
     */
    void reset();

    /**
     * Process a single sample and return VAD confidence.
     * @param sample Input audio sample
     * @return Confidence 0-1 (1 = vocal present, 0 = silence/bleed only)
     */
    float processSample(float sample);

    /**
     * Process a block of audio and return per-sample confidence.
     */
    void processBlock(const float* input, float* confidence, int numSamples);

    /**
     * Set the learned weights for each frequency band.
     * Higher weight = more important for vocal detection.
     * Weights should sum to 1.0 for normalized output.
     */
    void setBandWeights(const std::array<float, NUM_BANDS>& weights);

    /**
     * Set center frequencies for the bandpass filters.
     */
    void setBandFrequencies(const std::array<float, NUM_BANDS>& frequencies);

    /**
     * Set threshold and knee for confidence calculation.
     */
    void setThresholdDb(float thresholdDb);
    void setKneeDb(float kneeDb);

    /**
     * Set attack/release times for envelope following.
     */
    void setAttackMs(float attackMs);
    void setReleaseMs(float releaseMs);

    /**
     * Get current confidence (thread-safe, for UI).
     */
    float getConfidence() const { return currentConfidence_.load(); }

    /**
     * Get current band energies (for visualization).
     */
    std::array<float, NUM_BANDS> getBandEnergies() const;

    /**
     * Get current weights (for visualization).
     */
    std::array<float, NUM_BANDS> getBandWeights() const { return bandWeights_; }

private:
    // Biquad bandpass filter coefficients and state
    struct BandpassFilter
    {
        // Coefficients
        float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f;
        float a1 = 0.0f, a2 = 0.0f;

        // State
        float x1 = 0.0f, x2 = 0.0f;
        float y1 = 0.0f, y2 = 0.0f;

        // RMS envelope for this band
        float envelope = 0.0f;

        void reset()
        {
            x1 = x2 = y1 = y2 = 0.0f;
            envelope = 0.0f;
        }

        float process(float input)
        {
            float output = b0 * input + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = input;
            y2 = y1;
            y1 = output;
            return output;
        }
    };

    void computeBandpassCoeffs(BandpassFilter& filter, float freqHz, float q);
    void updateCoefficients();
    float computePitchConfidence();  // Autocorrelation-based pitch detection

    double sampleRate_ = 48000.0;

    // Bandpass filters for each frequency band
    std::array<BandpassFilter, NUM_BANDS> filters_;

    // Center frequencies for each band
    std::array<float, NUM_BANDS> bandFrequencies_;

    // Learned weights for each band (vocal vs bleed discrimination)
    std::array<float, NUM_BANDS> bandWeights_;

    // Q factor for bandpass filters
    float bandQ_ = DEFAULT_Q;

    // Envelope follower coefficients
    float attackCoeff_ = 0.0f;
    float releaseCoeff_ = 0.0f;

    // Confidence smoothing
    float smoothedConfidence_ = 0.0f;
    float confAttackCoeff_ = 0.0f;
    float confReleaseCoeff_ = 0.0f;

    // Parameters
    std::atomic<float> thresholdDb_{-40.0f};
    std::atomic<float> kneeDb_{15.0f};
    std::atomic<float> attackMs_{2.0f};    // Fast attack
    std::atomic<float> releaseMs_{30.0f};

    // Pitch detection buffer for autocorrelation
    static constexpr int PITCH_BUFFER_SIZE = 1024;  // ~21ms at 48kHz, covers 50Hz-500Hz range
    std::array<float, PITCH_BUFFER_SIZE> pitchBuffer_{};
    int pitchBufferIndex_ = 0;
    float pitchConfidence_ = 0.0f;  // 0 = unpitched (drums), 1 = clear pitch (vocals)

    // Formant detection - ratio of formant band energy to total
    float formantRatio_ = 0.0f;

    // Pitch range for vocals (fundamental frequency)
    static constexpr float MIN_VOCAL_PITCH_HZ = 80.0f;   // Low male voice
    static constexpr float MAX_VOCAL_PITCH_HZ = 400.0f;  // High female voice

    // Current confidence (for UI)
    std::atomic<float> currentConfidence_{0.0f};

    // Band energies for visualization
    std::array<std::atomic<float>, NUM_BANDS> bandEnergies_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(SpectralVAD)
};
