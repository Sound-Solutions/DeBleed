#pragma once

#include <JuceHeader.h>
#include <complex>
#include <vector>

/**
 * STFTProcessor - Short-Time Fourier Transform processor for real-time audio.
 *
 * Handles overlap-add STFT/iSTFT with low latency for the neural gate.
 * Uses JUCE's FFT for efficient computation.
 *
 * Parameters match the Python trainer:
 * - N_FFT: 256 (5.33ms at 48kHz)
 * - Hop: 128 (2.67ms)
 * - Window: Hann
 */
class STFTProcessor
{
public:
    static constexpr int N_FFT = 256;
    static constexpr int HOP_LENGTH = 128;
    static constexpr int WIN_LENGTH = 256;
    static constexpr int N_FREQ_BINS = N_FFT / 2 + 1;  // 129

    STFTProcessor();
    ~STFTProcessor() = default;

    /** Prepare the processor for a given sample rate and block size. */
    void prepare(double sampleRate, int maxBlockSize);

    /** Reset internal state (call when playback stops/starts). */
    void reset();

    /**
     * Process a block of audio samples.
     * Returns the number of complete STFT frames available.
     */
    int processBlock(const float* input, int numSamples);

    /**
     * Get the current magnitude spectrogram for neural network input.
     * Returns pointer to magnitude data [N_FREQ_BINS x numFrames].
     */
    const float* getMagnitudeData() const { return magnitudeBuffer.data(); }

    /**
     * Get the current phase data for reconstruction.
     * Returns pointer to phase data [N_FREQ_BINS x numFrames].
     */
    const float* getPhaseData() const { return phaseBuffer.data(); }

    /** Get the number of complete frames in the current buffer. */
    int getNumFrames() const { return numFramesReady; }

    /**
     * Apply a mask to the magnitude and reconstruct audio.
     * @param mask Pointer to mask data [N_FREQ_BINS x numFrames], values in [0, 1]
     * @param output Output buffer for reconstructed audio
     * @param numSamples Number of samples to write
     */
    void applyMaskAndReconstruct(const float* mask, float* output, int numSamples);

    /** Get the latency in samples introduced by STFT processing. */
    int getLatencySamples() const { return N_FFT; }

private:
    // FFT
    std::unique_ptr<juce::dsp::FFT> fft;
    int fftOrder;

    // Windows
    std::vector<float> analysisWindow;
    std::vector<float> synthesisWindow;

    // Input buffering
    std::vector<float> inputBuffer;
    int inputWritePos = 0;

    // STFT output buffers
    std::vector<float> magnitudeBuffer;
    std::vector<float> phaseBuffer;
    int numFramesReady = 0;
    int maxFrames = 0;

    // Overlap-add buffers for reconstruction
    std::vector<float> overlapAddBuffer;
    int overlapAddReadPos = 0;

    // Temporary FFT buffers
    std::vector<std::complex<float>> fftBuffer;
    std::vector<float> fftWorkBuffer;

    // Sample rate
    double currentSampleRate = 48000.0;

    // Helper methods
    void computeSTFTFrame(const float* frameData);
    void computeISTFTFrame(const float* magnitude, const float* phase, float* output);
    void createWindow();

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(STFTProcessor)
};
