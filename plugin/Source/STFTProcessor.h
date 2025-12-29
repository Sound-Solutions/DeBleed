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
    // Fixed output bins for neural network compatibility
    static constexpr int N_FREQ_BINS = 129;  // Always 129 for neural net

    // Configurable FFT sizes
    enum class Mode { Quality, LowLatency };

    // Quality mode: FFT=256, ~5.3ms latency
    static constexpr int N_FFT_QUALITY = 256;
    static constexpr int HOP_QUALITY = 128;

    // Low latency mode: FFT=128, ~2.7ms latency
    static constexpr int N_FFT_LOW_LATENCY = 128;
    static constexpr int HOP_LOW_LATENCY = 64;

    STFTProcessor();
    ~STFTProcessor() = default;

    /** Set the processing mode (must call before prepare) */
    void setMode(Mode newMode) { mode = newMode; }
    Mode getMode() const { return mode; }

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

    /**
     * Read pending output samples without processing new frames.
     * Call this when numFrames is 0 to get the tail of the overlap-add.
     */
    void readOutput(float* output, int numSamples);

    /** Get the latency in samples introduced by STFT processing. */
    // One FFT window offset for overlap-add to complete before reading
    int getLatencySamples() const { return currentFFTSize; }

    /** Get current FFT size */
    int getFFTSize() const { return currentFFTSize; }

    /** Get current hop length */
    int getHopLength() const { return currentHopLength; }

private:
    // Mode
    Mode mode = Mode::Quality;
    int currentFFTSize = N_FFT_QUALITY;
    int currentHopLength = HOP_QUALITY;
    int currentFreqBins = N_FFT_QUALITY / 2 + 1;  // Actual bins from FFT

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

    // Overlap-add output buffer (circular, persistent between blocks)
    std::vector<float> outputBuffer;
    int outputReadPos = 0;
    int outputWritePos = 0;
    float colaSum = 1.5f;  // COLA normalization factor

    // Temporary FFT buffers
    std::vector<std::complex<float>> fftBuffer;
    std::vector<float> fftWorkBuffer;

    // Sample rate
    double currentSampleRate = 48000.0;

    // Helper methods
    void computeSTFTFrame(const float* frameData);
    void computeISTFTFrame(const float* magnitude, const float* phase, float* output);
    void createWindow();
    void interpolateBins(const float* input, int inputBins, float* output, int outputBins);
    void decimateBins(const float* input, int inputBins, float* output, int outputBins);

    // Interpolation buffer for low-latency mode
    std::vector<float> interpMagBuffer;
    std::vector<float> interpPhaseBuffer;
    std::vector<float> decimatedMaskBuffer;

    // DEBUG: Store windowed frames for testing overlap-add without FFT
    std::vector<float> windowedFrameBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(STFTProcessor)
};
