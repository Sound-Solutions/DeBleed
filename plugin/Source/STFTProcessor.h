#pragma once

#include <JuceHeader.h>
#include <complex>
#include <vector>

/**
 * STFTProcessor - Short-Time Fourier Transform processor for real-time audio.
 *
 * Supports dual-stream STFT analysis for enhanced frequency resolution:
 * - Stream A (Fast): 256-point FFT, ~187Hz resolution, good for transients
 * - Stream B (Slow): 2048-point FFT, ~23Hz resolution, good for bass precision
 *
 * Both streams use the same hop size for frame alignment.
 * The neural network receives concatenated features: [Stream A (129 bins) | Stream B bass (128 bins)]
 */
class STFTProcessor
{
public:
    // ==========================================================================
    // Dual-Stream Configuration (matches Python trainer)
    // ==========================================================================

    // Stream A: Fast/Temporal (transients, high-frequency detail)
    static constexpr int N_FFT_A = 256;           // ~5.33ms @ 48kHz
    static constexpr int N_FREQ_BINS_A = 129;     // N_FFT_A / 2 + 1

    // Stream B: Slow/Spectral (bass precision)
    static constexpr int N_FFT_B = 2048;          // ~42.67ms @ 48kHz
    static constexpr int N_FREQ_BINS_B = 1025;    // N_FFT_B / 2 + 1
    static constexpr int STREAM_B_BINS_USED = 128; // Lower bins only (0 to ~1.5kHz)

    // Common hop size for frame alignment
    static constexpr int HOP_LENGTH = 128;        // ~2.67ms @ 48kHz

    // Total input features for neural network
    static constexpr int N_TOTAL_FEATURES = N_FREQ_BINS_A + STREAM_B_BINS_USED;  // 257

    // Legacy aliases
    static constexpr int N_FREQ_BINS = N_FREQ_BINS_A;  // Output mask is still 129 bins

    // Configurable FFT sizes (for legacy single-stream mode)
    enum class Mode { Quality, LowLatency };

    // Quality mode: FFT=256, ~5.3ms latency
    static constexpr int N_FFT_QUALITY = N_FFT_A;
    static constexpr int HOP_QUALITY = HOP_LENGTH;

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
     * Get Stream A magnitude spectrogram (legacy method).
     * Returns pointer to magnitude data [N_FREQ_BINS_A x numFrames].
     */
    const float* getMagnitudeData() const { return magnitudeBuffer.data(); }

    /**
     * Get the current phase data for reconstruction.
     * Returns pointer to phase data [N_FREQ_BINS x numFrames].
     */
    const float* getPhaseData() const { return phaseBuffer.data(); }

    /** Get the number of complete frames in the current buffer. */
    int getNumFrames() const { return numFramesReady; }

    // ==========================================================================
    // Dual-Stream Analysis Methods
    // ==========================================================================

    /**
     * Enable dual-stream analysis mode.
     * When enabled, both Stream A (256 FFT) and Stream B (2048 FFT) are computed.
     */
    void setDualStreamEnabled(bool enabled) { dualStreamEnabled = enabled; }
    bool isDualStreamEnabled() const { return dualStreamEnabled; }

    /**
     * Get Stream A magnitude (129 bins, ~187Hz resolution).
     */
    const float* getStreamAMagnitude() const { return magnitudeBuffer.data(); }

    /**
     * Get Stream B low-frequency magnitude (128 bins, ~23Hz resolution).
     * Only valid if dual-stream mode is enabled.
     */
    const float* getStreamBLowMagnitude() const { return streamBLowMagBuffer.data(); }

    /**
     * Get concatenated dual-stream features for neural network input.
     * Returns pointer to data [N_TOTAL_FEATURES x numFrames] = [257 x numFrames]
     * Layout: [Stream A (129) | Stream B bass (128)]
     */
    const float* getDualStreamFeatures() const { return dualStreamFeaturesBuffer.data(); }

    /**
     * Get the total number of input features for dual-stream mode.
     */
    static constexpr int getTotalFeatures() { return N_TOTAL_FEATURES; }

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

    // ==========================================================================
    // Stream A (Fast STFT - 256 point)
    // ==========================================================================
    std::unique_ptr<juce::dsp::FFT> fft;  // Stream A FFT
    int fftOrder;

    // Windows for Stream A
    std::vector<float> analysisWindow;
    std::vector<float> synthesisWindow;

    // Input buffering for Stream A
    std::vector<float> inputBuffer;
    int inputWritePos = 0;

    // STFT output buffers for Stream A
    std::vector<float> magnitudeBuffer;
    std::vector<float> phaseBuffer;
    int numFramesReady = 0;
    int maxFrames = 0;

    // ==========================================================================
    // Stream B (Slow STFT - 2048 point) for bass precision
    // ==========================================================================
    bool dualStreamEnabled = true;  // Enable by default for new models
    std::unique_ptr<juce::dsp::FFT> fftB;  // Stream B FFT (larger)
    int fftOrderB;

    // Window for Stream B
    std::vector<float> analysisWindowB;

    // Input buffering for Stream B (needs larger buffer)
    std::vector<float> inputBufferB;
    int inputWritePosB = 0;

    // Stream B output buffers (only low-frequency bins)
    std::vector<float> streamBLowMagBuffer;  // [STREAM_B_BINS_USED x maxFrames]

    // Concatenated dual-stream features [N_TOTAL_FEATURES x maxFrames]
    std::vector<float> dualStreamFeaturesBuffer;

    // FFT work buffer for Stream B
    std::vector<float> fftWorkBufferB;

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

    // Helper methods for Stream A
    void computeSTFTFrame(const float* frameData);
    void computeISTFTFrame(const float* magnitude, const float* phase, float* output);
    void createWindow();
    void interpolateBins(const float* input, int inputBins, float* output, int outputBins);
    void decimateBins(const float* input, int inputBins, float* output, int outputBins);

    // Helper methods for Stream B (dual-stream)
    void createWindowB();
    void computeStreamBFrame(const float* frameData);
    void concatenateDualStreamFeatures();

    // Interpolation buffer for low-latency mode
    std::vector<float> interpMagBuffer;
    std::vector<float> interpPhaseBuffer;
    std::vector<float> decimatedMaskBuffer;

    // DEBUG: Store windowed frames for testing overlap-add without FFT
    std::vector<float> windowedFrameBuffer;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(STFTProcessor)
};
