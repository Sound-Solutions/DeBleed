#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>

/**
 * BandMapper - Maps dual-stream mask (257 bins) to 64 IIR filter band gains.
 *
 * Dual-stream architecture:
 * - Stream A (bins 0-128): 129 bins, 187Hz resolution, for highs (~200Hz+)
 * - Stream B (bins 129-256): 128 bins, 23Hz resolution, for bass (20-200Hz)
 *
 * Low-frequency IIR bands use Stream B's high-resolution bass mask.
 * High-frequency IIR bands use Stream A's full-spectrum mask.
 */
class BandMapper
{
public:
    // Stream A: Full spectrum (187Hz resolution)
    static constexpr int NUM_STREAM_A_BINS = 129;
    // Stream B: Bass (23Hz resolution)
    static constexpr int NUM_STREAM_B_BINS = 128;
    // Total mask bins
    static constexpr int NUM_TOTAL_BINS = 257;

    // Legacy alias
    static constexpr int NUM_FFT_BINS = NUM_STREAM_A_BINS;
    static constexpr int NUM_IIR_BANDS = 64;

    // Crossover frequency: below this, use Stream B; above, use Stream A
    static constexpr float BASS_CROSSOVER_HZ = 200.0f;

    BandMapper();
    ~BandMapper() = default;

    /**
     * Initialize the mapper for the given sample rate.
     * Calculates the mapping weights between FFT bins and IIR bands.
     */
    void prepare(double sampleRate, const std::array<float, NUM_IIR_BANDS>& iirCenterFreqs);

    /**
     * Map FFT mask values to IIR band gains.
     * @param fftMask Input mask values from neural network (129 bins)
     * @param bandGains Output gains for IIR filter bank (64 bands)
     */
    void map(const float* fftMask, std::array<float, NUM_IIR_BANDS>& bandGains);

    /**
     * Map with strength scaling applied.
     * @param fftMask Input mask values (legacy 129-bin mode)
     * @param bandGains Output gains
     * @param strength Strength parameter (0.0 to 1.0+, where 1.0 = full effect)
     */
    void mapWithStrength(const float* fftMask,
                         std::array<float, NUM_IIR_BANDS>& bandGains,
                         float strength);

    /**
     * Map dual-stream mask (257 bins) to IIR band gains.
     * Uses Stream B (23Hz resolution) for bass bands, Stream A for highs.
     * @param dualMask Input mask values (257 bins: [Stream A 129 | Stream B 128])
     * @param bandGains Output gains for IIR filter bank (64 bands)
     */
    void mapDualStream(const float* dualMask, std::array<float, NUM_IIR_BANDS>& bandGains);

    /**
     * Map dual-stream mask with strength scaling.
     * @param dualMask Input mask values (257 bins)
     * @param bandGains Output gains
     * @param strength Strength parameter (0.0 to 2.0, where 1.0 = full effect)
     */
    void mapDualStreamWithStrength(const float* dualMask,
                                    std::array<float, NUM_IIR_BANDS>& bandGains,
                                    float strength);

    /**
     * Deprecated - frequency range filtering has been removed.
     * All bands now respond directly to the neural mask output.
     */
    void setFrequencyRange(float lowHz, float highHz);

private:
    // Precomputed mapping weights
    struct BandMapping
    {
        std::vector<int> binIndices;      // Which FFT bins contribute
        std::vector<float> binWeights;    // Weight for each bin
        float centerFreq = 0.0f;          // Band center frequency
    };

    std::array<BandMapping, NUM_IIR_BANDS> mappings;

    // FFT bin frequencies (depends on sample rate and FFT size)
    std::array<float, NUM_FFT_BINS> binFrequencies;

    // IIR band center frequencies (copied from filter bank)
    std::array<float, NUM_IIR_BANDS> bandCenterFreqs;

    double sampleRate = 48000.0;
};
