#pragma once

#include <JuceHeader.h>
#include <array>
#include <vector>

/**
 * BandMapper - Maps 129 FFT bins to 64 IIR filter band gains.
 *
 * Uses weighted interpolation based on frequency proximity to smoothly
 * convert the neural network's mask output (129 FFT bins) to the
 * IIR filter bank's gain controls (64 bands).
 */
class BandMapper
{
public:
    static constexpr int NUM_FFT_BINS = 129;
    static constexpr int NUM_IIR_BANDS = 64;

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
     * @param fftMask Input mask values
     * @param bandGains Output gains
     * @param strength Strength parameter (0.0 to 1.0+, where 1.0 = full effect)
     */
    void mapWithStrength(const float* fftMask,
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
