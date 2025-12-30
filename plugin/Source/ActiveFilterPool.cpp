#include "ActiveFilterPool.h"

ActiveFilterPool::ActiveFilterPool()
{
    // Initialize all hunters to neutral state
    for (auto& h : hunters)
    {
        h.coeffs = new juce::dsp::IIR::Coefficients<float>();
        h.smoothGain.setCurrentAndTargetValue(1.0f);
        h.smoothFreq.setCurrentAndTargetValue(1000.0f);
        h.currentFreq = 1000.0f;
        h.currentGain = 1.0f;
        h.currentQ = 4.0f;
        h.active = false;
    }

    // Initialize bin->freq lookups (will be recalculated in prepare())
    binToFreqA.fill(1000.0f);
    binToFreqB.fill(500.0f);
}

void ActiveFilterPool::prepare(double newSampleRate, int samplesPerBlock)
{
    sampleRate = newSampleRate;

    juce::dsp::ProcessSpec spec;
    spec.sampleRate = sampleRate;
    spec.maximumBlockSize = static_cast<juce::uint32>(samplesPerBlock);
    spec.numChannels = 1;  // We process channels separately

    float nyquist = static_cast<float>(sampleRate) * 0.5f;

    // Build bin -> frequency lookup tables
    // IMPORTANT: Always use 48kHz for bin mapping because the neural mask
    // is ALWAYS generated at 48kHz (sidechain analyzer resamples to 48kHz)
    static constexpr float MASK_SAMPLE_RATE = 48000.0f;

    // Stream A: 256-point FFT at 48kHz -> 187.5 Hz/bin
    // bin[i] = i * 48000 / 256
    for (int i = 0; i < STREAM_A_BINS; ++i)
    {
        binToFreqA[i] = static_cast<float>(i) * MASK_SAMPLE_RATE / 256.0f;
        binToFreqA[i] = juce::jlimit(MIN_FREQ, nyquist * 0.95f, binToFreqA[i]);
    }

    // Stream B: 2048-point FFT at 48kHz -> 23.4 Hz/bin (8x better resolution!)
    // bin[i] = i * 48000 / 2048
    // Only covers low frequencies (0 to ~1.5kHz)
    for (int i = 0; i < STREAM_B_BINS; ++i)
    {
        binToFreqB[i] = static_cast<float>(i) * MASK_SAMPLE_RATE / 2048.0f;
        binToFreqB[i] = juce::jlimit(MIN_FREQ, nyquist * 0.95f, binToFreqB[i]);
    }

    // Initialize all hunter filters
    for (auto& h : hunters)
    {
        // Set up smoothers
        h.smoothGain.reset(sampleRate, SMOOTHING_TIME);
        h.smoothFreq.reset(sampleRate, SMOOTHING_TIME);
        h.smoothGain.setCurrentAndTargetValue(1.0f);
        h.smoothFreq.setCurrentAndTargetValue(1000.0f);

        // Initialize filter coefficients to unity peak at 1kHz
        *h.coeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
            sampleRate, 1000.0f, 4.0f, 1.0f);

        // Prepare stereo filters
        for (int ch = 0; ch < 2; ++ch)
        {
            h.filter[ch].coefficients = h.coeffs;
            h.filter[ch].prepare(spec);
        }

        h.currentFreq = 1000.0f;
        h.currentGain = 1.0f;
        h.currentQ = 4.0f;
        h.active = false;
    }
}

void ActiveFilterPool::reset()
{
    for (auto& h : hunters)
    {
        h.filter[0].reset();
        h.filter[1].reset();
        h.smoothGain.setCurrentAndTargetValue(1.0f);
        h.smoothFreq.setCurrentAndTargetValue(1000.0f);
        h.currentGain = 1.0f;
        h.currentFreq = 1000.0f;
        h.active = false;
    }
}

void ActiveFilterPool::updateTargets(const float* mask)
{
    if (mask == nullptr)
    {
        // No mask - set all hunters to unity
        for (auto& h : hunters)
        {
            h.smoothGain.setTargetValue(1.0f);
            h.active = false;
        }
        return;
    }

    // Find local minima (valleys) in the mask
    struct Candidate
    {
        int bin;
        float gain;
        float freq;
    };
    std::vector<Candidate> valleys;
    valleys.reserve(64);

    // === STREAM B: HIGH-RESOLUTION LOW FREQUENCIES ===
    // Stream B bins are at mask indices 129-256 (offset by STREAM_A_BINS)
    // These have 8x better resolution (23.4 Hz/bin vs 187.5 Hz/bin)
    // Stream B covers 0 to ~1.5kHz
    const float* streamB = mask + STREAM_A_BINS;  // Point to Stream B portion

    // For low frequencies, use Stream B with relaxed valley detection
    // Stream B bin 1 = 23.4 Hz, bin 64 = 1500 Hz
    for (int i = 1; i < STREAM_B_BINS; ++i)
    {
        float val = streamB[i];
        float freq = binToFreqB[i];

        // Only process bins in the low frequency range (up to ~1.5kHz)
        if (freq > 1500.0f)
            break;

        if (val < VALLEY_THRESHOLD)
        {
            // For lows, just take any bin below threshold (broad cuts are common)
            valleys.push_back({i + STREAM_A_BINS, val, freq});  // Store original mask index
        }
    }

    // === STREAM A: MID AND HIGH FREQUENCIES ===
    // Stream A bins 0-128, skip low frequencies (covered by Stream B)
    // Start from bin ~8 (~1.5kHz) to avoid overlap with Stream B
    for (int i = 8; i < STREAM_A_BINS - 2; ++i)
    {
        float val = mask[i];
        float freq = binToFreqA[i];

        // Only care about actual cuts (gain < threshold)
        if (val < VALLEY_THRESHOLD)
        {
            // Is this a local minimum? (lower than both neighbors)
            if (val <= mask[i - 1] && val <= mask[i + 1])
            {
                // Also check it's lower than or equal to 2-away neighbors
                if (val <= mask[i - 2] && val <= mask[i + 2])
                {
                    valleys.push_back({i, val, freq});
                }
            }
        }
    }

    // Sort by depth (deepest cuts first)
    std::sort(valleys.begin(), valleys.end(),
              [](const Candidate& a, const Candidate& b)
              {
                  return a.gain < b.gain;
              });

    // === FREQUENCY BUCKET SYSTEM ===
    // Split hunters into guaranteed allocations per band
    // Low: 20-500Hz (8 filters), Mid: 500-4kHz (12 filters), High: 4k-20kHz (12 filters)

    struct BandConfig
    {
        float minFreq;
        float maxFreq;
        int maxFilters;
    };

    const std::array<BandConfig, 3> bands = {{
        {20.0f, 500.0f, 8},      // Low Squad
        {500.0f, 4000.0f, 12},   // Mid Squad
        {4000.0f, 20000.0f, 12}  // High Squad
    }};

    const float minOctaveSpacing = 0.15f;  // ~1/6 octave minimum between hunters

    std::vector<Candidate> selectedValleys;
    selectedValleys.reserve(MAX_FILTERS);

    // Process each frequency band separately
    for (const auto& band : bands)
    {
        std::vector<Candidate> bandSelection;

        for (const auto& valley : valleys)
        {
            // Check if valley is in this band
            if (valley.freq < band.minFreq || valley.freq >= band.maxFreq)
                continue;

            if (static_cast<int>(bandSelection.size()) >= band.maxFilters)
                break;

            bool tooClose = false;

            // Check distance to already-selected valleys IN THIS BAND
            for (const auto& selected : bandSelection)
            {
                float octaveDistance = std::abs(std::log2(valley.freq / selected.freq));
                if (octaveDistance < minOctaveSpacing)
                {
                    tooClose = true;
                    break;
                }
            }

            if (!tooClose)
            {
                bandSelection.push_back(valley);
            }
        }

        // Add this band's selections to the master list
        for (const auto& v : bandSelection)
        {
            selectedValleys.push_back(v);
        }
    }

    // Assign selected valleys to hunters
    int activeCount = static_cast<int>(selectedValleys.size());

    // Convert floor dB to linear gain (e.g., -60dB -> 0.001)
    float floorGain = juce::Decibels::decibelsToGain(floorDb);

    for (int i = 0; i < activeCount; ++i)
    {
        auto& h = hunters[i];
        float targetFreq = selectedValleys[i].freq;
        float rawGain = selectedValleys[i].gain;

        // Apply strength: interpolate between 1.0 (no effect) and raw gain
        float targetGain = 1.0f - strength * (1.0f - rawGain);

        // Apply floor: don't cut deeper than floorDb
        targetGain = std::max(targetGain, floorGain);

        // Clamp frequency to valid range
        targetFreq = juce::jlimit(MIN_FREQ, MAX_FREQ, targetFreq);

        h.smoothFreq.setTargetValue(targetFreq);
        h.smoothGain.setTargetValue(targetGain);
        h.active = true;
    }

    // Set unused hunters to unity (bypass)
    for (int i = activeCount; i < MAX_FILTERS; ++i)
    {
        hunters[i].smoothGain.setTargetValue(1.0f);
        hunters[i].active = false;
    }
}

void ActiveFilterPool::process(juce::AudioBuffer<float>& buffer, const float* neuralMask)
{
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // Update hunter targets from the neural mask (once per block)
    updateTargets(neuralMask);

    float nyquist = static_cast<float>(sampleRate) * 0.5f;

    // Process each sample
    for (int s = 0; s < numSamples; ++s)
    {
        // Update each hunter
        for (auto& h : hunters)
        {
            // Get smoothed values
            float targetGain = h.smoothGain.getNextValue();
            float targetFreq = h.smoothFreq.getNextValue();

            // Clamp frequency
            targetFreq = juce::jlimit(MIN_FREQ, nyquist * 0.95f, targetFreq);

            // Check if coefficients need updating
            bool freqChanged = std::abs(targetFreq - h.currentFreq) > FREQ_UPDATE_THRESH;
            bool gainChanged = std::abs(targetGain - h.currentGain) > GAIN_UPDATE_THRESH;

            if (freqChanged || gainChanged)
            {
                h.currentFreq = targetFreq;
                h.currentGain = targetGain;

                // Dynamic Q: deeper cuts get sharper Q
                // gain 1.0 (no cut) -> Q = MIN_Q
                // gain 0.0 (full cut) -> Q = MAX_Q
                float dynamicQ = MIN_Q + (1.0f - targetGain) * (MAX_Q - MIN_Q);
                h.currentQ = dynamicQ;

                // Convert linear gain to the gain parameter for peak filter
                // For cuts: gain < 1.0 means attenuation at center frequency
                float peakGain = targetGain;

                // Clamp to prevent extreme values
                peakGain = juce::jlimit(0.01f, 1.0f, peakGain);

                // Update coefficients
                *h.coeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
                    sampleRate, h.currentFreq, dynamicQ, peakGain);

                // Apply new coefficients to both channels
                h.filter[0].coefficients = h.coeffs;
                h.filter[1].coefficients = h.coeffs;
            }

            // Process this sample through the filter (all channels)
            for (int ch = 0; ch < numChannels; ++ch)
            {
                float* data = buffer.getWritePointer(ch);
                data[s] = h.filter[ch].processSample(data[s]);
            }
        }
    }
}

std::array<ActiveFilterPool::FilterState, ActiveFilterPool::MAX_FILTERS>
ActiveFilterPool::getFilterStates() const
{
    std::array<FilterState, MAX_FILTERS> states;

    for (int i = 0; i < MAX_FILTERS; ++i)
    {
        states[i].freq = hunters[i].currentFreq;
        states[i].gain = hunters[i].currentGain;
        states[i].active = hunters[i].active;
    }

    return states;
}

float ActiveFilterPool::getBinFrequency(int bin) const
{
    if (bin >= 0 && bin < STREAM_A_BINS)
        return binToFreqA[bin];
    else if (bin >= STREAM_A_BINS && bin < TOTAL_MASK_SIZE)
        return binToFreqB[bin - STREAM_A_BINS];
    return 0.0f;
}
