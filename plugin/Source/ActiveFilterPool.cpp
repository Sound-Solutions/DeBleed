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
        h.smoothQ.reset(sampleRate, SMOOTHING_TIME);
        h.smoothGain.setCurrentAndTargetValue(1.0f);
        h.smoothFreq.setCurrentAndTargetValue(1000.0f);
        h.smoothQ.setCurrentAndTargetValue(MIN_Q);

        // Initialize filter coefficients to unity peak at 1kHz
        *h.coeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
            sampleRate, 1000.0f, MIN_Q, 1.0f);

        // Prepare stereo filters
        for (int ch = 0; ch < 2; ++ch)
        {
            h.filter[ch].coefficients = h.coeffs;
            h.filter[ch].prepare(spec);
        }

        h.currentFreq = 1000.0f;
        h.currentGain = 1.0f;
        h.currentQ = MIN_Q;
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
        h.smoothQ.setCurrentAndTargetValue(MIN_Q);
        h.currentGain = 1.0f;
        h.currentFreq = 1000.0f;
        h.currentQ = MIN_Q;
        h.targetGainFromMask = 1.0f;
        h.active = false;
        h.samplesAtCurrentFreq = 0;
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
            h.smoothQ.setTargetValue(MIN_Q);
            h.targetGainFromMask = 1.0f;
            h.active = false;
        }
        return;
    }

    // Calculate tightness in samples
    int tightnessSamples = static_cast<int>(tightnessMs * sampleRate / 1000.0f);

    // === COLLECT TARGETS FROM MASK ===
    struct FilterTarget
    {
        float freq;
        float gain;  // Raw mask gain (0.0-1.0)
    };
    std::vector<FilterTarget> targets;
    targets.reserve(128);

    // === STREAM B: HIGH-RESOLUTION LOW FREQUENCIES ===
    const float* streamB = mask + STREAM_A_BINS;

    for (int i = 1; i < STREAM_B_BINS - 1; ++i)
    {
        float val = streamB[i];
        float freq = binToFreqB[i];

        // Skip frequencies outside HPF/LPF bounds
        if (freq < hpfBound || freq > lpfBound)
            continue;

        if (freq > 1500.0f)
            break;

        if (val < VALLEY_THRESHOLD)
        {
            targets.push_back({freq, val});
        }
    }

    // === STREAM A: MID AND HIGH FREQUENCIES ===
    for (int i = 8; i < STREAM_A_BINS - 1; ++i)
    {
        float val = mask[i];
        float freq = binToFreqA[i];

        // Skip frequencies outside HPF/LPF bounds
        if (freq < hpfBound || freq > lpfBound)
            continue;

        if (val < VALLEY_THRESHOLD)
        {
            targets.push_back({freq, val});
        }
    }

    // Sort by depth (deepest cuts first - these get priority)
    std::sort(targets.begin(), targets.end(),
              [](const FilterTarget& a, const FilterTarget& b)
              {
                  return a.gain < b.gain;
              });

    // === FREQUENCY BUCKET SYSTEM ===
    // Lows: can stack (no spacing), Mids/Highs: minimum spacing for coverage

    struct BandConfig
    {
        float minFreq;
        float maxFreq;
        int maxFilters;
        float minOctaveSpacing;  // 0 = no spacing (stacking allowed)
    };

    const std::array<BandConfig, 3> bands = {{
        {20.0f, 500.0f, 8, 0.0f},           // Low Squad - stacking allowed
        {500.0f, 4000.0f, 12, 1.0f/12.0f},  // Mid Squad - 1/12 octave spacing
        {4000.0f, 20000.0f, 12, 1.0f/6.0f}  // High Squad - 1/6 octave spacing
    }};

    std::vector<FilterTarget> selectedTargets;
    selectedTargets.reserve(MAX_FILTERS);

    // Process each frequency band with band-specific spacing rules
    for (const auto& band : bands)
    {
        // Clamp band to HPF/LPF bounds
        float bandMin = std::max(band.minFreq, hpfBound);
        float bandMax = std::min(band.maxFreq, lpfBound);
        if (bandMin >= bandMax)
            continue;

        std::vector<FilterTarget> bandSelection;

        for (const auto& target : targets)
        {
            if (target.freq < bandMin || target.freq >= bandMax)
                continue;

            if (static_cast<int>(bandSelection.size()) >= band.maxFilters)
                break;

            // Check spacing (if required for this band)
            bool tooClose = false;
            if (band.minOctaveSpacing > 0.0f)
            {
                for (const auto& selected : bandSelection)
                {
                    float octaveDistance = std::abs(std::log2(target.freq / selected.freq));
                    if (octaveDistance < band.minOctaveSpacing)
                    {
                        tooClose = true;
                        break;
                    }
                }
            }

            if (!tooClose)
            {
                bandSelection.push_back(target);
            }
        }

        // Add this band's selections to master list
        for (const auto& t : bandSelection)
        {
            selectedTargets.push_back(t);
        }
    }

    // === ASSIGN TARGETS TO AVAILABLE HUNTERS ===
    // A hunter is available if: !active OR samplesAtCurrentFreq >= tightnessSamples

    float floorGain = juce::Decibels::decibelsToGain(floorDb);

    // Collect indices of available hunters (tightness period elapsed)
    std::vector<int> availableHunters;
    availableHunters.reserve(MAX_FILTERS);
    for (int i = 0; i < MAX_FILTERS; ++i)
    {
        if (!hunters[i].active || hunters[i].samplesAtCurrentFreq >= tightnessSamples)
        {
            availableHunters.push_back(i);
        }
    }

    // For hunters still in tightness period, update their gain from current mask position
    for (int i = 0; i < MAX_FILTERS; ++i)
    {
        auto& h = hunters[i];
        if (h.active && h.samplesAtCurrentFreq < tightnessSamples)
        {
            // Sample mask at this hunter's current frequency
            float freq = h.currentFreq;
            float maskGain = 1.0f;

            if (freq < 1500.0f)
            {
                int bin = static_cast<int>(freq * 2048.0f / 48000.0f);
                bin = juce::jlimit(0, STREAM_B_BINS - 1, bin);
                maskGain = mask[STREAM_A_BINS + bin];
            }
            else
            {
                int bin = static_cast<int>(freq * 256.0f / 48000.0f);
                bin = juce::jlimit(0, STREAM_A_BINS - 1, bin);
                maskGain = mask[bin];
            }

            // Apply strength to determine cut depth, then scale by floor (Range)
            float cutAmount = (1.0f - maskGain) * strength;  // How much to cut (0-1)
            float scaledCut = cutAmount * (1.0f - floorGain);  // Scale by Range
            float targetGain = 1.0f - scaledCut;
            targetGain = juce::jlimit(floorGain, 1.0f, targetGain);
            h.smoothGain.setTargetValue(targetGain);
            h.targetGainFromMask = maskGain;
        }
    }

    // Assign targets to available hunters
    int targetsToAssign = std::min(static_cast<int>(selectedTargets.size()),
                                   static_cast<int>(availableHunters.size()));

    for (int t = 0; t < targetsToAssign; ++t)
    {
        int hunterIdx = availableHunters[t];
        auto& h = hunters[hunterIdx];
        const auto& target = selectedTargets[t];

        float targetFreq = juce::jlimit(hpfBound, lpfBound, target.freq);
        float rawGain = target.gain;

        // === NEW Q CALCULATION ===
        // Lows (<500Hz): Q matches bin width with slight flex
        // Mids/Highs (500Hz+): Fixed surgical Q scaled by depth
        float targetQ;
        if (targetFreq < 500.0f)
        {
            // LOWS: Q = freq / binWidth (Stream B = 23.4 Hz/bin)
            static constexpr float BIN_WIDTH_HZ = 23.4375f;  // 48000 / 2048
            float baseQ = targetFreq / BIN_WIDTH_HZ;

            // Allow +/-20% flex based on mask value (deeper = tighter Q)
            float depthFactor = 1.0f - rawGain;  // 0 = no cut, 1 = full cut
            float flexFactor = 1.0f + depthFactor * 0.2f;  // Up to +20% Q for deep cuts
            targetQ = juce::jlimit(2.0f, 16.0f, baseQ * flexFactor);
        }
        else
        {
            // MIDS/HIGHS: Fixed surgical Q scaled by attenuation depth
            float depthFactor = 1.0f - rawGain;  // 0 = no cut, 1 = full cut
            float baseQ = 8.0f;
            float maxQ = 24.0f;
            targetQ = baseQ + depthFactor * (maxQ - baseQ);
        }

        // Apply strength to determine cut depth, then scale by floor (Range)
        float cutAmount = (1.0f - rawGain) * strength;  // How much to cut (0-1)
        float scaledCut = cutAmount * (1.0f - floorGain);  // Scale by Range
        float targetGain = 1.0f - scaledCut;
        targetGain = juce::jlimit(floorGain, 1.0f, targetGain);

        // Hysteresis check
        float currentTargetGain = h.smoothGain.getTargetValue();
        float currentTargetFreq = h.smoothFreq.getTargetValue();

        float gainDelta = std::abs(targetGain - currentTargetGain);
        float freqRatio = (currentTargetFreq > 0.0f)
            ? std::abs(std::log2(targetFreq / currentTargetFreq))
            : 1.0f;

        bool shouldUpdate = (gainDelta > HYSTERESIS_GAIN) || (freqRatio > HYSTERESIS_FREQ);

        if (shouldUpdate || !h.active)
        {
            // Check if frequency is actually changing
            bool freqChanged = std::abs(targetFreq - h.currentFreq) > FREQ_UPDATE_THRESH;
            if (freqChanged)
            {
                h.samplesAtCurrentFreq = 0;  // Reset tightness counter on freq change
            }

            h.smoothFreq.setTargetValue(targetFreq);
            h.smoothGain.setTargetValue(targetGain);
            h.smoothQ.setTargetValue(targetQ);
            h.targetGainFromMask = rawGain;
        }
        h.active = true;
    }

    // Set remaining available hunters to unity (bypass)
    for (int t = targetsToAssign; t < static_cast<int>(availableHunters.size()); ++t)
    {
        int hunterIdx = availableHunters[t];
        hunters[hunterIdx].smoothGain.setTargetValue(1.0f);
        hunters[hunterIdx].smoothQ.setTargetValue(MIN_Q);
        hunters[hunterIdx].active = false;
        hunters[hunterIdx].samplesAtCurrentFreq = 0;
    }
}

void ActiveFilterPool::process(juce::AudioBuffer<float>& buffer, const float* neuralMask)
{
    const int numSamples = buffer.getNumSamples();
    const int numChannels = juce::jmin(buffer.getNumChannels(), 2);

    // Store block size for tightness tracking
    currentBlockSize = numSamples;

    // Update hunter targets from the neural mask (once per block)
    updateTargets(neuralMask);

    // Increment tightness counters for all active hunters
    for (auto& h : hunters)
    {
        if (h.active)
        {
            h.samplesAtCurrentFreq += numSamples;
        }
    }

    float nyquist = static_cast<float>(sampleRate) * 0.5f;

    // Calculate attack/release coefficients (COMPRESSOR-style)
    // Attack = gain DECREASING (cut engaging, signal getting cut)
    // Release = gain INCREASING (cut releasing, signal coming back)
    float attackCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * attackMs / 1000.0f));
    float releaseCoeff = std::exp(-1.0f / (static_cast<float>(sampleRate) * releaseMs / 1000.0f));

    // Process each sample
    for (int s = 0; s < numSamples; ++s)
    {
        // Update each hunter
        for (auto& h : hunters)
        {
            // Get target values (smoothGain holds target, we'll manually smooth)
            float targetGain = h.smoothGain.getTargetValue();
            float targetFreq = h.smoothFreq.getNextValue();
            float targetQ = h.smoothQ.getNextValue();

            // Compressor-style asymmetric smoothing for gain:
            // - Gain DECREASING (cut engaging) = use attack time
            // - Gain INCREASING (cut releasing) = use release time
            float coeff = (targetGain < h.currentGain) ? attackCoeff : releaseCoeff;
            float smoothedGain = h.currentGain * coeff + targetGain * (1.0f - coeff);

            // Clamp frequency and Q
            targetFreq = juce::jlimit(hpfBound, std::min(lpfBound, nyquist * 0.95f), targetFreq);
            targetQ = juce::jlimit(MIN_Q, MAX_Q, targetQ);

            // Check if coefficients need updating (use smoothedGain for comparison)
            bool freqChanged = std::abs(targetFreq - h.currentFreq) > FREQ_UPDATE_THRESH;
            bool gainChanged = std::abs(smoothedGain - h.currentGain) > GAIN_UPDATE_THRESH;
            bool qChanged = std::abs(targetQ - h.currentQ) > Q_UPDATE_THRESH;

            if (freqChanged || gainChanged || qChanged)
            {
                h.currentFreq = targetFreq;
                h.currentGain = smoothedGain;  // Use smoothed gain with attack/release
                h.currentQ = targetQ;

                // Convert linear gain to the gain parameter for peak filter
                // For cuts: gain < 1.0 means attenuation at center frequency
                float peakGain = juce::jlimit(0.01f, 1.0f, smoothedGain);

                // Update coefficients with width-adaptive Q
                *h.coeffs = *juce::dsp::IIR::Coefficients<float>::makePeakFilter(
                    sampleRate, h.currentFreq, h.currentQ, peakGain);

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
        states[i].q = hunters[i].currentQ;
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
