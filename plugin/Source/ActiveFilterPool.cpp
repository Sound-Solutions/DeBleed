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
        h.holdSamplesRemaining = 0;
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

    // === DEDICATED BAND HUNTERS ===
    // Each band has dedicated hunter slots - they NEVER cross bands
    // Hunters 0-7:   Low band (20-500 Hz)
    // Hunters 8-19:  Mid band (500-4000 Hz)
    // Hunters 20-31: High band (4000-20000 Hz)

    struct BandConfig
    {
        float minFreq;
        float maxFreq;
        int startHunter;
        int numHunters;
        float minOctaveSpacing;  // 0 = no spacing (stacking allowed)
    };

    const std::array<BandConfig, 3> bands = {{
        {20.0f, 500.0f, 0, 8, 0.0f},           // Low Squad - hunters 0-7, stacking allowed
        {500.0f, 4000.0f, 8, 12, 1.0f/12.0f},  // Mid Squad - hunters 8-19, 1/12 octave spacing
        {4000.0f, 20000.0f, 20, 12, 1.0f/6.0f} // High Squad - hunters 20-31, 1/6 octave spacing
    }};

    // Helper to calculate target gain from mask (FIXED Range scaling in dB domain)
    auto calculateTargetGain = [this](float rawMaskGain) -> float {
        // cutAmount: how much the mask wants to cut (0 = no cut, 1 = full cut)
        float cutAmount = (1.0f - rawMaskGain) * strength;

        // Range (floorDb) determines maximum cut depth
        // floorDb = -60 means max 60dB cut, floorDb = -20 means max 20dB cut
        // Scale the cut proportionally in dB domain
        float maxCutDb = -floorDb;  // e.g., -(-60) = 60dB max
        float actualCutDb = cutAmount * maxCutDb;  // e.g., 0.5 * 60 = 30dB cut

        float targetGain = juce::Decibels::decibelsToGain(-actualCutDb);
        return juce::jlimit(0.001f, 1.0f, targetGain);
    };

    // Helper to calculate Q
    auto calculateQ = [](float freq, float rawMaskGain) -> float {
        if (freq < 500.0f)
        {
            // LOWS: Q = freq / binWidth (Stream B = 23.4 Hz/bin)
            static constexpr float BIN_WIDTH_HZ = 23.4375f;
            float baseQ = freq / BIN_WIDTH_HZ;
            float depthFactor = 1.0f - rawMaskGain;
            float flexFactor = 1.0f + depthFactor * 0.2f;
            return juce::jlimit(2.0f, 16.0f, baseQ * flexFactor);
        }
        else
        {
            // MIDS/HIGHS: Fixed surgical Q scaled by depth
            float depthFactor = 1.0f - rawMaskGain;
            return 8.0f + depthFactor * 16.0f;
        }
    };

    // Helper to sample mask at a frequency
    auto sampleMaskAtFreq = [mask, this](float freq) -> float {
        if (freq < 1500.0f)
        {
            int bin = static_cast<int>(freq * 2048.0f / 48000.0f);
            bin = juce::jlimit(0, STREAM_B_BINS - 1, bin);
            return mask[STREAM_A_BINS + bin];
        }
        else
        {
            int bin = static_cast<int>(freq * 256.0f / 48000.0f);
            bin = juce::jlimit(0, STREAM_A_BINS - 1, bin);
            return mask[bin];
        }
    };

    // Process each band with its dedicated hunters
    for (const auto& band : bands)
    {
        // Clamp band to HPF/LPF bounds
        float bandMin = std::max(band.minFreq, hpfBound);
        float bandMax = std::min(band.maxFreq, lpfBound);

        // Collect targets for THIS band only
        std::vector<FilterTarget> bandTargets;
        for (const auto& target : targets)
        {
            if (target.freq >= bandMin && target.freq < bandMax)
            {
                bandTargets.push_back(target);
            }
        }

        // Sort THIS band's targets by depth (deepest first)
        std::sort(bandTargets.begin(), bandTargets.end(),
                  [](const FilterTarget& a, const FilterTarget& b)
                  {
                      return a.gain < b.gain;
                  });

        // Select targets with spacing rules
        std::vector<FilterTarget> selectedTargets;
        for (const auto& target : bandTargets)
        {
            if (static_cast<int>(selectedTargets.size()) >= band.numHunters)
                break;

            // Check spacing (if required for this band)
            bool tooClose = false;
            if (band.minOctaveSpacing > 0.0f)
            {
                for (const auto& selected : selectedTargets)
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
                selectedTargets.push_back(target);
            }
        }

        // === ASSIGN TARGETS TO THIS BAND'S DEDICATED HUNTERS ===
        for (int h = 0; h < band.numHunters; ++h)
        {
            int hunterIdx = band.startHunter + h;
            auto& hunter = hunters[hunterIdx];

            if (h < static_cast<int>(selectedTargets.size()))
            {
                // This hunter has a target assigned
                const auto& target = selectedTargets[h];
                float targetFreq = juce::jlimit(bandMin, bandMax, target.freq);
                float rawGain = target.gain;
                float targetGain = calculateTargetGain(rawGain);
                float targetQ = calculateQ(targetFreq, rawGain);

                // Check if hunter is in tightness period (can't change frequency)
                bool inTightness = hunter.active && hunter.samplesAtCurrentFreq < tightnessSamples;

                if (inTightness)
                {
                    // TIGHTNESS: Hunter stays at current frequency, but gain still follows mask
                    // Sample mask at hunter's current (locked) frequency
                    float maskAtCurrentFreq = sampleMaskAtFreq(hunter.currentFreq);

                    // If mask at this frequency is above valley threshold, release to unity
                    // (no real valley here anymore - let the hunter release)
                    if (maskAtCurrentFreq >= VALLEY_THRESHOLD)
                    {
                        hunter.smoothGain.setTargetValue(1.0f);
                        hunter.smoothQ.setTargetValue(MIN_Q);
                        hunter.targetGainFromMask = 1.0f;
                    }
                    else
                    {
                        // Still a valley at this frequency - update gain based on mask
                        float lockedGain = calculateTargetGain(maskAtCurrentFreq);
                        float lockedQ = calculateQ(hunter.currentFreq, maskAtCurrentFreq);
                        hunter.smoothGain.setTargetValue(lockedGain);
                        hunter.smoothQ.setTargetValue(lockedQ);
                        hunter.targetGainFromMask = maskAtCurrentFreq;
                    }
                    // Don't reset samplesAtCurrentFreq - keep counting
                }
                else
                {
                    // Hunter can move to new frequency
                    bool freqChanged = std::abs(targetFreq - hunter.currentFreq) > FREQ_UPDATE_THRESH;

                    hunter.smoothFreq.setTargetValue(targetFreq);
                    hunter.smoothGain.setTargetValue(targetGain);
                    hunter.smoothQ.setTargetValue(targetQ);
                    hunter.targetGainFromMask = rawGain;

                    if (freqChanged)
                    {
                        hunter.samplesAtCurrentFreq = 0;  // Reset tightness counter
                    }
                }
                hunter.active = true;
            }
            else
            {
                // No target for this hunter - set to unity (bypass)
                hunter.smoothGain.setTargetValue(1.0f);
                hunter.smoothQ.setTargetValue(MIN_Q);
                hunter.targetGainFromMask = 1.0f;
                hunter.active = false;
                hunter.samplesAtCurrentFreq = 0;
            }
        }
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

    // Calculate hold time in samples
    int holdSamples = static_cast<int>(holdMs * sampleRate / 1000.0f);

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

            // Compressor-style asymmetric smoothing for gain with HOLD:
            // - Gain DECREASING (cut engaging) = use attack time, reset hold counter
            // - In hold period = stay at current depth
            // - Gain INCREASING (cut releasing) = use release time (only after hold expires)
            float coeff;
            float smoothedGain;

            if (targetGain < h.currentGain)
            {
                // Cut is engaging - use attack, reset hold
                coeff = attackCoeff;
                smoothedGain = h.currentGain * coeff + targetGain * (1.0f - coeff);
                h.holdSamplesRemaining = holdSamples;
            }
            else if (h.holdSamplesRemaining > 0)
            {
                // In hold period - stay at current depth
                h.holdSamplesRemaining--;
                smoothedGain = h.currentGain;
            }
            else
            {
                // Hold expired - release can happen
                coeff = releaseCoeff;
                smoothedGain = h.currentGain * coeff + targetGain * (1.0f - coeff);
            }

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
