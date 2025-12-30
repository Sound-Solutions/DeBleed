#include "SidechainAnalyzer.h"

SidechainAnalyzer::SidechainAnalyzer()
{
    rawBandGains.fill(1.0f);
    smoothedBandGains.fill(1.0f);
}

void SidechainAnalyzer::prepare(double newSampleRate, int maxBlockSize,
                                 const std::array<float, NUM_IIR_BANDS>& iirCenterFreqs)
{
    sampleRate = newSampleRate;

    // Prepare STFT (analysis only, Quality mode for best mask resolution)
    stftProcessor.setMode(STFTProcessor::Mode::Quality);
    stftProcessor.prepare(sampleRate, maxBlockSize);

    // Prepare neural engine
    int maxFrames = (maxBlockSize / stftProcessor.getHopLength()) + 2;
    neuralEngine.prepare(maxFrames);

    // Prepare band mapper with IIR center frequencies
    bandMapper.prepare(sampleRate, iirCenterFreqs);

    // Prepare envelope followers
    envelopeBank.prepare(sampleRate);

    reset();
}

void SidechainAnalyzer::reset()
{
    stftProcessor.reset();
    envelopeBank.reset();

    rawBandGains.fill(1.0f);
    smoothedBandGains.fill(1.0f);

    lastMask = nullptr;
    lastMagnitude = nullptr;
}

void SidechainAnalyzer::analyze(const float* input, int numSamples)
{
    // Run dual-stream STFT analysis
    int numFrames = stftProcessor.processBlock(input, numSamples);

    if (numFrames > 0 && neuralEngine.isModelLoaded())
    {
        // Get dual-stream features (257 bins) or single-stream magnitude (129 bins)
        const float* inputFeatures;
        int inputFeatureSize;

        if (stftProcessor.isDualStreamEnabled())
        {
            // Use concatenated dual-stream features [Stream A (129) | Stream B bass (128)]
            inputFeatures = stftProcessor.getDualStreamFeatures();
            inputFeatureSize = N_TOTAL_FEATURES;  // 257
        }
        else
        {
            // Legacy single-stream mode
            inputFeatures = stftProcessor.getMagnitudeData();
            inputFeatureSize = N_FREQ_BINS;  // 129
        }

        // Store Stream A magnitude for visualization
        lastMagnitude = stftProcessor.getMagnitudeData();

        // Run neural network inference with dual-stream features
        lastMask = neuralEngine.process(inputFeatures, numFrames);

        if (lastMask != nullptr)
        {
            // Use the last frame's mask for band gains
            const float* frameMask = lastMask + (numFrames - 1) * N_FREQ_BINS;

            // Convert floor/range from dB to linear
            // floorDb = -60 → floorLinear ≈ 0.001 (deep cut possible)
            // floorDb = 0   → floorLinear = 1.0 (no cut, everything passes)
            float floorLinear = juce::Decibels::decibelsToGain(floorDb);

            // Apply floor/range scaling to mask
            std::array<float, N_FREQ_BINS> processedMask;
            for (int i = 0; i < N_FREQ_BINS; ++i)
            {
                float maskVal = frameMask[i];

                // Clamp raw mask to [0, 1]
                maskVal = juce::jlimit(0.0f, 1.0f, maskVal);

                // Scale mask from [0,1] to [floorLinear, 1.0]
                // mask=0 → floorLinear (maximum cut, limited by floor)
                // mask=1 → 1.0 (full signal, no cut)
                maskVal = floorLinear + maskVal * (1.0f - floorLinear);

                processedMask[i] = maskVal;
            }

            // Map FFT mask to IIR band gains with strength
            bandMapper.mapWithStrength(processedMask.data(), rawBandGains, strength);
        }
    }
    else if (!neuralEngine.isModelLoaded())
    {
        // No model - pass through (unity gain)
        rawBandGains.fill(1.0f);
    }

    // Apply envelope followers for smooth transitions
    envelopeBank.processBlock(rawBandGains, smoothedBandGains, numSamples);
}

bool SidechainAnalyzer::loadModel(const juce::String& modelPath)
{
    return neuralEngine.loadModel(modelPath);
}

void SidechainAnalyzer::unloadModel()
{
    neuralEngine.unloadModel();
}

void SidechainAnalyzer::setStrength(float newStrength)
{
    strength = juce::jlimit(0.0f, 2.0f, newStrength);
}

void SidechainAnalyzer::setAttack(float attackMs)
{
    envelopeBank.setAttack(attackMs);
}

void SidechainAnalyzer::setRelease(float releaseMs)
{
    envelopeBank.setRelease(releaseMs);
}

void SidechainAnalyzer::setFrequencyRange(float lowHz, float highHz)
{
    bandMapper.setFrequencyRange(lowHz, highHz);
}

void SidechainAnalyzer::setThreshold(float threshDb)
{
    thresholdDb = threshDb;
}

void SidechainAnalyzer::setFloor(float floorDbVal)
{
    floorDb = floorDbVal;
}

float SidechainAnalyzer::getAverageGainReduction() const
{
    // Calculate average gain reduction across all bands
    float sum = 0.0f;
    for (int i = 0; i < NUM_IIR_BANDS; ++i)
    {
        sum += smoothedBandGains[i];
    }
    float avgGain = sum / NUM_IIR_BANDS;

    // Convert to dB (negative values = reduction)
    if (avgGain > 0.0001f)
        return juce::Decibels::gainToDecibels(avgGain);
    else
        return -60.0f;
}
