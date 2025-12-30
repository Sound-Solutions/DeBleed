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

    // Debug: count frames for periodic logging
    static int debugFrameCount = 0;
    debugFrameCount += numFrames;

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
            // Dual-stream mask: 257 bins [Stream A (129) | Stream B bass (128)]
            const float* frameMask = lastMask + (numFrames - 1) * N_OUTPUT_BINS;

            // Convert floor/range from dB to linear
            // floorDb = -60 → floorLinear ≈ 0.001 (deep cut possible)
            // floorDb = 0   → floorLinear = 1.0 (no cut, everything passes)
            float floorLinear = juce::Decibels::decibelsToGain(floorDb);

            // Apply floor/range scaling to dual-stream mask (257 bins)
            std::array<float, N_OUTPUT_BINS> processedMask;
            for (int i = 0; i < N_OUTPUT_BINS; ++i)
            {
                float maskVal = frameMask[i];

                // Clamp raw mask to [0, 1]
                maskVal = juce::jlimit(0.0f, 1.0f, maskVal);

                // Neural network outputs KEEP mask (IRM = clean/mixture):
                // mask ≈ 1.0 means "target present, keep it"
                // mask ≈ 0.0 means "bleed present, cut it"
                // NO INVERSION - model trained with clean_audio_dir = target

                // Scale mask from [0,1] to [floorLinear, 1.0]
                // mask=0 → floorLinear (maximum cut, limited by floor)
                // mask=1 → 1.0 (full signal, no cut)
                maskVal = floorLinear + maskVal * (1.0f - floorLinear);

                processedMask[i] = maskVal;
            }

            // Map dual-stream mask to IIR band gains with strength
            // Uses Stream B (23Hz resolution) for bass, Stream A for highs
            bandMapper.mapDualStreamWithStrength(processedMask.data(), rawBandGains, strength);

            // Debug: Log mask statistics every ~1 second (about 47 frames at 48kHz with 1024 hop)
            if (debugFrameCount >= 47)
            {
                debugFrameCount = 0;

                // Log model info on first debug output
                static bool modelInfoLogged = false;
                if (!modelInfoLogged && neuralEngine.isModelLoaded())
                {
                    juce::File debugFile("/Users/ksellarsm4lt/Documents/DeBleed/debug.txt");
                    juce::String modelInfo = juce::String::formatted(
                        "=== MODEL INFO ===\nInput features: %d\nDual-stream: %s\nPath: %s\n==================\n",
                        neuralEngine.getModelInputFeatures(),
                        neuralEngine.isDualStreamModel() ? "YES" : "NO",
                        neuralEngine.getModelPath().toRawUTF8());
                    debugFile.appendText(modelInfo);
                    modelInfoLogged = true;
                }

                // Get raw mask from neural network (BEFORE floor scaling)
                const float* rawNNMask = lastMask + (numFrames - 1) * N_OUTPUT_BINS;
                float nnMin = 1.0f, nnMax = 0.0f, nnAvg = 0.0f;
                for (int i = 0; i < N_OUTPUT_BINS; ++i) {
                    nnMin = std::min(nnMin, rawNNMask[i]);
                    nnMax = std::max(nnMax, rawNNMask[i]);
                    nnAvg += rawNNMask[i];
                }
                nnAvg /= N_OUTPUT_BINS;

                // STFT feature levels (what the NN actually sees)
                // Stream A magnitude (bins 0-128) and Stream B (bins 129-256)
                float stftAMin = 1e10f, stftAMax = 0.0f, stftASum = 0.0f;
                float stftBMin = 1e10f, stftBMax = 0.0f, stftBSum = 0.0f;
                for (int i = 0; i < 129; ++i) {
                    float val = inputFeatures[i];
                    stftAMin = std::min(stftAMin, val);
                    stftAMax = std::max(stftAMax, val);
                    stftASum += val;
                }
                for (int i = 129; i < 257; ++i) {
                    float val = inputFeatures[i];
                    stftBMin = std::min(stftBMin, val);
                    stftBMax = std::max(stftBMax, val);
                    stftBSum += val;
                }

                // RAW band gains by region (before envelope smoothing)
                // Hybrid 192-band topology: 0-31 = bass (20-500Hz), 32-191 = highs (500Hz-20kHz)
                float rawBassAvg = 0.0f, rawMidAvg = 0.0f, rawHighAvg = 0.0f;
                for (int i = 0; i < 32; ++i) rawBassAvg += rawBandGains[i];           // Bands 0-31 (bass)
                for (int i = 32; i < 96; ++i) rawMidAvg += rawBandGains[i];           // Bands 32-95 (mids ~500Hz-2kHz)
                for (int i = 160; i < NUM_IIR_BANDS; ++i) rawHighAvg += rawBandGains[i]; // Bands 160-191 (highs ~10kHz+)
                rawBassAvg /= 32.0f;
                rawMidAvg /= 64.0f;
                rawHighAvg /= 32.0f;

                // SMOOTHED band gains (what actually gets applied to IIR filters)
                float smoothBassAvg = 0.0f, smoothMidAvg = 0.0f, smoothHighAvg = 0.0f;
                for (int i = 0; i < 32; ++i) smoothBassAvg += smoothedBandGains[i];
                for (int i = 32; i < 96; ++i) smoothMidAvg += smoothedBandGains[i];
                for (int i = 160; i < NUM_IIR_BANDS; ++i) smoothHighAvg += smoothedBandGains[i];
                smoothBassAvg /= 32.0f;
                smoothMidAvg /= 64.0f;
                smoothHighAvg /= 32.0f;

                // Write to debug file
                juce::File debugFile("/Users/ksellarsm4lt/Documents/DeBleed/debug.txt");
                juce::String line = juce::String::formatted(
                    "NN[%.3f/%.3f] | Raw[%.3f/%.3f/%.3f] | Smooth[%.3f/%.3f/%.3f]\n",
                    nnMin, nnMax, rawBassAvg, rawMidAvg, rawHighAvg,
                    smoothBassAvg, smoothMidAvg, smoothHighAvg);
                debugFile.appendText(line);
            }
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
