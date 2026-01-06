#include "SpectralVAD.h"
#include <cmath>

SpectralVAD::SpectralVAD()
{
    // Initialize with default frequencies
    for (int i = 0; i < NUM_BANDS; ++i)
        bandFrequencies_[i] = DEFAULT_FREQUENCIES[i];

    // Initialize with vocal-optimized weights (based on formant frequencies)
    // These emphasize the 800Hz-2500Hz range where vocal formants F1-F3 live,
    // and de-emphasize low frequencies (kick bleed) and high frequencies (cymbal bleed)
    //
    // Band 0: 200Hz  - kick drum territory, low weight
    // Band 1: 400Hz  - bass/low vocal, moderate weight
    // Band 2: 800Hz  - F1 formant zone, good weight
    // Band 3: 1600Hz - F2 formant zone, highest weight (vocal "body")
    // Band 4: 2500Hz - F2/F3 zone, highest weight (vocal "presence")
    // Band 5: 4000Hz - vocal brightness, moderate weight
    // Band 6: 6000Hz - cymbal territory, low weight
    // Band 7: 8000Hz - cymbal/air, lowest weight
    bandWeights_[0] = 0.05f;   // 200Hz
    bandWeights_[1] = 0.08f;   // 400Hz
    bandWeights_[2] = 0.15f;   // 800Hz
    bandWeights_[3] = 0.25f;   // 1600Hz - F2 formant, most important
    bandWeights_[4] = 0.25f;   // 2500Hz - F2/F3 presence
    bandWeights_[5] = 0.12f;   // 4000Hz
    bandWeights_[6] = 0.06f;   // 6000Hz
    bandWeights_[7] = 0.04f;   // 8000Hz

    // Initialize atomic band energies
    for (int i = 0; i < NUM_BANDS; ++i)
        bandEnergies_[i].store(0.0f);
}

void SpectralVAD::prepare(double sampleRate)
{
    sampleRate_ = sampleRate;
    updateCoefficients();
    reset();
}

void SpectralVAD::reset()
{
    for (auto& filter : filters_)
        filter.reset();

    smoothedConfidence_ = 0.0f;
    currentConfidence_.store(0.0f);
    pitchConfidence_ = 0.0f;
    formantRatio_ = 0.0f;
    pitchBufferIndex_ = 0;
    std::fill(pitchBuffer_.begin(), pitchBuffer_.end(), 0.0f);

    for (int i = 0; i < NUM_BANDS; ++i)
        bandEnergies_[i].store(0.0f);
}

float SpectralVAD::computePitchConfidence()
{
    // Autocorrelation-based pitch detection
    // Vocals have clear periodic structure, drums don't
    //
    // We compute autocorrelation at lags corresponding to vocal pitch range
    // (80-400Hz = periods of 2.5ms to 12.5ms)
    // High autocorrelation peak = pitched (vocal)
    // Low/no peak = unpitched (drums, noise)

    int minLag = static_cast<int>(sampleRate_ / MAX_VOCAL_PITCH_HZ);  // ~120 samples at 48kHz for 400Hz
    int maxLag = static_cast<int>(sampleRate_ / MIN_VOCAL_PITCH_HZ);  // ~600 samples at 48kHz for 80Hz
    maxLag = std::min(maxLag, PITCH_BUFFER_SIZE / 2);  // Don't exceed half buffer

    // Compute energy at lag 0 (for normalization)
    float energy = 0.0f;
    for (int i = 0; i < PITCH_BUFFER_SIZE; ++i)
        energy += pitchBuffer_[i] * pitchBuffer_[i];

    if (energy < 1e-10f)
        return 0.0f;

    // Find peak autocorrelation in vocal pitch range
    float maxCorr = 0.0f;

    // Sample at several lags for efficiency (every 4 samples)
    for (int lag = minLag; lag <= maxLag; lag += 4)
    {
        float corr = 0.0f;
        for (int i = 0; i < PITCH_BUFFER_SIZE - lag; ++i)
        {
            int idx1 = (pitchBufferIndex_ + i) % PITCH_BUFFER_SIZE;
            int idx2 = (pitchBufferIndex_ + i + lag) % PITCH_BUFFER_SIZE;
            corr += pitchBuffer_[idx1] * pitchBuffer_[idx2];
        }

        // Normalize
        corr /= energy;

        if (corr > maxCorr)
            maxCorr = corr;
    }

    // Map correlation to confidence
    // Vocals typically have correlation > 0.5, drums < 0.2
    float pitchConf = (maxCorr - 0.15f) / 0.45f;  // Maps 0.15-0.6 to 0-1
    return std::clamp(pitchConf, 0.0f, 1.0f);
}

void SpectralVAD::computeBandpassCoeffs(BandpassFilter& filter, float freqHz, float q)
{
    // Bandpass filter design using bilinear transform
    float omega = 2.0f * juce::MathConstants<float>::pi * freqHz / static_cast<float>(sampleRate_);
    float sinOmega = std::sin(omega);
    float cosOmega = std::cos(omega);
    float alpha = sinOmega / (2.0f * q);

    float a0 = 1.0f + alpha;

    filter.b0 = (alpha) / a0;
    filter.b1 = 0.0f;
    filter.b2 = (-alpha) / a0;
    filter.a1 = (-2.0f * cosOmega) / a0;
    filter.a2 = (1.0f - alpha) / a0;
}

void SpectralVAD::updateCoefficients()
{
    if (sampleRate_ <= 0.0)
        return;

    // Update bandpass filter coefficients
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        // Clamp frequency to valid range
        float freq = std::clamp(bandFrequencies_[i], 20.0f, static_cast<float>(sampleRate_) * 0.45f);
        computeBandpassCoeffs(filters_[i], freq, bandQ_);
    }

    // Update envelope follower coefficients
    float attackMs = attackMs_.load();
    float releaseMs = releaseMs_.load();

    float attackTau = attackMs / 1000.0f;
    float releaseTau = releaseMs / 1000.0f;

    attackCoeff_ = 1.0f - std::exp(-1.0f / (attackTau * static_cast<float>(sampleRate_)));
    releaseCoeff_ = 1.0f - std::exp(-1.0f / (releaseTau * static_cast<float>(sampleRate_)));

    // Confidence smoothing (slightly slower than envelope)
    confAttackCoeff_ = 1.0f - std::exp(-1.0f / (attackTau * 2.0f * static_cast<float>(sampleRate_)));
    confReleaseCoeff_ = 1.0f - std::exp(-1.0f / (releaseTau * 2.0f * static_cast<float>(sampleRate_)));
}

float SpectralVAD::processSample(float sample)
{
    float thresholdDb = thresholdDb_.load();
    float kneeDb = kneeDb_.load();

    // Process through each bandpass filter and update envelopes
    float weightedEnergy = 0.0f;
    float totalEnergy = 0.0f;
    float formantEnergy = 0.0f;  // Energy in formant bands (800Hz-2.5kHz = bands 2,3,4)

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        // Filter the sample
        float bandOutput = filters_[i].process(sample);

        // Envelope follower (RMS-style)
        float bandLevel = bandOutput * bandOutput;  // Squared for RMS
        float coeff = (bandLevel > filters_[i].envelope) ? attackCoeff_ : releaseCoeff_;
        filters_[i].envelope = coeff * bandLevel + (1.0f - coeff) * filters_[i].envelope;

        // Apply learned weight and accumulate
        weightedEnergy += bandWeights_[i] * filters_[i].envelope;
        totalEnergy += filters_[i].envelope;

        // Formant bands: 800Hz, 1600Hz, 2500Hz (indices 2, 3, 4)
        if (i >= 2 && i <= 4)
            formantEnergy += filters_[i].envelope;

        // Store for visualization (less frequent update)
        bandEnergies_[i].store(filters_[i].envelope);
    }

    // Compute formant ratio (vocals have high energy in formant region)
    formantRatio_ = (totalEnergy > 1e-10f) ? (formantEnergy / totalEnergy) : 0.0f;

    // Store sample in pitch buffer for autocorrelation
    pitchBuffer_[pitchBufferIndex_] = sample;
    pitchBufferIndex_ = (pitchBufferIndex_ + 1) % PITCH_BUFFER_SIZE;

    // Compute pitch confidence every 256 samples (for efficiency)
    static int pitchUpdateCounter = 0;
    if (++pitchUpdateCounter >= 256)
    {
        pitchUpdateCounter = 0;
        pitchConfidence_ = computePitchConfidence();
    }

    // Convert weighted energy to dB
    float energyDb = 10.0f * std::log10(weightedEnergy + 1e-10f);

    // Soft-knee threshold for energy-based confidence
    float energyConfidence;
    float kneeStart = thresholdDb - kneeDb * 0.5f;
    float kneeEnd = thresholdDb + kneeDb * 0.5f;

    if (energyDb < kneeStart)
    {
        energyConfidence = 0.0f;
    }
    else if (energyDb > kneeEnd)
    {
        energyConfidence = 1.0f;
    }
    else
    {
        float kneePos = (energyDb - kneeStart) / kneeDb;
        energyConfidence = kneePos * kneePos;
    }

    // Combine energy, pitch, and formant detection:
    // - energyConfidence: is there enough signal?
    // - pitchConfidence_: is it pitched (vocal) vs unpitched (drums)?
    // - formantRatio_: is energy concentrated in vocal formant region?
    //
    // Final confidence = energy * (pitch * 0.6 + formant * 0.4)
    // This means even if there's energy, drums (unpitched, low formant) score low
    float vocalScore = pitchConfidence_ * 0.6f + formantRatio_ * 0.4f;
    float rawConfidence = energyConfidence * vocalScore;

    // Smooth the confidence output
    float confCoeff = (rawConfidence > smoothedConfidence_) ? confAttackCoeff_ : confReleaseCoeff_;
    smoothedConfidence_ = confCoeff * rawConfidence + (1.0f - confCoeff) * smoothedConfidence_;

    // Clamp
    smoothedConfidence_ = std::clamp(smoothedConfidence_, 0.0f, 1.0f);

    // Update for UI
    currentConfidence_.store(smoothedConfidence_);

    return smoothedConfidence_;
}

void SpectralVAD::processBlock(const float* input, float* confidence, int numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        confidence[i] = processSample(input[i]);
    }
}

void SpectralVAD::setBandWeights(const std::array<float, NUM_BANDS>& weights)
{
    // Normalize weights to sum to 1
    float sum = 0.0f;
    for (float w : weights)
        sum += std::abs(w);

    if (sum > 1e-6f)
    {
        for (int i = 0; i < NUM_BANDS; ++i)
            bandWeights_[i] = std::abs(weights[i]) / sum;
    }
}

void SpectralVAD::setBandFrequencies(const std::array<float, NUM_BANDS>& frequencies)
{
    bandFrequencies_ = frequencies;
    updateCoefficients();
}

void SpectralVAD::setThresholdDb(float thresholdDb)
{
    thresholdDb_.store(std::clamp(thresholdDb, -80.0f, 0.0f));
}

void SpectralVAD::setKneeDb(float kneeDb)
{
    kneeDb_.store(std::clamp(kneeDb, 1.0f, 30.0f));
}

void SpectralVAD::setAttackMs(float attackMs)
{
    attackMs_.store(std::clamp(attackMs, 0.1f, 100.0f));
    updateCoefficients();
}

void SpectralVAD::setReleaseMs(float releaseMs)
{
    releaseMs_.store(std::clamp(releaseMs, 1.0f, 500.0f));
    updateCoefficients();
}

std::array<float, SpectralVAD::NUM_BANDS> SpectralVAD::getBandEnergies() const
{
    std::array<float, NUM_BANDS> energies;
    for (int i = 0; i < NUM_BANDS; ++i)
        energies[i] = bandEnergies_[i].load();
    return energies;
}
