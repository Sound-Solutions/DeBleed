#include "STFTProcessor.h"
#include <cmath>

STFTProcessor::STFTProcessor()
{
    // Will be properly initialized in prepare()
}

void STFTProcessor::createWindow()
{
    // Create Sine window for current FFT size (Stream A)
    // With 50% overlap, Sine * Sine sums to exactly 1.0 (perfect COLA)
    analysisWindow.resize(currentFFTSize);
    synthesisWindow.resize(currentFFTSize);

    for (int i = 0; i < currentFFTSize; ++i)
    {
        float w = std::sin(juce::MathConstants<float>::pi * i / currentFFTSize);
        analysisWindow[i] = w;
        synthesisWindow[i] = w;
    }

    // COLA normalization factor for Sine window with 50% overlap = 1.0
    colaSum = 1.0f;
}

void STFTProcessor::createWindowB()
{
    // Create Sine window for Stream B (2048-point FFT)
    analysisWindowB.resize(N_FFT_B);

    for (int i = 0; i < N_FFT_B; ++i)
    {
        analysisWindowB[i] = std::sin(juce::MathConstants<float>::pi * i / N_FFT_B);
    }
}

void STFTProcessor::prepare(double sampleRate, int maxBlockSize)
{
    currentSampleRate = sampleRate;

    // Set FFT parameters based on mode
    if (mode == Mode::LowLatency)
    {
        currentFFTSize = N_FFT_LOW_LATENCY;
        currentHopLength = HOP_LOW_LATENCY;
    }
    else
    {
        currentFFTSize = N_FFT_QUALITY;
        currentHopLength = HOP_QUALITY;
    }

    currentFreqBins = currentFFTSize / 2 + 1;

    // ==========================================================================
    // Stream A (Fast STFT)
    // ==========================================================================
    fftOrder = static_cast<int>(std::log2(currentFFTSize));
    fft = std::make_unique<juce::dsp::FFT>(fftOrder);

    // Create windows for Stream A
    createWindow();

    // Calculate maximum number of frames per block
    maxFrames = (maxBlockSize / currentHopLength) + 4;

    // Allocate Stream A buffers
    inputBuffer.resize(currentFFTSize, 0.0f);
    inputWritePos = 0;

    // Magnitude/phase buffers - always N_FREQ_BINS (129) for neural net compatibility
    magnitudeBuffer.resize(N_FREQ_BINS * maxFrames, 0.0f);
    phaseBuffer.resize(N_FREQ_BINS * maxFrames, 0.0f);
    numFramesReady = 0;

    // Output buffer - larger to handle circular wrap-around safely
    outputBuffer.resize(currentFFTSize * 8 + maxBlockSize, 0.0f);
    std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0f);
    outputReadPos = 0;
    // Start write position ahead by FFTSize - this provides the latency needed
    // for overlap-add to complete before samples are read
    outputWritePos = currentFFTSize;

    // FFT work buffer for Stream A
    fftWorkBuffer.resize(currentFFTSize * 2, 0.0f);

    // Interpolation buffers for low-latency mode
    if (mode == Mode::LowLatency)
    {
        interpMagBuffer.resize(N_FREQ_BINS, 0.0f);
        interpPhaseBuffer.resize(N_FREQ_BINS, 0.0f);
        decimatedMaskBuffer.resize(currentFreqBins, 0.0f);
    }

    // DEBUG: Buffer to store windowed frames for testing overlap-add
    windowedFrameBuffer.resize(currentFFTSize * maxFrames, 0.0f);

    // ==========================================================================
    // Stream B (Slow STFT for bass precision) - Dual Stream Mode
    // ==========================================================================
    if (dualStreamEnabled)
    {
        // Create Stream B FFT (2048-point)
        fftOrderB = static_cast<int>(std::log2(N_FFT_B));
        fftB = std::make_unique<juce::dsp::FFT>(fftOrderB);

        // Create window for Stream B
        createWindowB();

        // Stream B input buffer (larger for 2048-point FFT)
        inputBufferB.resize(N_FFT_B, 0.0f);
        inputWritePosB = 0;

        // Stream B output buffer (only low-frequency bins)
        streamBLowMagBuffer.resize(STREAM_B_BINS_USED * maxFrames, 0.0f);

        // FFT work buffer for Stream B
        fftWorkBufferB.resize(N_FFT_B * 2, 0.0f);

        // Concatenated dual-stream features buffer
        dualStreamFeaturesBuffer.resize(N_TOTAL_FEATURES * maxFrames, 0.0f);
    }
}

void STFTProcessor::reset()
{
    // Reset Stream A buffers
    std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
    inputWritePos = 0;

    std::fill(magnitudeBuffer.begin(), magnitudeBuffer.end(), 0.0f);
    std::fill(phaseBuffer.begin(), phaseBuffer.end(), 0.0f);
    numFramesReady = 0;

    std::fill(outputBuffer.begin(), outputBuffer.end(), 0.0f);
    outputReadPos = 0;
    outputWritePos = currentFFTSize;  // Match prepare() - start ahead for overlap-add latency

    // Reset Stream B buffers (dual-stream mode)
    if (dualStreamEnabled)
    {
        std::fill(inputBufferB.begin(), inputBufferB.end(), 0.0f);
        inputWritePosB = 0;
        std::fill(streamBLowMagBuffer.begin(), streamBLowMagBuffer.end(), 0.0f);
        std::fill(dualStreamFeaturesBuffer.begin(), dualStreamFeaturesBuffer.end(), 0.0f);
    }
}

int STFTProcessor::processBlock(const float* input, int numSamples)
{
    numFramesReady = 0;

    for (int i = 0; i < numSamples; ++i)
    {
        // ==========================================================================
        // Stream A processing (256-point FFT)
        // ==========================================================================
        inputBuffer[inputWritePos] = input[i];
        inputWritePos++;

        if (inputWritePos >= currentFFTSize)
        {
            computeSTFTFrame(inputBuffer.data());

            std::memmove(inputBuffer.data(),
                        inputBuffer.data() + currentHopLength,
                        (currentFFTSize - currentHopLength) * sizeof(float));
            inputWritePos = currentFFTSize - currentHopLength;
        }

        // ==========================================================================
        // Stream B processing (2048-point FFT) - Dual Stream Mode
        // ==========================================================================
        if (dualStreamEnabled)
        {
            inputBufferB[inputWritePosB] = input[i];
            inputWritePosB++;

            // Stream B uses larger FFT but same hop size for frame alignment
            if (inputWritePosB >= N_FFT_B)
            {
                computeStreamBFrame(inputBufferB.data());

                std::memmove(inputBufferB.data(),
                            inputBufferB.data() + currentHopLength,
                            (N_FFT_B - currentHopLength) * sizeof(float));
                inputWritePosB = N_FFT_B - currentHopLength;
            }
        }
    }

    // After processing, concatenate dual-stream features
    if (dualStreamEnabled && numFramesReady > 0)
    {
        concatenateDualStreamFeatures();
    }

    return numFramesReady;
}

void STFTProcessor::interpolateBins(const float* input, int inputBins, float* output, int outputBins)
{
    // Linear interpolation from inputBins to outputBins
    for (int i = 0; i < outputBins; ++i)
    {
        float srcPos = static_cast<float>(i) * (inputBins - 1) / (outputBins - 1);
        int srcIdx = static_cast<int>(srcPos);
        float frac = srcPos - srcIdx;

        if (srcIdx >= inputBins - 1)
        {
            output[i] = input[inputBins - 1];
        }
        else
        {
            output[i] = input[srcIdx] * (1.0f - frac) + input[srcIdx + 1] * frac;
        }
    }
}

void STFTProcessor::decimateBins(const float* input, int inputBins, float* output, int outputBins)
{
    // Linear interpolation from inputBins to outputBins (downsampling)
    for (int i = 0; i < outputBins; ++i)
    {
        float srcPos = static_cast<float>(i) * (inputBins - 1) / (outputBins - 1);
        int srcIdx = static_cast<int>(srcPos);
        float frac = srcPos - srcIdx;

        if (srcIdx >= inputBins - 1)
        {
            output[i] = input[inputBins - 1];
        }
        else
        {
            output[i] = input[srcIdx] * (1.0f - frac) + input[srcIdx + 1] * frac;
        }
    }
}

void STFTProcessor::computeSTFTFrame(const float* frameData)
{
    if (numFramesReady >= maxFrames)
        return;

    // Apply analysis window
    for (int i = 0; i < currentFFTSize; ++i)
    {
        fftWorkBuffer[i] = frameData[i] * analysisWindow[i];
    }

    // DEBUG: Store windowed frame for testing overlap-add without FFT
    float* windowedPtr = windowedFrameBuffer.data() + numFramesReady * currentFFTSize;
    std::memcpy(windowedPtr, fftWorkBuffer.data(), currentFFTSize * sizeof(float));

    for (int i = currentFFTSize; i < currentFFTSize * 2; ++i)
    {
        fftWorkBuffer[i] = 0.0f;
    }

    // Forward FFT
    fft->performRealOnlyForwardTransform(fftWorkBuffer.data());

    // Extract magnitude and phase to temporary buffers
    std::vector<float> tempMag(currentFreqBins);
    std::vector<float> tempPhase(currentFreqBins);

    // DC component
    tempMag[0] = std::abs(fftWorkBuffer[0]);
    tempPhase[0] = (fftWorkBuffer[0] >= 0) ? 0.0f : juce::MathConstants<float>::pi;

    // Nyquist
    tempMag[currentFreqBins - 1] = std::abs(fftWorkBuffer[1]);
    tempPhase[currentFreqBins - 1] = (fftWorkBuffer[1] >= 0) ? 0.0f : juce::MathConstants<float>::pi;

    // Other bins
    for (int bin = 1; bin < currentFreqBins - 1; ++bin)
    {
        float real = fftWorkBuffer[bin * 2];
        float imag = fftWorkBuffer[bin * 2 + 1];
        tempMag[bin] = std::sqrt(real * real + imag * imag);
        tempPhase[bin] = std::atan2(imag, real);
    }

    // Output pointers (always N_FREQ_BINS = 129)
    float* magPtr = magnitudeBuffer.data() + numFramesReady * N_FREQ_BINS;
    float* phasePtr = phaseBuffer.data() + numFramesReady * N_FREQ_BINS;

    // Interpolate if needed (low-latency mode: 65 bins -> 129 bins)
    if (currentFreqBins != N_FREQ_BINS)
    {
        interpolateBins(tempMag.data(), currentFreqBins, magPtr, N_FREQ_BINS);
        interpolateBins(tempPhase.data(), currentFreqBins, phasePtr, N_FREQ_BINS);
    }
    else
    {
        std::memcpy(magPtr, tempMag.data(), N_FREQ_BINS * sizeof(float));
        std::memcpy(phasePtr, tempPhase.data(), N_FREQ_BINS * sizeof(float));
    }

    numFramesReady++;
}

void STFTProcessor::applyMaskAndReconstruct(const float* mask, float* output, int numSamples)
{
    // Temporary buffer for ISTFT output
    std::vector<float> istftOutput(currentFFTSize);

    // Process each STFT frame: apply mask, IFFT, overlap-add
    for (int frame = 0; frame < numFramesReady; ++frame)
    {
        const float* magPtr = magnitudeBuffer.data() + frame * N_FREQ_BINS;
        const float* phasePtr = phaseBuffer.data() + frame * N_FREQ_BINS;

        // Get mask for this frame (or assume 1.0 if null)
        const float* maskPtr = (mask != nullptr) ? mask + frame * N_FREQ_BINS : nullptr;

        // Prepare magnitude and phase at correct resolution
        std::vector<float> reconMag(currentFreqBins);
        std::vector<float> reconPhase(currentFreqBins);

        if (currentFreqBins != N_FREQ_BINS)
        {
            // Low-latency mode: decimate from 129 bins to 65 bins
            decimateBins(magPtr, N_FREQ_BINS, reconMag.data(), currentFreqBins);
            decimateBins(phasePtr, N_FREQ_BINS, reconPhase.data(), currentFreqBins);
        }
        else
        {
            std::memcpy(reconMag.data(), magPtr, currentFreqBins * sizeof(float));
            std::memcpy(reconPhase.data(), phasePtr, currentFreqBins * sizeof(float));
        }

        // Apply mask to magnitude
        if (maskPtr != nullptr)
        {
            if (currentFreqBins != N_FREQ_BINS)
            {
                // Decimate mask too
                decimateBins(maskPtr, N_FREQ_BINS, decimatedMaskBuffer.data(), currentFreqBins);
                for (int bin = 0; bin < currentFreqBins; ++bin)
                    reconMag[bin] *= decimatedMaskBuffer[bin];
            }
            else
            {
                for (int bin = 0; bin < currentFreqBins; ++bin)
                    reconMag[bin] *= maskPtr[bin];
            }
        }
        // If mask is null, magnitude is unchanged (pass-through)

        // Compute ISTFT frame (synthesis window applied inside)
        computeISTFTFrame(reconMag.data(), reconPhase.data(), istftOutput.data());

        // Overlap-add to circular buffer
        for (int i = 0; i < currentFFTSize; ++i)
        {
            int outIdx = (outputWritePos + i) % outputBuffer.size();
            outputBuffer[outIdx] += istftOutput[i];
        }
        outputWritePos = (outputWritePos + currentHopLength) % outputBuffer.size();
    }

    // Read from circular buffer
    for (int i = 0; i < numSamples; ++i)
    {
        int readIdx = (outputReadPos + i) % outputBuffer.size();
        float sample = outputBuffer[readIdx];

        if (std::isnan(sample) || std::isinf(sample))
            sample = 0.0f;
        else
            sample = std::clamp(sample, -10.0f, 10.0f);

        output[i] = sample;
        outputBuffer[readIdx] = 0.0f;  // Clear after reading
    }

    outputReadPos = (outputReadPos + numSamples) % outputBuffer.size();
}

void STFTProcessor::readOutput(float* output, int numSamples)
{
    // Read from circular output buffer (for when no frames were processed)
    for (int i = 0; i < numSamples; ++i)
    {
        int readIdx = (outputReadPos + i) % outputBuffer.size();
        float sample = outputBuffer[readIdx];

        if (std::isnan(sample) || std::isinf(sample))
            sample = 0.0f;
        else
            sample = std::clamp(sample, -10.0f, 10.0f);

        output[i] = sample;
        outputBuffer[readIdx] = 0.0f;  // Clear after reading
    }

    outputReadPos = (outputReadPos + numSamples) % outputBuffer.size();
}

void STFTProcessor::computeISTFTFrame(const float* magnitude, const float* phase, float* output)
{
    // Clear work buffer
    std::fill(fftWorkBuffer.begin(), fftWorkBuffer.end(), 0.0f);

    // DC
    fftWorkBuffer[0] = magnitude[0] * std::cos(phase[0]);

    // Nyquist
    fftWorkBuffer[1] = magnitude[currentFreqBins - 1] * std::cos(phase[currentFreqBins - 1]);

    // Other bins
    for (int bin = 1; bin < currentFreqBins - 1; ++bin)
    {
        fftWorkBuffer[bin * 2] = magnitude[bin] * std::cos(phase[bin]);
        fftWorkBuffer[bin * 2 + 1] = magnitude[bin] * std::sin(phase[bin]);
    }

    fft->performRealOnlyInverseTransform(fftWorkBuffer.data());

    // Apply synthesis window (JUCE FFT handles normalization internally)
    for (int i = 0; i < currentFFTSize; ++i)
    {
        output[i] = fftWorkBuffer[i] * synthesisWindow[i];
    }
}

// =============================================================================
// Stream B (Dual-Stream) Processing Methods
// =============================================================================

void STFTProcessor::computeStreamBFrame(const float* frameData)
{
    if (numFramesReady >= maxFrames || !dualStreamEnabled || fftB == nullptr)
        return;

    // Apply analysis window for Stream B
    for (int i = 0; i < N_FFT_B; ++i)
    {
        fftWorkBufferB[i] = frameData[i] * analysisWindowB[i];
    }

    // Zero-pad the rest
    for (int i = N_FFT_B; i < N_FFT_B * 2; ++i)
    {
        fftWorkBufferB[i] = 0.0f;
    }

    // Forward FFT for Stream B
    fftB->performRealOnlyForwardTransform(fftWorkBufferB.data());

    // Extract only the low-frequency magnitude bins (0 to STREAM_B_BINS_USED)
    // We match the frame index from Stream A (numFramesReady - 1 because A already incremented)
    int frameIdx = numFramesReady - 1;
    if (frameIdx < 0) frameIdx = 0;

    float* magBPtr = streamBLowMagBuffer.data() + frameIdx * STREAM_B_BINS_USED;

    // DC component
    magBPtr[0] = std::abs(fftWorkBufferB[0]);

    // Extract bins 1 to STREAM_B_BINS_USED-1
    for (int bin = 1; bin < STREAM_B_BINS_USED; ++bin)
    {
        float real = fftWorkBufferB[bin * 2];
        float imag = fftWorkBufferB[bin * 2 + 1];
        magBPtr[bin] = std::sqrt(real * real + imag * imag);
    }
}

void STFTProcessor::concatenateDualStreamFeatures()
{
    if (!dualStreamEnabled)
        return;

    // For each frame, concatenate [Stream A (129 bins) | Stream B bass (128 bins)]
    for (int frame = 0; frame < numFramesReady; ++frame)
    {
        float* featPtr = dualStreamFeaturesBuffer.data() + frame * N_TOTAL_FEATURES;
        const float* magAPtr = magnitudeBuffer.data() + frame * N_FREQ_BINS_A;
        const float* magBPtr = streamBLowMagBuffer.data() + frame * STREAM_B_BINS_USED;

        // Copy Stream A (129 bins)
        std::memcpy(featPtr, magAPtr, N_FREQ_BINS_A * sizeof(float));

        // Copy Stream B bass (128 bins)
        std::memcpy(featPtr + N_FREQ_BINS_A, magBPtr, STREAM_B_BINS_USED * sizeof(float));
    }
}
