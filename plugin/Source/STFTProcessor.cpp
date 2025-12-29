#include "STFTProcessor.h"
#include <cmath>

STFTProcessor::STFTProcessor()
{
    // Calculate FFT order (log2 of N_FFT)
    fftOrder = static_cast<int>(std::log2(N_FFT));
    jassert((1 << fftOrder) == N_FFT);  // N_FFT must be power of 2

    fft = std::make_unique<juce::dsp::FFT>(fftOrder);

    createWindow();
}

void STFTProcessor::createWindow()
{
    // Create Hann window for analysis
    analysisWindow.resize(WIN_LENGTH);
    for (int i = 0; i < WIN_LENGTH; ++i)
    {
        analysisWindow[i] = 0.5f * (1.0f - std::cos(2.0f * juce::MathConstants<float>::pi * i / (WIN_LENGTH - 1)));
    }

    // Synthesis window is the same (will be normalized in overlap-add)
    synthesisWindow = analysisWindow;
}

void STFTProcessor::prepare(double sampleRate, int maxBlockSize)
{
    currentSampleRate = sampleRate;

    // Calculate maximum number of frames we might need
    // For a 3-second buffer at 48kHz with hop of 128: ~1125 frames
    maxFrames = static_cast<int>(std::ceil(3.0 * sampleRate / HOP_LENGTH)) + 16;

    // Allocate buffers
    inputBuffer.resize(N_FFT * 2, 0.0f);  // Double buffer for overlap
    inputWritePos = 0;

    magnitudeBuffer.resize(N_FREQ_BINS * maxFrames, 0.0f);
    phaseBuffer.resize(N_FREQ_BINS * maxFrames, 0.0f);
    numFramesReady = 0;

    // Overlap-add buffer (enough for several frames)
    overlapAddBuffer.resize(N_FFT + maxBlockSize * 4, 0.0f);
    overlapAddReadPos = 0;

    // FFT work buffers
    fftBuffer.resize(N_FFT);
    fftWorkBuffer.resize(N_FFT * 2);  // Real FFT needs 2x size
}

void STFTProcessor::reset()
{
    std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
    inputWritePos = 0;

    std::fill(magnitudeBuffer.begin(), magnitudeBuffer.end(), 0.0f);
    std::fill(phaseBuffer.begin(), phaseBuffer.end(), 0.0f);
    numFramesReady = 0;

    std::fill(overlapAddBuffer.begin(), overlapAddBuffer.end(), 0.0f);
    overlapAddReadPos = 0;
}

int STFTProcessor::processBlock(const float* input, int numSamples)
{
    numFramesReady = 0;

    for (int i = 0; i < numSamples; ++i)
    {
        // Add sample to input buffer
        inputBuffer[inputWritePos] = input[i];
        inputWritePos++;

        // Check if we have enough samples for a new frame
        if (inputWritePos >= N_FFT)
        {
            // Compute STFT for this frame
            computeSTFTFrame(inputBuffer.data());

            // Shift input buffer by hop length
            std::memmove(inputBuffer.data(),
                        inputBuffer.data() + HOP_LENGTH,
                        (N_FFT - HOP_LENGTH) * sizeof(float));
            inputWritePos = N_FFT - HOP_LENGTH;
        }
    }

    return numFramesReady;
}

void STFTProcessor::computeSTFTFrame(const float* frameData)
{
    if (numFramesReady >= maxFrames)
        return;  // Buffer full

    // Apply analysis window and prepare for FFT
    for (int i = 0; i < N_FFT; ++i)
    {
        fftWorkBuffer[i] = frameData[i] * analysisWindow[i];
    }

    // Zero pad the second half (for real FFT)
    std::fill(fftWorkBuffer.begin() + N_FFT, fftWorkBuffer.end(), 0.0f);

    // Perform forward FFT
    fft->performRealOnlyForwardTransform(fftWorkBuffer.data(), true);

    // Extract magnitude and phase
    // JUCE FFT output: [real0, imag0, real1, imag1, ...]
    // For real FFT: DC and Nyquist are real-only
    float* magPtr = magnitudeBuffer.data() + numFramesReady * N_FREQ_BINS;
    float* phasePtr = phaseBuffer.data() + numFramesReady * N_FREQ_BINS;

    for (int bin = 0; bin < N_FREQ_BINS; ++bin)
    {
        float real, imag;

        if (bin == 0)
        {
            // DC component
            real = fftWorkBuffer[0];
            imag = 0.0f;
        }
        else if (bin == N_FFT / 2)
        {
            // Nyquist component
            real = fftWorkBuffer[1];
            imag = 0.0f;
        }
        else
        {
            // Regular bins
            real = fftWorkBuffer[bin * 2];
            imag = fftWorkBuffer[bin * 2 + 1];
        }

        magPtr[bin] = std::sqrt(real * real + imag * imag);
        phasePtr[bin] = std::atan2(imag, real);
    }

    numFramesReady++;
}

void STFTProcessor::applyMaskAndReconstruct(const float* mask, float* output, int numSamples)
{
    // Clear output
    std::fill(output, output + numSamples, 0.0f);

    if (numFramesReady == 0)
        return;

    // Process each frame
    for (int frame = 0; frame < numFramesReady; ++frame)
    {
        const float* magPtr = magnitudeBuffer.data() + frame * N_FREQ_BINS;
        const float* phasePtr = phaseBuffer.data() + frame * N_FREQ_BINS;
        const float* maskPtr = mask + frame * N_FREQ_BINS;

        // Apply mask to magnitude
        std::vector<float> maskedMag(N_FREQ_BINS);
        for (int bin = 0; bin < N_FREQ_BINS; ++bin)
        {
            maskedMag[bin] = magPtr[bin] * maskPtr[bin];
        }

        // Reconstruct complex spectrum
        for (int bin = 0; bin < N_FREQ_BINS; ++bin)
        {
            float mag = maskedMag[bin];
            float phase = phasePtr[bin];
            float real = mag * std::cos(phase);
            float imag = mag * std::sin(phase);

            if (bin == 0)
            {
                fftWorkBuffer[0] = real;
            }
            else if (bin == N_FFT / 2)
            {
                fftWorkBuffer[1] = real;
            }
            else
            {
                fftWorkBuffer[bin * 2] = real;
                fftWorkBuffer[bin * 2 + 1] = imag;
            }
        }

        // Perform inverse FFT
        fft->performRealOnlyInverseTransform(fftWorkBuffer.data());

        // Apply synthesis window and overlap-add
        int frameStart = frame * HOP_LENGTH;
        for (int i = 0; i < N_FFT; ++i)
        {
            int outIdx = frameStart + i;
            if (outIdx < static_cast<int>(overlapAddBuffer.size()))
            {
                overlapAddBuffer[outIdx] += fftWorkBuffer[i] * synthesisWindow[i];
            }
        }
    }

    // Normalize and copy to output
    // Compute window normalization factor
    static std::vector<float> windowNorm;
    if (windowNorm.empty() || static_cast<int>(windowNorm.size()) < numSamples + N_FFT)
    {
        windowNorm.resize(numSamples + N_FFT * 2, 0.0f);
        std::fill(windowNorm.begin(), windowNorm.end(), 0.0f);

        int totalFrames = (numSamples + N_FFT) / HOP_LENGTH + 1;
        for (int frame = 0; frame < totalFrames; ++frame)
        {
            int frameStart = frame * HOP_LENGTH;
            for (int i = 0; i < N_FFT; ++i)
            {
                int idx = frameStart + i;
                if (idx < static_cast<int>(windowNorm.size()))
                {
                    float w = synthesisWindow[i];
                    windowNorm[idx] += w * w;
                }
            }
        }

        // Avoid division by zero
        for (auto& v : windowNorm)
        {
            if (v < 1e-8f) v = 1e-8f;
        }
    }

    // Copy normalized output
    for (int i = 0; i < numSamples; ++i)
    {
        int idx = overlapAddReadPos + i;
        if (idx < static_cast<int>(overlapAddBuffer.size()) && i < static_cast<int>(windowNorm.size()))
        {
            output[i] = overlapAddBuffer[idx] / windowNorm[i];
        }
    }

    // Shift overlap-add buffer
    int shift = numSamples;
    if (shift > 0 && shift < static_cast<int>(overlapAddBuffer.size()))
    {
        std::memmove(overlapAddBuffer.data(),
                    overlapAddBuffer.data() + shift,
                    (overlapAddBuffer.size() - shift) * sizeof(float));
        std::fill(overlapAddBuffer.end() - shift, overlapAddBuffer.end(), 0.0f);
    }
}

void STFTProcessor::computeISTFTFrame(const float* magnitude, const float* phase, float* output)
{
    // Reconstruct complex spectrum
    for (int bin = 0; bin < N_FREQ_BINS; ++bin)
    {
        float mag = magnitude[bin];
        float ph = phase[bin];
        float real = mag * std::cos(ph);
        float imag = mag * std::sin(ph);

        if (bin == 0)
        {
            fftWorkBuffer[0] = real;
        }
        else if (bin == N_FFT / 2)
        {
            fftWorkBuffer[1] = real;
        }
        else
        {
            fftWorkBuffer[bin * 2] = real;
            fftWorkBuffer[bin * 2 + 1] = imag;
        }
    }

    // Perform inverse FFT
    fft->performRealOnlyInverseTransform(fftWorkBuffer.data());

    // Apply synthesis window
    for (int i = 0; i < N_FFT; ++i)
    {
        output[i] = fftWorkBuffer[i] * synthesisWindow[i];
    }
}
