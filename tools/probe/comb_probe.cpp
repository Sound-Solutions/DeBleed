#include "../../plugin/Source/PitchTracker.h"
#include "../../plugin/Source/HarmonicComb.h"
#include "../../plugin/Source/LinkwitzRiley4.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace
{
constexpr double sampleRate = 96000.0;
const double pi = std::acos(-1.0);

double gainDb(double outputEnergy, double inputEnergy)
{
    return 10.0 * std::log10(outputEnergy / inputEnergy);
}

double measureComb(double frequency, float feedback)
{
    HarmonicComb comb;
    comb.prepare(sampleRate);
    comb.setForcedState(static_cast<float>(sampleRate / 220.0), feedback);
    double inputEnergy = 0.0, outputEnergy = 0.0;
    for (int n = 0; n < 192000; ++n)
    {
        const float input = static_cast<float>(std::sin(2.0 * pi * frequency * n / sampleRate));
        const float output = comb.process(input);
        if (n >= 96000)
        {
            inputEnergy += static_cast<double>(input) * input;
            outputEnergy += static_cast<double>(output) * output;
        }
    }
    return gainDb(outputEnergy, inputEnergy);
}

double measureCrossover(double frequency)
{
    LinkwitzRiley4 crossover;
    crossover.prepare(sampleRate, 4000.0);
    const int cycles = std::max(10, static_cast<int>(frequency * 0.15));
    const int measuredSamples = static_cast<int>(std::round(cycles * sampleRate / frequency));
    const int settleSamples = 24000;
    double inputEnergy = 0.0, outputEnergy = 0.0;
    for (int n = 0; n < settleSamples + measuredSamples; ++n)
    {
        const float input = static_cast<float>(std::sin(2.0 * pi * frequency * n / sampleRate));
        float low, high;
        crossover.processSplit(input, low, high);
        if (n >= settleSamples)
        {
            inputEnergy += static_cast<double>(input) * input;
            outputEnergy += static_cast<double>(low + high) * (low + high);
        }
    }
    // LR4 sums to an allpass: compare magnitude, not a sample-wise null.
    return gainDb(outputEnergy, inputEnergy);
}
}

int main()
{
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Sample rate: 96000 Hz\n";
    PitchTracker tracker;
    HarmonicComb trackedComb;
    tracker.prepare(sampleRate);
    trackedComb.prepare(sampleRate);
    trackedComb.setDepthDb(9.0f);

    double toneRmsSquared = 0.0;
    for (int h = 1; h <= 8; ++h)
        toneRmsSquared += 0.5 / (h * h);
    const double noiseScale = std::sqrt(3.0 * toneRmsSquared) * std::pow(10.0, -30.0 / 20.0);
    std::uint32_t noiseState = 0x12345678u;
    for (int n = 0; n < 192000; ++n)
    {
        double tone = 0.0;
        for (int h = 1; h <= 8; ++h)
            tone += std::sin(2.0 * pi * 220.0 * h * n / sampleRate) / h;
        noiseState = 1664525u * noiseState + 1013904223u;
        const double noise = 2.0 * (static_cast<double>(noiseState) / 4294967295.0) - 1.0;
        const float input = static_cast<float>(tone + noiseScale * noise);
        tracker.pushSample(input);
        if (n % 256 == 255)
            trackedComb.setTarget(tracker.getPeriodSamples(), tracker.getConfidence() >= 0.85f);
        trackedComb.setVocalConfidence(1.0f);
        trackedComb.process(input);
    }
    const double frequency = tracker.getFrequencyHz();
    const double confidence = tracker.getConfidence();
    int passed = 0, failed = 0;
    auto check = [&](bool ok) { if (ok) ++passed; else ++failed; return ok ? "PASS" : "FAIL"; };
    std::cout << "Tracked f0: " << frequency << " Hz (220 +/- 0.5): "
              << check(std::abs(frequency - 220.0) <= 0.5) << '\n';
    std::cout << "Confidence: " << confidence << " (> 0.9): " << check(confidence > 0.9) << '\n';

    const float r = std::pow(10.0f, -9.0f / 20.0f);
    const float gDepth = (1.0f - r) / (1.0f + r);
    std::cout << "Pinned comb: depth 9 dB, period " << sampleRate / 220.0
              << " samples, g " << gDepth << ", envelope forced to 1\n";
    for (double f : {220.0, 440.0, 660.0, 880.0})
    {
        const double gain = measureComb(f, gDepth);
        std::cout << "Harmonic " << f << " Hz: " << gain << " dB: "
                  << check(std::abs(gain) <= 0.2) << '\n';
    }
    for (double f : {330.0, 550.0, 770.0})
    {
        const double gain = measureComb(f, gDepth);
        std::cout << "Midpoint " << f << " Hz: " << gain << " dB: "
                  << check(std::abs(gain + 9.0) <= 0.5) << '\n';
    }

    double maxError = 0.0, worstFrequency = 0.0;
    for (int step = 0; step < 65; ++step)
    {
        const double f = 20.0 * std::pow(1000.0, step / 64.0);
        const double error = std::abs(measureCrossover(f));
        if (error > maxError)
        {
            maxError = error;
            worstFrequency = f;
        }
    }
    maxError = std::max(maxError, std::abs(measureCrossover(4000.0)));
    std::cout << "LR4 sum magnitude error (66-point stepped sine sweep, 20-20000 Hz): "
              << maxError << " dB, worst log-sweep frequency " << worstFrequency << " Hz: "
              << check(maxError <= 0.05) << '\n';

    // Feed real silence to the tracker. An immediate VAD=0 is the fastest possible
    // release; a real smoothed VAD can only take longer. Do not force the comb dry.
    std::cout << "Comb g before silence: " << trackedComb.getFeedbackGain() << '\n';
    float atTenMs = 0.0f;
    int belowThresholdAt = 0;
    for (int n = 1; n <= 9600; ++n)
    {
        tracker.pushSample(0.0f);
        if (n % 256 == 0)
            trackedComb.setTarget(tracker.getPeriodSamples(), tracker.getConfidence() >= 0.85f);
        trackedComb.setVocalConfidence(0.0f);
        trackedComb.process(0.0f);
        if (n == 960)
            atTenMs = trackedComb.getFeedbackGain();
        if (belowThresholdAt == 0 && trackedComb.getFeedbackGain() < 0.01f)
            belowThresholdAt = n;
    }
    // A 5 ms one-pole from full 9 dB depth (g = 0.476) reaches 0.01 after 5 * ln(47.6) = 19.3 ms.
    const double fadeMs = 1000.0 * belowThresholdAt / sampleRate;
    std::cout << "Comb g after 10 ms silence (immediate VAD off, 5 ms pole): " << atTenMs << '\n';
    std::cout << "Comb g first below 0.01 after: " << fadeMs << " ms (< 25 ms): "
              << check(belowThresholdAt > 0 && fadeMs < 25.0) << '\n';

    // Processor formula: round(sampleRate * 0.001), reported as 0 when disabled.
    for (double fs : {44100.0, 48000.0, 96000.0})
        std::cout << "Lookahead latency at " << fs << " Hz: off 0, on "
                  << static_cast<int>(std::round(fs * 0.001)) << " samples (formula only)\n";
    std::cout << "Checks: " << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}
