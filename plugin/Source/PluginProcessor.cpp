#include "PluginProcessor.h"
#include "PluginEditor.h"
#include <cmath>
#include <algorithm>

// Parameter IDs
const juce::String DeBleedAudioProcessor::PARAM_STRENGTH = "strength";
const juce::String DeBleedAudioProcessor::PARAM_MIX = "mix";
const juce::String DeBleedAudioProcessor::PARAM_BYPASS = "bypass";
const juce::String DeBleedAudioProcessor::PARAM_LOW_LATENCY = "lowLatency";
const juce::String DeBleedAudioProcessor::PARAM_ATTACK = "attack";
const juce::String DeBleedAudioProcessor::PARAM_RELEASE = "release";
const juce::String DeBleedAudioProcessor::PARAM_THRESHOLD = "threshold";
const juce::String DeBleedAudioProcessor::PARAM_FLOOR = "floor";
const juce::String DeBleedAudioProcessor::PARAM_LIVE_MODE = "liveMode";
const juce::String DeBleedAudioProcessor::PARAM_HPF_BOUND = "hpfBound";
const juce::String DeBleedAudioProcessor::PARAM_LPF_BOUND = "lpfBound";
const juce::String DeBleedAudioProcessor::PARAM_TIGHTNESS = "tightness";

// Linkwitz-Riley Gate parameters
const juce::String DeBleedAudioProcessor::PARAM_LR_ENABLED = "lrEnabled";

// Per-band gate parameter IDs
const std::array<juce::String, 6> DeBleedAudioProcessor::PARAM_GATE_THRESHOLD = {{
    "gateBand0Threshold", "gateBand1Threshold", "gateBand2Threshold",
    "gateBand3Threshold", "gateBand4Threshold", "gateBand5Threshold"
}};
const std::array<juce::String, 6> DeBleedAudioProcessor::PARAM_GATE_ATTACK = {{
    "gateBand0Attack", "gateBand1Attack", "gateBand2Attack",
    "gateBand3Attack", "gateBand4Attack", "gateBand5Attack"
}};
const std::array<juce::String, 6> DeBleedAudioProcessor::PARAM_GATE_RELEASE = {{
    "gateBand0Release", "gateBand1Release", "gateBand2Release",
    "gateBand3Release", "gateBand4Release", "gateBand5Release"
}};
const std::array<juce::String, 6> DeBleedAudioProcessor::PARAM_GATE_HOLD = {{
    "gateBand0Hold", "gateBand1Hold", "gateBand2Hold",
    "gateBand3Hold", "gateBand4Hold", "gateBand5Hold"
}};
const std::array<juce::String, 6> DeBleedAudioProcessor::PARAM_GATE_RANGE = {{
    "gateBand0Range", "gateBand1Range", "gateBand2Range",
    "gateBand3Range", "gateBand4Range", "gateBand5Range"
}};
const std::array<juce::String, 6> DeBleedAudioProcessor::PARAM_GATE_ENABLED = {{
    "gateBand0Enabled", "gateBand1Enabled", "gateBand2Enabled",
    "gateBand3Enabled", "gateBand4Enabled", "gateBand5Enabled"
}};

// Crossover frequency parameter IDs
const std::array<juce::String, 5> DeBleedAudioProcessor::PARAM_GATE_CROSSOVER = {{
    "gateCrossover0", "gateCrossover1", "gateCrossover2", "gateCrossover3", "gateCrossover4"
}};

DeBleedAudioProcessor::DeBleedAudioProcessor()
    : AudioProcessor(BusesProperties()
                     .withInput("Input", juce::AudioChannelSet::stereo(), true)
                     .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      parameters(*this, nullptr, juce::Identifier("DeBleedParams"), createParameterLayout())
{
    // Add parameter listeners
    parameters.addParameterListener(PARAM_STRENGTH, this);
    parameters.addParameterListener(PARAM_MIX, this);
    parameters.addParameterListener(PARAM_BYPASS, this);
    parameters.addParameterListener(PARAM_LOW_LATENCY, this);
    parameters.addParameterListener(PARAM_ATTACK, this);
    parameters.addParameterListener(PARAM_RELEASE, this);
    parameters.addParameterListener(PARAM_THRESHOLD, this);
    parameters.addParameterListener(PARAM_FLOOR, this);
    parameters.addParameterListener(PARAM_HPF_BOUND, this);
    parameters.addParameterListener(PARAM_LPF_BOUND, this);
    parameters.addParameterListener(PARAM_TIGHTNESS, this);
    parameters.addParameterListener(PARAM_LR_ENABLED, this);

    // Per-band gate parameters
    for (int b = 0; b < 6; ++b)
    {
        parameters.addParameterListener(PARAM_GATE_THRESHOLD[b], this);
        parameters.addParameterListener(PARAM_GATE_ATTACK[b], this);
        parameters.addParameterListener(PARAM_GATE_RELEASE[b], this);
        parameters.addParameterListener(PARAM_GATE_HOLD[b], this);
        parameters.addParameterListener(PARAM_GATE_RANGE[b], this);
        parameters.addParameterListener(PARAM_GATE_ENABLED[b], this);
    }

    // Crossover parameters
    for (int x = 0; x < 5; ++x)
    {
        parameters.addParameterListener(PARAM_GATE_CROSSOVER[x], this);
    }

    // Initialize atomic values
    strength.store(*parameters.getRawParameterValue(PARAM_STRENGTH));
    mix.store(*parameters.getRawParameterValue(PARAM_MIX));
    bypassed.store(*parameters.getRawParameterValue(PARAM_BYPASS) > 0.5f);
    lowLatency.store(*parameters.getRawParameterValue(PARAM_LOW_LATENCY) > 0.5f);
    attackMs.store(*parameters.getRawParameterValue(PARAM_ATTACK));
    releaseMs.store(*parameters.getRawParameterValue(PARAM_RELEASE));
    threshold.store(*parameters.getRawParameterValue(PARAM_THRESHOLD));
    floorDb.store(*parameters.getRawParameterValue(PARAM_FLOOR));
    hpfBound.store(*parameters.getRawParameterValue(PARAM_HPF_BOUND));
    lpfBound.store(*parameters.getRawParameterValue(PARAM_LPF_BOUND));
    tightness.store(*parameters.getRawParameterValue(PARAM_TIGHTNESS));
    lrEnabled.store(*parameters.getRawParameterValue(PARAM_LR_ENABLED) > 0.5f);

    // Initialize per-band gate parameters
    for (int b = 0; b < 6; ++b)
    {
        gateBandParams[b].threshold.store(*parameters.getRawParameterValue(PARAM_GATE_THRESHOLD[b]));
        gateBandParams[b].attack.store(*parameters.getRawParameterValue(PARAM_GATE_ATTACK[b]));
        gateBandParams[b].release.store(*parameters.getRawParameterValue(PARAM_GATE_RELEASE[b]));
        gateBandParams[b].hold.store(*parameters.getRawParameterValue(PARAM_GATE_HOLD[b]));
        gateBandParams[b].range.store(*parameters.getRawParameterValue(PARAM_GATE_RANGE[b]));
        gateBandParams[b].enabled.store(*parameters.getRawParameterValue(PARAM_GATE_ENABLED[b]) > 0.5f);
    }

    // Initialize crossover frequencies
    for (int x = 0; x < 5; ++x)
    {
        gateCrossovers[x].store(*parameters.getRawParameterValue(PARAM_GATE_CROSSOVER[x]));
    }
}

DeBleedAudioProcessor::~DeBleedAudioProcessor()
{
    parameters.removeParameterListener(PARAM_STRENGTH, this);
    parameters.removeParameterListener(PARAM_MIX, this);
    parameters.removeParameterListener(PARAM_BYPASS, this);
    parameters.removeParameterListener(PARAM_LOW_LATENCY, this);
    parameters.removeParameterListener(PARAM_ATTACK, this);
    parameters.removeParameterListener(PARAM_RELEASE, this);
    parameters.removeParameterListener(PARAM_THRESHOLD, this);
    parameters.removeParameterListener(PARAM_FLOOR, this);
    parameters.removeParameterListener(PARAM_HPF_BOUND, this);
    parameters.removeParameterListener(PARAM_LPF_BOUND, this);
    parameters.removeParameterListener(PARAM_TIGHTNESS, this);
    parameters.removeParameterListener(PARAM_LR_ENABLED, this);

    // Per-band gate parameters
    for (int b = 0; b < 6; ++b)
    {
        parameters.removeParameterListener(PARAM_GATE_THRESHOLD[b], this);
        parameters.removeParameterListener(PARAM_GATE_ATTACK[b], this);
        parameters.removeParameterListener(PARAM_GATE_RELEASE[b], this);
        parameters.removeParameterListener(PARAM_GATE_HOLD[b], this);
        parameters.removeParameterListener(PARAM_GATE_RANGE[b], this);
        parameters.removeParameterListener(PARAM_GATE_ENABLED[b], this);
    }

    // Crossover parameters
    for (int x = 0; x < 5; ++x)
    {
        parameters.removeParameterListener(PARAM_GATE_CROSSOVER[x], this);
    }
}

juce::AudioProcessorValueTreeState::ParameterLayout DeBleedAudioProcessor::createParameterLayout()
{
    std::vector<std::unique_ptr<juce::RangedAudioParameter>> params;

    // Strength: How aggressively to apply the mask (0.0 = pass-through, 1.0 = full)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_STRENGTH, 1},
        "Strength",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        1.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value * 100.0f, 0) + "%"; },
        nullptr
    ));

    // Mix: Dry/Wet blend
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_MIX, 1},
        "Mix",
        juce::NormalisableRange<float>(0.0f, 1.0f, 0.01f),
        1.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value * 100.0f, 0) + "%"; },
        nullptr
    ));

    // Bypass
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_BYPASS, 1},
        "Bypass",
        false
    ));

    // Low Latency Mode
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_LOW_LATENCY, 1},
        "Low Latency",
        false
    ));

    // Attack time (ms) - how fast cuts engage
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_ATTACK, 1},
        "Attack",
        juce::NormalisableRange<float>(0.1f, 500.0f, 0.1f, 0.4f),  // Extended to 500ms
        10.0f,
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " ms"; },
        nullptr
    ));

    // Release time (ms) - how fast cuts release back to unity
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_RELEASE, 1},
        "Release",
        juce::NormalisableRange<float>(0.1f, 2000.0f, 0.1f, 0.4f),  // Extended down to 0.1ms
        500.0f,  // Slow release for smooth behavior
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 0) + " ms"; },
        nullptr
    ));

    // Threshold - input magnitude (dB) below this keeps gate closed
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_THRESHOLD, 1},
        "Threshold",
        juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),
        -80.0f,  // Default: effectively disabled (very low)
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; },
        nullptr
    ));

    // Range - depth of attenuation when gated (mask=0 gives this level)
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_FLOOR, 1},
        "Range",
        juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),
        -80.0f,  // Default: full attenuation when gated
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 1) + " dB"; },
        nullptr
    ));

    // Live Mode - prevents accidental training during live shows (UI-only, no audio effect)
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_LIVE_MODE, 1},
        "Live Mode",
        false  // Default: off (training enabled)
    ));

    // HPF Bound - minimum frequency for hunter filters
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_HPF_BOUND, 1},
        "HPF Bound",
        juce::NormalisableRange<float>(20.0f, 500.0f, 1.0f, 0.5f),
        20.0f,  // Default: 20Hz (full range)
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 0) + " Hz"; },
        nullptr
    ));

    // LPF Bound - maximum frequency for hunter filters
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_LPF_BOUND, 1},
        "LPF Bound",
        juce::NormalisableRange<float>(1000.0f, 20000.0f, 10.0f, 0.5f),
        20000.0f,  // Default: 20kHz (full range)
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value / 1000.0f, 1) + " kHz"; },
        nullptr
    ));

    // Tightness - minimum time before hunter can change frequency
    params.push_back(std::make_unique<juce::AudioParameterFloat>(
        juce::ParameterID{PARAM_TIGHTNESS, 1},
        "Tightness",
        juce::NormalisableRange<float>(0.0f, 500.0f, 1.0f, 0.5f),
        50.0f,  // Default: 50ms
        juce::String(),
        juce::AudioProcessorParameter::genericParameter,
        [](float value, int) { return juce::String(value, 0) + " ms"; },
        nullptr
    ));

    // === Linkwitz-Riley 6-Band Gate Parameters ===

    // Master gate enable
    params.push_back(std::make_unique<juce::AudioParameterBool>(
        juce::ParameterID{PARAM_LR_ENABLED, 1},
        "Gate Enable",
        true  // Default: enabled
    ));

    // Band names for display
    static const std::array<juce::String, 6> bandNames = {{"Sub", "Low", "L-Mid", "Mid", "H-Mid", "High"}};

    // Default values per band (sub to high: slower to faster timing)
    static const std::array<float, 6> defaultThreshold = {{-40.0f, -40.0f, -40.0f, -40.0f, -40.0f, -40.0f}};
    static const std::array<float, 6> defaultAttack = {{35.0f, 20.0f, 12.0f, 6.0f, 3.0f, 1.5f}};
    static const std::array<float, 6> defaultRelease = {{350.0f, 225.0f, 150.0f, 100.0f, 65.0f, 50.0f}};
    static const std::array<float, 6> defaultHold = {{75.0f, 55.0f, 35.0f, 20.0f, 12.0f, 8.0f}};
    static const std::array<float, 6> defaultRange = {{-60.0f, -60.0f, -60.0f, -60.0f, -60.0f, -60.0f}};

    // Per-band parameters (36 total)
    for (int b = 0; b < 6; ++b)
    {
        // Threshold (-80 to 0 dB)
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{PARAM_GATE_THRESHOLD[b], 1},
            bandNames[b] + " Threshold",
            juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),
            defaultThreshold[b],
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int) { return juce::String(value, 1) + " dB"; },
            nullptr
        ));

        // Attack (0.1 to 100 ms)
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{PARAM_GATE_ATTACK[b], 1},
            bandNames[b] + " Attack",
            juce::NormalisableRange<float>(0.1f, 100.0f, 0.1f, 0.4f),
            defaultAttack[b],
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int) { return juce::String(value, 1) + " ms"; },
            nullptr
        ));

        // Release (10 to 1000 ms)
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{PARAM_GATE_RELEASE[b], 1},
            bandNames[b] + " Release",
            juce::NormalisableRange<float>(10.0f, 1000.0f, 1.0f, 0.4f),
            defaultRelease[b],
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int) { return juce::String(value, 0) + " ms"; },
            nullptr
        ));

        // Hold (0 to 500 ms)
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{PARAM_GATE_HOLD[b], 1},
            bandNames[b] + " Hold",
            juce::NormalisableRange<float>(0.0f, 500.0f, 1.0f, 0.5f),
            defaultHold[b],
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int) { return juce::String(value, 0) + " ms"; },
            nullptr
        ));

        // Range (-80 to 0 dB)
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{PARAM_GATE_RANGE[b], 1},
            bandNames[b] + " Range",
            juce::NormalisableRange<float>(-80.0f, 0.0f, 0.1f),
            defaultRange[b],
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int) { return juce::String(value, 1) + " dB"; },
            nullptr
        ));

        // Band enable
        params.push_back(std::make_unique<juce::AudioParameterBool>(
            juce::ParameterID{PARAM_GATE_ENABLED[b], 1},
            bandNames[b] + " Enable",
            true
        ));
    }

    // Crossover frequencies (5 crossovers)
    // Ranges overlap to allow adjustment, but we enforce 1-octave minimum spacing in parameterChanged
    static const std::array<float, 5> crossoverDefaults = {{80.0f, 250.0f, 800.0f, 2500.0f, 8000.0f}};
    static const std::array<std::pair<float, float>, 5> crossoverRanges = {{
        {40.0f, 160.0f},      // X0: Sub/Low boundary
        {100.0f, 500.0f},     // X1: Low/L-Mid boundary
        {300.0f, 1500.0f},    // X2: L-Mid/Mid boundary
        {1000.0f, 5000.0f},   // X3: Mid/H-Mid boundary
        {3000.0f, 12000.0f}   // X4: H-Mid/High boundary
    }};

    for (int x = 0; x < 5; ++x)
    {
        params.push_back(std::make_unique<juce::AudioParameterFloat>(
            juce::ParameterID{PARAM_GATE_CROSSOVER[x], 1},
            "Crossover " + juce::String(x + 1),
            juce::NormalisableRange<float>(crossoverRanges[x].first, crossoverRanges[x].second, 1.0f, 0.5f),
            crossoverDefaults[x],
            juce::String(),
            juce::AudioProcessorParameter::genericParameter,
            [](float value, int) {
                if (value >= 1000.0f)
                    return juce::String(value / 1000.0f, 1) + " kHz";
                return juce::String(value, 0) + " Hz";
            },
            nullptr
        ));
    }

    return {params.begin(), params.end()};
}

void DeBleedAudioProcessor::parameterChanged(const juce::String& parameterID, float newValue)
{
    if (parameterID == PARAM_STRENGTH)
        strength.store(newValue);
    else if (parameterID == PARAM_MIX)
        mix.store(newValue);
    else if (parameterID == PARAM_BYPASS)
        bypassed.store(newValue > 0.5f);
    else if (parameterID == PARAM_LOW_LATENCY)
    {
        bool newLowLatency = newValue > 0.5f;
        if (lowLatency.load() != newLowLatency)
        {
            lowLatency.store(newLowLatency);
            // Mark that we need to reinitialize STFT processor
            needsReinit.store(true);
        }
    }
    else if (parameterID == PARAM_ATTACK)
        attackMs.store(newValue);
    else if (parameterID == PARAM_RELEASE)
        releaseMs.store(newValue);
    else if (parameterID == PARAM_THRESHOLD)
        threshold.store(newValue);
    else if (parameterID == PARAM_FLOOR)
        floorDb.store(newValue);
    else if (parameterID == PARAM_HPF_BOUND)
        hpfBound.store(newValue);
    else if (parameterID == PARAM_LPF_BOUND)
        lpfBound.store(newValue);
    else if (parameterID == PARAM_TIGHTNESS)
        tightness.store(newValue);
    else if (parameterID == PARAM_LR_ENABLED)
        lrEnabled.store(newValue > 0.5f);
    else
    {
        // Check per-band gate parameters
        for (int b = 0; b < 6; ++b)
        {
            if (parameterID == PARAM_GATE_THRESHOLD[b])
            {
                gateBandParams[b].threshold.store(newValue);
                return;
            }
            if (parameterID == PARAM_GATE_ATTACK[b])
            {
                gateBandParams[b].attack.store(newValue);
                return;
            }
            if (parameterID == PARAM_GATE_RELEASE[b])
            {
                gateBandParams[b].release.store(newValue);
                return;
            }
            if (parameterID == PARAM_GATE_HOLD[b])
            {
                gateBandParams[b].hold.store(newValue);
                return;
            }
            if (parameterID == PARAM_GATE_RANGE[b])
            {
                gateBandParams[b].range.store(newValue);
                return;
            }
            if (parameterID == PARAM_GATE_ENABLED[b])
            {
                gateBandParams[b].enabled.store(newValue > 0.5f);
                return;
            }
        }

        // Check crossover parameters
        for (int x = 0; x < 5; ++x)
        {
            if (parameterID == PARAM_GATE_CROSSOVER[x])
            {
                // Enforce minimum 1-octave spacing between crossovers
                float minAllowed = (x > 0) ? gateCrossovers[x - 1].load() * 2.0f : 40.0f;
                float maxAllowed = (x < 4) ? gateCrossovers[x + 1].load() / 2.0f : 12000.0f;
                float constrainedValue = juce::jlimit(minAllowed, maxAllowed, newValue);
                gateCrossovers[x].store(constrainedValue);
                return;
            }
        }
    }
}

int DeBleedAudioProcessor::freqToBin(float freqHz) const
{
    // Convert frequency to FFT bin index
    float binFloat = freqHz * stftProcessor.getFFTSize() / static_cast<float>(TARGET_SAMPLE_RATE);
    return std::clamp(static_cast<int>(binFloat), 0, STFTProcessor::N_FREQ_BINS - 1);
}

void DeBleedAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock)
{
    currentSampleRate = sampleRate;
    currentBlockSize = samplesPerBlock;

    // Check if we need to resample (for sidechain analysis)
    needsResampling = (std::abs(sampleRate - TARGET_SAMPLE_RATE) > 1.0);
    resampleRatio = TARGET_SAMPLE_RATE / sampleRate;

    // Calculate resampled block size for sidechain
    int resampledBlockSize = needsResampling
        ? static_cast<int>(std::ceil(samplesPerBlock * resampleRatio)) + 16
        : samplesPerBlock;

    // === NEW: Dynamic Hunter Filter Pool (Zero-Latency Audio Path) ===
    activeFilterPool.prepare(sampleRate, samplesPerBlock);

    // === NEW: 6-band Linkwitz-Riley Gate (Broadband Macro Gating) ===
    linkwitzGate.prepare(sampleRate, samplesPerBlock);

    // === Sidechain Analyzer (Control Path) ===
    // Note: We still use 192-band analysis internally, but filter pool uses raw 129-bin mask
    std::array<float, 192> dummyFreqs;  // Sidechain analyzer expects this but we don't use it anymore
    for (int i = 0; i < 192; ++i)
        dummyFreqs[i] = 20.0f * std::pow(1000.0f, static_cast<float>(i) / 191.0f);
    sidechainAnalyzer.prepare(TARGET_SAMPLE_RATE, resampledBlockSize, dummyFreqs);

    // Allocate sidechain buffer
    sidechainBuffer.resize(resampledBlockSize, 0.0f);

    // === LEGACY: Keep STFT/Neural for visualization and fallback ===
    stftProcessor.setMode(lowLatency.load() ? STFTProcessor::Mode::LowLatency : STFTProcessor::Mode::Quality);
    stftProcessor.prepare(TARGET_SAMPLE_RATE, resampledBlockSize);

    int hopLength = stftProcessor.getHopLength();
    int maxFrames = (resampledBlockSize / hopLength) + 8;
    neuralEngine.prepare(maxFrames);

    // Allocate buffers (kept for visualization)
    processBuffer.resize(resampledBlockSize * 2, 0.0f);
    maskBuffer.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);
    transposedMagnitude.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 0.0f);
    transposedMask.resize(STFTProcessor::N_FREQ_BINS * maxFrames, 1.0f);

    if (smoothedMask.size() != STFTProcessor::N_FREQ_BINS)
    {
        smoothedMask.resize(STFTProcessor::N_FREQ_BINS, 1.0f);
    }

    // Allocate resampling buffers for sidechain
    if (needsResampling)
    {
        resampledInput.resize(resampledBlockSize + 64, 0.0f);
        resampledOutput.resize(samplesPerBlock + 64, 0.0f);

        inputHistory.resize(RESAMPLER_HISTORY_SIZE, 0.0f);
        outputHistory.resize(RESAMPLER_HISTORY_SIZE, 0.0f);

        inputResampler.reset();
        outputResampler.reset();
    }

    // === ZERO LATENCY ===
    // IIR filters are causal - no lookahead needed
    // Only report minimal latency for filter group delay (~1-2 samples)
    setLatencySamples(0);
}

void DeBleedAudioProcessor::releaseResources()
{
    activeFilterPool.reset();
    linkwitzGate.reset();
    sidechainAnalyzer.reset();
    stftProcessor.reset();
}

bool DeBleedAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const
{
    // Support mono and stereo
    if (layouts.getMainOutputChannelSet() != juce::AudioChannelSet::mono() &&
        layouts.getMainOutputChannelSet() != juce::AudioChannelSet::stereo())
        return false;

    // Input must match output
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;

    return true;
}

void DeBleedAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                          juce::MidiBuffer& /*midiMessages*/)
{
    juce::ScopedNoDenormals noDenormals;

    auto totalNumInputChannels = getTotalNumInputChannels();
    auto totalNumOutputChannels = getTotalNumOutputChannels();
    int numSamples = buffer.getNumSamples();

    // Clear any unused output channels
    for (auto i = totalNumInputChannels; i < totalNumOutputChannels; ++i)
        buffer.clear(i, 0, numSamples);

    // Early return if bypassed
    if (bypassed.load())
        return;

    // Check if model is loaded in sidechain analyzer
    if (!sidechainAnalyzer.isModelLoaded())
        return;  // Pass through (no processing)

    // Get parameter values
    float currentStrength = strength.load();
    float currentMix = mix.load();

    // Early bypass if mix is 0%
    if (currentMix < 0.001f)
        return;

    const float* inputData = buffer.getReadPointer(0);

    // Keep a copy of the dry signal for mix
    std::vector<float> drySignal(inputData, inputData + numSamples);

    // === SIDECHAIN ANALYSIS (Control Path) ===
    // Prepare data for sidechain (resample to 48kHz if needed)
    const float* analysisInput = inputData;
    int analysisNumSamples = numSamples;

    if (needsResampling)
    {
        analysisNumSamples = static_cast<int>(numSamples * resampleRatio);
        sidechainBuffer.resize(analysisNumSamples + 16);

        std::vector<float> inputWithHistory(RESAMPLER_HISTORY_SIZE + numSamples);
        std::memcpy(inputWithHistory.data(), inputHistory.data(), RESAMPLER_HISTORY_SIZE * sizeof(float));
        std::memcpy(inputWithHistory.data() + RESAMPLER_HISTORY_SIZE, inputData, numSamples * sizeof(float));

        inputResampler.process(1.0 / resampleRatio,
                               inputWithHistory.data() + RESAMPLER_HISTORY_SIZE,
                               sidechainBuffer.data(),
                               analysisNumSamples);

        int historyStart = numSamples - RESAMPLER_HISTORY_SIZE;
        if (historyStart >= 0)
            std::memcpy(inputHistory.data(), inputData + historyStart, RESAMPLER_HISTORY_SIZE * sizeof(float));

        analysisInput = sidechainBuffer.data();
    }

    // Update sidechain analyzer parameters
    sidechainAnalyzer.setStrength(currentStrength);
    sidechainAnalyzer.setAttack(attackMs.load());
    sidechainAnalyzer.setRelease(releaseMs.load());
    sidechainAnalyzer.setThreshold(threshold.load());
    sidechainAnalyzer.setFloor(floorDb.load());

    // Run sidechain analysis (STFT → Neural Net → Band Mapping → Envelopes)
    sidechainAnalyzer.analyze(analysisInput, analysisNumSamples);

    // Get raw neural mask (129 bins) for the hunter filter pool
    const float* rawMask = sidechainAnalyzer.getRawMask();

    // === AUDIO PATH: 6-Band Linkwitz-Riley Gate (Broadband Macro Gating) ===
    // Signal-level gate with per-band user-controlled parameters
    linkwitzGate.setEnabled(lrEnabled.load());

    // Update per-band gate parameters from atomic storage
    for (int b = 0; b < 6; ++b)
    {
        linkwitzGate.setBandThreshold(b, gateBandParams[b].threshold.load());
        linkwitzGate.setBandAttack(b, gateBandParams[b].attack.load());
        linkwitzGate.setBandRelease(b, gateBandParams[b].release.load());
        linkwitzGate.setBandHold(b, gateBandParams[b].hold.load());
        linkwitzGate.setBandRange(b, gateBandParams[b].range.load());
        linkwitzGate.setBandEnabled(b, gateBandParams[b].enabled.load());
    }

    // Update crossover frequencies
    for (int x = 0; x < 5; ++x)
    {
        linkwitzGate.setCrossover(x, gateCrossovers[x].load());
    }

    // Process gate (signal-level detection per band)
    linkwitzGate.process(buffer);

    // === AUDIO PATH: Dynamic Hunter Filters (Surgical Micro Cuts) ===
    // The ActiveFilterPool finds valleys in the mask and assigns 32 filters to chase them
    activeFilterPool.setStrength(currentStrength);
    activeFilterPool.setFloorDb(floorDb.load());
    activeFilterPool.setAttackMs(attackMs.load());
    activeFilterPool.setReleaseMs(releaseMs.load());
    activeFilterPool.setHpfBound(hpfBound.load());
    activeFilterPool.setLpfBound(lpfBound.load());
    activeFilterPool.setTightnessMs(tightness.load());
    activeFilterPool.process(buffer, rawMask);

    // === VISUALIZATION ===
    // Also run through legacy STFT for visualization data
    int numFrames = stftProcessor.processBlock(analysisInput, analysisNumSamples);
    if (numFrames > 0)
    {
        const float* magnitude = stftProcessor.getMagnitudeData();
        const float* rawMask = sidechainAnalyzer.getRawMask();

        if (rawMask != nullptr)
        {
            // rawMask is 257 bins: [0..128] = Stream A, [129..256] = Stream B
            const float* streamB = rawMask + VisualizationData::N_FREQ_BINS;  // Point to Stream B

            for (int frame = 0; frame < numFrames; ++frame)
            {
                visualizationData.pushFrame(
                    magnitude + frame * STFTProcessor::N_FREQ_BINS,
                    rawMask,   // Stream A: first 129 bins
                    streamB    // Stream B: next 128 bins (high-res lows)
                );
            }
        }
    }

    // Update gain reduction meter
    float avgReductionDb = sidechainAnalyzer.getAverageGainReduction();
    visualizationData.averageGainReductionDb.store(avgReductionDb);

    // === MIX: Dry/Wet blend ===
    float* outputData = buffer.getWritePointer(0);
    if (currentMix < 1.0f)
    {
        float wet = currentMix;
        float dry = 1.0f - currentMix;

        for (int i = 0; i < numSamples; ++i)
        {
            outputData[i] = outputData[i] * wet + drySignal[i] * dry;
        }
    }

    // === SAFETY: Final hard limit ===
    for (int i = 0; i < numSamples; ++i)
    {
        if (std::isnan(outputData[i]) || std::isinf(outputData[i]))
            outputData[i] = 0.0f;
        else
            outputData[i] = std::clamp(outputData[i], -1.0f, 1.0f);
    }

    // Copy to other channels if stereo
    for (int channel = 1; channel < totalNumOutputChannels; ++channel)
    {
        buffer.copyFrom(channel, 0, outputData, numSamples);
    }
}

bool DeBleedAudioProcessor::loadModel(const juce::String& modelPath)
{
    // Load model in both the legacy engine (for visualization) and sidechain analyzer
    bool legacyLoaded = neuralEngine.loadModel(modelPath);
    bool sidechainLoaded = sidechainAnalyzer.loadModel(modelPath);
    return legacyLoaded && sidechainLoaded;
}

void DeBleedAudioProcessor::unloadModel()
{
    neuralEngine.unloadModel();
    sidechainAnalyzer.unloadModel();
}

juce::AudioProcessorEditor* DeBleedAudioProcessor::createEditor()
{
    return new DeBleedAudioProcessorEditor(*this);
}

void DeBleedAudioProcessor::getStateInformation(juce::MemoryBlock& destData)
{
    auto state = parameters.copyState();

    // Add model path to state
    state.setProperty("modelPath", neuralEngine.getModelPath(), nullptr);

    std::unique_ptr<juce::XmlElement> xml(state.createXml());
    copyXmlToBinary(*xml, destData);
}

void DeBleedAudioProcessor::setStateInformation(const void* data, int sizeInBytes)
{
    std::unique_ptr<juce::XmlElement> xmlState(getXmlFromBinary(data, sizeInBytes));

    if (xmlState != nullptr)
    {
        if (xmlState->hasTagName(parameters.state.getType()))
        {
            parameters.replaceState(juce::ValueTree::fromXml(*xmlState));

            // Restore model
            juce::String modelPath = parameters.state.getProperty("modelPath", "").toString();
            if (modelPath.isNotEmpty())
            {
                loadModel(modelPath);
            }
        }
    }
}

std::array<ActiveFilterPool::FilterState, DeBleedAudioProcessor::NUM_HUNTERS> DeBleedAudioProcessor::getHunterStates() const
{
    return activeFilterPool.getFilterStates();
}

// Plugin instantiation
juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter()
{
    return new DeBleedAudioProcessor();
}
