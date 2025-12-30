#include "RTAVisualization.h"
#include "PluginProcessor.h"

RTAVisualization::RTAVisualization(DeBleedAudioProcessor& processor)
    : audioProcessor(processor)
{
    // Initialize display buffers to low values
    displayMagnitudeDb.fill(-100.0f);
    displayMask.fill(1.0f);
    displayMaskB.fill(1.0f);
}

void RTAVisualization::updateFromQueue()
{
    // Consume all available frames from FIFO, use the latest
    DeBleedAudioProcessor::VisualizationData::FrameData frame;
    bool hasNewData = false;

    while (audioProcessor.getVisualizationData().popFrame(frame))
    {
        hasNewData = true;
    }

    if (!hasNewData)
        return;

    // Apply smoothing with frequency-dependent decay
    for (int i = 0; i < N_FREQ_BINS; ++i)
    {
        // Convert magnitude to dB
        float targetDb = 20.0f * std::log10f(std::max(frame.magnitude[i], 1e-10f));

        // Adaptive decay based on frequency (bass decays slower)
        float normIndex = static_cast<float>(i) / static_cast<float>(N_FREQ_BINS);
        float curve = std::sqrt(normIndex);  // Curve adjustment
        float decay = minDecayRate + (maxDecayRate - minDecayRate) * curve;

        // Instant attack, smooth decay
        if (targetDb > displayMagnitudeDb[i])
            displayMagnitudeDb[i] = targetDb;
        else
            displayMagnitudeDb[i] = std::max(displayMagnitudeDb[i] - decay, targetDb);

        // Clamp to display range
        displayMagnitudeDb[i] = std::clamp(displayMagnitudeDb[i], MIN_DB - 40.0f, MAX_DB);

        // Smooth mask values (Stream A)
        displayMask[i] = displayMask[i] * (1.0f - maskSmoothCoeff) + frame.mask[i] * maskSmoothCoeff;
    }

    // Smooth Stream B mask (high-resolution lows)
    for (int i = 0; i < STREAM_B_BINS; ++i)
    {
        displayMaskB[i] = displayMaskB[i] * (1.0f - maskSmoothCoeff) + frame.maskB[i] * maskSmoothCoeff;
    }

    repaint();
}

float RTAVisualization::binToFreq(int bin) const
{
    // FFT bin to frequency: bin * sampleRate / fftSize
    // DeBleed uses 256-point FFT at 48kHz
    return static_cast<float>(bin) * 48000.0f / 256.0f;
}

float RTAVisualization::freqToNormX(float freq) const
{
    // Logarithmic mapping: 20Hz -> 0.0, 20kHz -> 1.0
    freq = std::clamp(freq, MIN_FREQ, MAX_FREQ);
    return std::log(freq / MIN_FREQ) / std::log(MAX_FREQ / MIN_FREQ);
}

float RTAVisualization::pixelToFreq(float x, float width) const
{
    float normX = std::clamp(x / width, 0.0f, 1.0f);
    return MIN_FREQ * std::pow(MAX_FREQ / MIN_FREQ, normX);
}

int RTAVisualization::freqToBin(float freq) const
{
    // Frequency to FFT bin: freq * fftSize / sampleRate
    int bin = static_cast<int>(freq * 256.0f / 48000.0f);
    return std::clamp(bin, 0, N_FREQ_BINS - 1);
}

float RTAVisualization::getInterpolatedMagnitude(float binIndex) const
{
    // Cubic Hermite interpolation for smooth curves
    int i = static_cast<int>(binIndex);
    if (i < 0 || i >= N_FREQ_BINS - 1)
        return displayMagnitudeDb[std::clamp(i, 0, N_FREQ_BINS - 1)];

    float frac = binIndex - static_cast<float>(i);

    float y0 = (i > 0) ? displayMagnitudeDb[i - 1] : displayMagnitudeDb[i];
    float y1 = displayMagnitudeDb[i];
    float y2 = displayMagnitudeDb[i + 1];
    float y3 = (i < N_FREQ_BINS - 2) ? displayMagnitudeDb[i + 2] : displayMagnitudeDb[i + 1];

    float a = -0.5f * y0 + 1.5f * y1 - 1.5f * y2 + 0.5f * y3;
    float b = y0 - 2.5f * y1 + 2.0f * y2 - 0.5f * y3;
    float c = -0.5f * y0 + 0.5f * y2;
    float d = y1;

    return a * frac * frac * frac + b * frac * frac + c * frac + d;
}

float RTAVisualization::getInterpolatedMask(float binIndex) const
{
    // Linear interpolation for mask (doesn't need to be as smooth)
    int i = static_cast<int>(binIndex);
    if (i < 0 || i >= N_FREQ_BINS - 1)
        return displayMask[std::clamp(i, 0, N_FREQ_BINS - 1)];

    float frac = binIndex - static_cast<float>(i);
    return displayMask[i] * (1.0f - frac) + displayMask[i + 1] * frac;
}

float RTAVisualization::getInterpolatedMaskB(float binIndex) const
{
    // Linear interpolation for Stream B mask (high-resolution lows)
    int i = static_cast<int>(binIndex);
    if (i < 0 || i >= STREAM_B_BINS - 1)
        return displayMaskB[std::clamp(i, 0, STREAM_B_BINS - 1)];

    float frac = binIndex - static_cast<float>(i);
    return displayMaskB[i] * (1.0f - frac) + displayMaskB[i + 1] * frac;
}

void RTAVisualization::paint(juce::Graphics& g)
{
    drawBackground(g);
    drawGrid(g);
    drawSpectrumCurve(g);
    drawDividerLine(g);
    drawReductionCurve(g);
}

void RTAVisualization::resized()
{
    // Nothing special needed
}

void RTAVisualization::drawBackground(juce::Graphics& g)
{
    // Dark background matching DeBleed theme
    g.fillAll(juce::Colour::fromRGB(12, 12, 14));  // visualizerBackground
}

void RTAVisualization::drawGrid(juce::Graphics& g)
{
    const float width = static_cast<float>(getWidth());
    const float height = static_cast<float>(getHeight());
    const float spectrumHeight = getSpectrumHeight();
    const float dividerY = getDividerY();

    // Frequency grid lines (logarithmic)
    const float freqMarkers[] = {20.0f, 50.0f, 100.0f, 200.0f, 500.0f,
                                  1000.0f, 2000.0f, 5000.0f, 10000.0f, 20000.0f};

    g.setColour(juce::Colours::white.withAlpha(0.05f));
    g.setFont(9.0f);

    for (float freq : freqMarkers)
    {
        float normX = freqToNormX(freq);
        float x = normX * width;

        // Vertical line
        g.drawVerticalLine(static_cast<int>(x), 0.0f, height);

        // Label
        g.setColour(juce::Colours::white.withAlpha(0.4f));
        juce::String label;
        if (freq >= 1000.0f)
            label = juce::String(static_cast<int>(freq / 1000.0f)) + "k";
        else
            label = juce::String(static_cast<int>(freq));

        g.drawText(label, static_cast<int>(x) - 15, static_cast<int>(height) - 12, 30, 12,
                   juce::Justification::centred, false);
        g.setColour(juce::Colours::white.withAlpha(0.05f));
    }

    // dB grid lines for spectrum section (top)
    const float dbMarkers[] = {0.0f, -10.0f, -20.0f, -30.0f, -40.0f, -50.0f};

    for (float db : dbMarkers)
    {
        float normY = juce::jmap(db, MIN_DB, MAX_DB, 1.0f, 0.0f);
        float y = spectrumHeight * normY;

        // Horizontal line
        g.setColour(juce::Colours::white.withAlpha(0.05f));
        g.drawHorizontalLine(static_cast<int>(y), 0.0f, width);

        // Label on left
        g.setColour(juce::Colours::white.withAlpha(0.3f));
        g.drawText(juce::String(static_cast<int>(db)), 2, static_cast<int>(y) - 6, 25, 12,
                   juce::Justification::left, false);
    }

    // dB grid lines for reduction section (bottom)
    const float reductionMarkers[] = {0.0f, -10.0f, -20.0f, -30.0f, -40.0f};
    const float reductionHeight = getReductionHeight();

    for (float db : reductionMarkers)
    {
        float normY = juce::jmap(db, 0.0f, MIN_REDUCTION_DB, 0.0f, 1.0f);
        float y = dividerY + reductionHeight * normY;

        if (db != 0.0f)  // Skip 0dB as it's the divider line
        {
            g.setColour(juce::Colours::white.withAlpha(0.03f));
            g.drawHorizontalLine(static_cast<int>(y), 0.0f, width);
        }

        // Label on right
        g.setColour(juce::Colours::white.withAlpha(0.25f));
        g.drawText(juce::String(static_cast<int>(db)), static_cast<int>(width) - 27,
                   static_cast<int>(y) - 6, 25, 12, juce::Justification::right, false);
    }
}

void RTAVisualization::drawSpectrumCurve(juce::Graphics& g)
{
    const float width = static_cast<float>(getWidth());
    const float spectrumHeight = getSpectrumHeight();

    juce::Path curve;
    bool pathStarted = false;

    const float pixelStep = 2.0f;  // Process every 2 pixels for performance

    for (float x = 0; x < width; x += pixelStep)
    {
        float freq = pixelToFreq(x, width);
        float binIndex = freq * 256.0f / 48000.0f;

        float db = getInterpolatedMagnitude(binIndex);

        // Map dB to Y pixel
        float normY = juce::jmap(db, MIN_DB, MAX_DB, 1.0f, 0.0f);
        normY = std::clamp(normY, 0.0f, 1.0f);
        float y = spectrumHeight * normY;

        if (!pathStarted)
        {
            curve.startNewSubPath(x, y);
            pathStarted = true;
        }
        else
        {
            curve.lineTo(x, y);
        }
    }

    // Create fill path
    juce::Path fillPath = curve;
    fillPath.lineTo(width, spectrumHeight);
    fillPath.lineTo(0, spectrumHeight);
    fillPath.closeSubPath();

    // Fill with cyan gradient
    juce::ColourGradient gradient(
        juce::Colours::cyan.withAlpha(0.15f), 0, spectrumHeight,
        juce::Colours::cyan.withAlpha(0.0f), 0, 0, false);
    g.setGradientFill(gradient);
    g.fillPath(fillPath);

    // Stroke curve
    g.setColour(juce::Colours::white.withAlpha(0.9f));
    g.strokePath(curve, juce::PathStrokeType(1.5f));
}

void RTAVisualization::drawDividerLine(juce::Graphics& g)
{
    const float dividerY = getDividerY();
    g.setColour(juce::Colours::white.withAlpha(0.3f));
    g.drawHorizontalLine(static_cast<int>(dividerY), 0.0f, static_cast<float>(getWidth()));

    // Label "GR" on the left of reduction section
    g.setColour(juce::Colours::white.withAlpha(0.4f));
    g.setFont(10.0f);
    g.drawText("GR", 2, static_cast<int>(dividerY) + 2, 20, 12, juce::Justification::left, false);
}

float RTAVisualization::getBandpassMagnitude(float freq, float centerFreq, float Q) const
{
    // 2nd-order bandpass magnitude response
    // |H(f)| = 1 / sqrt(1 + Q² * ((f/fc) - (fc/f))²)

    if (freq <= 0.0f || centerFreq <= 0.0f)
        return 0.0f;

    float ratio = freq / centerFreq;
    float term = ratio - (1.0f / ratio);  // (f/fc) - (fc/f)
    float denominator = 1.0f + Q * Q * term * term;

    return 1.0f / std::sqrt(denominator);
}

float RTAVisualization::getPeakFilterMagnitude(float freq, float centerFreq, float Q, float gainDb) const
{
    // 2nd-order peaking EQ magnitude response
    // At center frequency: |H(fc)| = gain
    // Away from center: approaches unity
    //
    // Simplified approximation for visualization:
    // |H(f)| ≈ 1 + (gain - 1) * bandpassShape
    // where bandpassShape = 1 / sqrt(1 + Q² * ((f/fc) - (fc/f))²)

    if (freq <= 0.0f || centerFreq <= 0.0f)
        return 1.0f;

    float gainLinear = juce::Decibels::decibelsToGain(gainDb);

    float ratio = freq / centerFreq;
    float term = ratio - (1.0f / ratio);
    float denominator = 1.0f + Q * Q * term * term;
    float bandpassShape = 1.0f / std::sqrt(denominator);

    // Interpolate between unity (1.0) and target gain based on bandpass shape
    return 1.0f + (gainLinear - 1.0f) * bandpassShape;
}

float RTAVisualization::getCombinedGainAtFreq(float freq) const
{
    // Get the hunter filter states from processor
    auto hunterStates = audioProcessor.getHunterStates();

    // Start with unity gain
    float combinedGain = 1.0f;

    // Each active hunter contributes a peak/notch at its frequency
    for (const auto& hunter : hunterStates)
    {
        if (!hunter.active)
            continue;

        // Calculate dynamic Q based on gain (matches ActiveFilterPool logic: Q 4-16)
        float dynamicQ = 4.0f + (1.0f - hunter.gain) * 12.0f;

        // Get the peak filter magnitude response at this frequency
        float peakMag = getPeakFilterMagnitude(freq, hunter.freq, dynamicQ,
            juce::Decibels::gainToDecibels(std::max(hunter.gain, 0.01f)));

        // Apply this filter's effect (multiply gains for series processing)
        combinedGain *= peakMag;
    }

    return std::max(combinedGain, 0.0001f);  // Prevent log(0)
}

void RTAVisualization::drawReductionCurve(juce::Graphics& g)
{
    const float width = static_cast<float>(getWidth());
    const float dividerY = getDividerY();
    const float reductionHeight = getReductionHeight();

    // === DRAW RAW NEURAL MASK (cyan outline) ===
    // This shows what the neural network is actually requesting
    // Uses DUAL-BAND: Stream B (23.4 Hz/bin) for lows, Stream A (187.5 Hz/bin) for mids/highs
    juce::Path maskCurve;
    bool maskStarted = false;

    const float pixelStep = 2.0f;
    static constexpr float CROSSOVER_FREQ = 1500.0f;  // Stream B to Stream A crossover

    for (float x = 0; x < width; x += pixelStep)
    {
        float freq = pixelToFreq(x, width);
        float maskGain;

        if (freq < CROSSOVER_FREQ)
        {
            // Use Stream B for lows (23.4 Hz/bin resolution at 48kHz, 2048-point FFT)
            float binIndex = freq * 2048.0f / 48000.0f;
            maskGain = getInterpolatedMaskB(binIndex);
        }
        else
        {
            // Use Stream A for mids/highs (187.5 Hz/bin resolution at 48kHz, 256-point FFT)
            float binIndex = freq * 256.0f / 48000.0f;
            maskGain = getInterpolatedMask(binIndex);
        }

        // Convert mask to dB (mask is 0-1 linear gain)
        float maskDb = 20.0f * std::log10f(std::max(maskGain, 0.0001f));
        maskDb = std::clamp(maskDb, MIN_REDUCTION_DB, 0.0f);

        float normY = juce::jmap(maskDb, 0.0f, MIN_REDUCTION_DB, 0.0f, 1.0f);
        float y = dividerY + reductionHeight * normY;

        if (!maskStarted)
        {
            maskCurve.startNewSubPath(x, y);
            maskStarted = true;
        }
        else
        {
            maskCurve.lineTo(x, y);
        }
    }

    // Draw raw mask as cyan line
    g.setColour(juce::Colours::cyan.withAlpha(0.4f));
    g.strokePath(maskCurve, juce::PathStrokeType(1.0f));

    // === DRAW HUNTER FILTER RESPONSE (purple filled) ===
    juce::Path curve;
    bool pathStarted = false;

    for (float x = 0; x < width; x += pixelStep)
    {
        float freq = pixelToFreq(x, width);

        // Get combined frequency response from all 32 hunters
        float gainLinear = getCombinedGainAtFreq(freq);

        // Convert to dB
        float gainDb = 20.0f * std::log10f(gainLinear);
        gainDb = std::clamp(gainDb, MIN_REDUCTION_DB, 0.0f);

        // Map reduction dB (0 to -40) to Y pixel (dividerY to bottom)
        float normY = juce::jmap(gainDb, 0.0f, MIN_REDUCTION_DB, 0.0f, 1.0f);
        float y = dividerY + reductionHeight * normY;

        if (!pathStarted)
        {
            curve.startNewSubPath(x, y);
            pathStarted = true;
        }
        else
        {
            curve.lineTo(x, y);
        }
    }

    // Create fill path
    juce::Path fillPath = curve;
    fillPath.lineTo(width, dividerY);
    fillPath.lineTo(0, dividerY);
    fillPath.closeSubPath();

    // Fill with purple gradient (matches Range knob and GR meter)
    juce::ColourGradient gradient(
        juce::Colour(0xffcc66ff).withAlpha(0.35f), 0, dividerY,  // Bright purple
        juce::Colour(0xff800080).withAlpha(0.1f), 0, static_cast<float>(getHeight()), false);  // Purple accent
    g.setGradientFill(gradient);
    g.fillPath(fillPath);

    // Stroke curve
    g.setColour(juce::Colour(0xffcc66ff).withAlpha(0.8f));  // Bright purple
    g.strokePath(curve, juce::PathStrokeType(1.5f));
}
