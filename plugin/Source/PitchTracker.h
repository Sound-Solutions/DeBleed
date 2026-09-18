#pragma once
#include <vector>

// Streaming McLeod NSDF tracker; storage is allocated only by prepare().
class PitchTracker
{
public:
    void prepare(double sampleRate);
    void reset();
    void pushSample(float input);
    float getPeriodSamples() const { return periodSamples_; }
    float getConfidence() const { return confidence_; }
    float getFrequencyHz() const;

private:
    struct Biquad
    {
        double b0 = 0.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;
        double process(double input);
    };

    void analyse();
    static constexpr int windowSize = 1024;
    double sampleRate_ = 0.0;
    int decimation_ = 1, decimationCounter_ = 0;
    int writeIndex_ = 0, samplesFilled_ = 0, updateCounter_ = 0;
    int minLag_ = 1, maxLag_ = 1;
    float periodSamples_ = 0.0f, confidence_ = 0.0f;
    Biquad lowpass_[2];
    std::vector<float> ring_, window_, nsdf_;
};
