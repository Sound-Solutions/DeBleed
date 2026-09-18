#pragma once
#include <vector>

// Normalised feedback comb; all mutable DSP state belongs to the audio thread.
class HarmonicComb
{
public:
    void prepare(double sampleRate);
    void reset();
    void setDepthDb(float depthDb);
    void setTarget(float periodSamples, bool confident);
    void setVocalConfidence(float confidence) { vocalPresent_ = confidence >= 0.5f; }
    float process(float input);
    float getFeedbackGain() const { return feedbackGain_; }

    // Probe hook: bypass control slews for steady-state transfer measurements.
    // reset() restores normal tracking; the plugin never calls this hook.
    void setForcedState(float periodSamples, float gain);

private:
    double sampleRate_ = 0.0;
    float depthGain_ = 0.0f, envelope_ = 0.0f, feedbackGain_ = 0.0f;
    float riseCoeff_ = 0.0f, fallCoeff_ = 0.0f, periodCoeff_ = 0.0f;
    float period_ = 0.0f, targetPeriod_ = 0.0f;
    bool confident_ = false, vocalPresent_ = false, forced_ = false;
    int writeIndex_ = 0;
    std::vector<float> delay_;
};
