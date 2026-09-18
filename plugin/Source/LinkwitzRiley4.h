#pragma once

// A fourth-order Linkwitz-Riley crossover. The sum is a unity-magnitude allpass.
class LinkwitzRiley4
{
public:
    void prepare(double sampleRate, double cutoffHz);
    void reset();
    void processSplit(float input, float& low, float& high);
    float processHigh(float input);

private:
    struct Biquad
    {
        double b0 = 0.0, b1 = 0.0, b2 = 0.0, a1 = 0.0, a2 = 0.0;
        double z1 = 0.0, z2 = 0.0;
        double process(double input);
    };

    Biquad low_[2], high_[2];
};
