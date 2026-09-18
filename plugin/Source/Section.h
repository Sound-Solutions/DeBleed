#pragma once

#include <JuceHeader.h>

class Section : public juce::Component
{
public:
    void paint(juce::Graphics&) override;

protected:
    Section(const juce::String& title, juce::uint32 colour, float labelBaseline);

    struct Knob
    {
        juce::Slider slider;
        std::unique_ptr<juce::AudioProcessorValueTreeState::SliderAttachment> attachment;

        void setup(Section&, juce::AudioProcessorValueTreeState&, const juce::String& parameterID,
                   const juce::String& name, const juce::String& unit,
                   float centreX, float centreY, bool big = false);
    };

    struct RingButton
    {
        juce::ToggleButton button;
        std::unique_ptr<juce::AudioProcessorValueTreeState::ButtonAttachment> attachment;

        void setup(Section&, juce::AudioProcessorValueTreeState&, const juce::String& parameterID,
                   const juce::String& name, const juce::String& unit,
                   float centreX, float centreY);
    };

private:
    struct PaintedLabel
    {
        juce::String name, unit;
        float centreX;
    };

    juce::String title_;
    juce::uint32 colour_;
    float labelBaseline_;
    std::vector<PaintedLabel> labels_;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(Section)
};

class ExpanderSection : public Section
{
public:
    explicit ExpanderSection(juce::AudioProcessorValueTreeState&);

private:
    Knob threshold_, ratio_, attack_, release_, range_;
};

class VocalSection : public Section
{
public:
    explicit VocalSection(juce::AudioProcessorValueTreeState&);

private:
    RingButton look_, clarity_, harmonic_;
    Knob depth_;
};

class OutputSection : public Section
{
public:
    explicit OutputSection(juce::AudioProcessorValueTreeState&);

private:
    Knob mix_, output_;
};
