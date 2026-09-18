#include "Section.h"
#include "DeBleedLookAndFeel.h"
#include "PluginProcessor.h"

namespace
{
void placeControl(juce::Component& control, float centreX, float centreY, int size)
{
    const float x = centreX - static_cast<float>(size) * 0.5f;
    const float y = centreY - static_cast<float>(size) * 0.5f;
    const int left = static_cast<int>(std::floor(x));
    const int top = static_cast<int>(std::floor(y));
    control.setBounds(left, top, size, size);
    // The 113 px expander cells put four knob centres on half pixels.
    control.setTransform(juce::AffineTransform::translation(x - static_cast<float>(left),
                                                           y - static_cast<float>(top)));
}
}

Section::Section(const juce::String& title, juce::uint32 colour, float labelBaseline)
    : title_(title), colour_(colour), labelBaseline_(labelBaseline)
{
    setOpaque(true);
}

void Section::Knob::setup(Section& parent, juce::AudioProcessorValueTreeState& parameters,
                          const juce::String& parameterID, const juce::String& name,
                          const juce::String& unit, float centreX, float centreY, bool big)
{
    slider.setSliderStyle(juce::Slider::RotaryVerticalDrag);
    slider.setTextBoxStyle(juce::Slider::NoTextBox, false, 0, 0);
    slider.setTitle(name);
    slider.getProperties().set("knobColor", static_cast<juce::int64>(parent.colour_));
    slider.getProperties().set("bigKnob", big);
    placeControl(slider, centreX, centreY, big ? 84 : 60);
    parent.addAndMakeVisible(slider);
    parent.labels_.push_back({name, unit, centreX});
    attachment = std::make_unique<juce::AudioProcessorValueTreeState::SliderAttachment>(
        parameters, parameterID, slider);
}

void Section::RingButton::setup(Section& parent, juce::AudioProcessorValueTreeState& parameters,
                                const juce::String& parameterID, const juce::String& name,
                                const juce::String& unit, float centreX, float centreY)
{
    button.setTitle(name);
    button.getProperties().set("ringStyle", true);
    button.getProperties().set("ringColour", static_cast<juce::int64>(parent.colour_));
    placeControl(button, centreX, centreY, 28);
    parent.addAndMakeVisible(button);
    parent.labels_.push_back({name, unit, centreX});
    attachment = std::make_unique<juce::AudioProcessorValueTreeState::ButtonAttachment>(
        parameters, parameterID, button);
}

void Section::paint(juce::Graphics& g)
{
    g.fillAll(juce::Colour(DeBleedLookAndFeel::panelBackground));

    {
        const juce::Graphics::ScopedSaveState saved(g);
        g.addTransform(juce::AffineTransform::rotation(-juce::MathConstants<float>::halfPi)
                           .translated(14.0f, static_cast<float>(getHeight()) * 0.5f));
        g.setColour(juce::Colour(DeBleedLookAndFeel::dimText));
        const auto titleFont = juce::Font(juce::FontOptions(9.0f, juce::Font::bold))
                                   .withExtraKerningFactor(2.2f / 9.0f);
        DeBleedLookAndFeel::drawTextAtBaseline(g, title_, titleFont, 0.0f, 0.0f);
    }

    const auto nameFont = juce::Font(juce::FontOptions(8.0f, juce::Font::bold))
                              .withExtraKerningFactor(1.2f / 8.0f);
    const auto unitFont = juce::Font(juce::FontOptions(7.0f).withStyle("Medium"))
                              .withExtraKerningFactor(0.4f / 7.0f);
    for (const auto& label : labels_)
    {
        juce::GlyphArrangement name, unit;
        name.addLineOfText(nameFont, label.name, 0.0f, 0.0f);
        const auto nameBounds = name.getBoundingBox(0, name.getNumGlyphs(), true);
        float right = nameBounds.getRight();
        if (label.unit.isNotEmpty())
        {
            unit.addLineOfText(unitFont, " " + label.unit, right, 0.0f);
            right = unit.getBoundingBox(0, unit.getNumGlyphs(), true).getRight();
        }
        const auto position = juce::AffineTransform::translation(
            label.centreX - (nameBounds.getX() + right) * 0.5f, labelBaseline_);
        g.setColour(juce::Colour(DeBleedLookAndFeel::labelText));
        name.draw(g, position);
        g.setColour(juce::Colour(DeBleedLookAndFeel::dimText));
        unit.draw(g, position);
    }
}

ExpanderSection::ExpanderSection(juce::AudioProcessorValueTreeState& parameters)
    : Section("EXPANDER", DeBleedLookAndFeel::orangeAccent, 114.0f)
{
    threshold_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_EXP_THRESHOLD, "THRESH", "dB", 82.0f, 60.0f, true);
    ratio_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_EXP_RATIO, "RATIO", ":1", 194.5f, 60.0f);
    attack_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_EXP_ATTACK, "ATTACK", "ms", 307.5f, 60.0f);
    release_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_EXP_RELEASE, "RELEASE", "ms", 420.5f, 60.0f);
    range_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_EXP_RANGE, "RANGE", "dB", 533.5f, 60.0f);
}

VocalSection::VocalSection(juce::AudioProcessorValueTreeState& parameters)
    : Section("VOCAL", DeBleedLookAndFeel::greenAccent, 120.0f)
{
    look_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_LOOKAHEAD, "LOOK", "1 ms", 66.0f, 57.0f);
    clarity_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_CONSONANT, "CLARITY", "", 134.0f, 57.0f);
    harmonic_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_COMB, "HARMONIC", "", 202.0f, 57.0f);
    depth_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_COMB_DEPTH, "DEPTH", "dB", 278.0f, 57.0f);
}

OutputSection::OutputSection(juce::AudioProcessorValueTreeState& parameters)
    : Section("OUTPUT", DeBleedLookAndFeel::cyanAccent, 120.0f)
{
    mix_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_MIX, "MIX", "%", 85.0f, 57.0f);
    output_.setup(*this, parameters, DeBleedAudioProcessor::PARAM_OUTPUT_GAIN, "OUTPUT", "dB", 203.0f, 57.0f);
}
