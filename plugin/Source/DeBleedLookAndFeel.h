#pragma once

#include <JuceHeader.h>

class DeBleedLookAndFeel : public juce::LookAndFeel_V4
{
public:
    DeBleedLookAndFeel();

    void drawRotarySlider(juce::Graphics&, int x, int y, int width, int height,
                          float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                          juce::Slider&) override;
    void drawToggleButton(juce::Graphics&, juce::ToggleButton&, bool, bool) override;

    static void drawBody(juce::Graphics&, juce::Point<float> centre, float radius,
                         bool innerRing);
    // The mock's glow is the stroke blurred (SVG feGaussianBlur, sigma 2.2) under the solid
    // stroke. JUCE's DropShadow box blur at radius 4 lands on sigma 2.3, so this is the same
    // halo, not a wider ring. Always paint the solid stroke over it afterwards.
    static void drawGlow(juce::Graphics&, const juce::Path&, const juce::PathStrokeType&,
                         juce::Colour, float alpha);
    // Every font goes through here so the typeface is explicit: JUCE resolves default-named
    // fonts through the Desktop's LookAndFeel, not the editor's, and its macOS default is
    // Lucida Grande (no Medium face).
    static juce::Font font(float height, const juce::String& style);
    static void drawTextAtBaseline(juce::Graphics&, const juce::String&, const juce::Font&,
                                   float x, float baseline, bool centred = true);

    static constexpr juce::uint32 groove = 0xff080808;
    static constexpr juce::uint32 rim = 0xff0a0a0a;
    static constexpr juce::uint32 mainBackground = 0xff0f0f0f;
    static constexpr juce::uint32 panelBackground = 0xff111111;
    static constexpr juce::uint32 bodyBackground = 0xff171717;
    static constexpr juce::uint32 meterBackground = 0xff141414;
    static constexpr juce::uint32 bodyEdge = 0xff1a1a1a;
    static constexpr juce::uint32 arcBackground = 0xff282828;
    static constexpr juce::uint32 bodyCentre = 0xff2c2c2c;
    static constexpr juce::uint32 inactiveRing = 0xff3a3a3a;
    static constexpr juce::uint32 valueText = 0xffededed;
    static constexpr juce::uint32 labelText = 0xff8a8a8a;
    static constexpr juce::uint32 dimText = 0xff5e5e5e;
    static constexpr juce::uint32 orangeAccent = 0xffff9500;
    static constexpr juce::uint32 greenAccent = 0xff39ff14;
    static constexpr juce::uint32 cyanAccent = 0xff00d4ff;
};
