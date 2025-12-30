/*
  ==============================================================================
    DeBleedLookAndFeel.h
    Style: FabFilter-inspired (Vector Knobs, Amber Power, Dark Theme)
    Based on KineticsLookAndFeel
  ==============================================================================
*/
#pragma once
#include <JuceHeader.h>

class DeBleedLookAndFeel : public juce::LookAndFeel_V4
{
public:
    DeBleedLookAndFeel();

    // Button rendering
    void drawButtonBackground(juce::Graphics&, juce::Button&, const juce::Colour&, bool, bool) override;
    void drawButtonText(juce::Graphics& g, juce::TextButton& button,
                        bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;

    // Rotary knob with arc
    void drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                          float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                          juce::Slider& slider) override;

    // Amber power toggle button
    void drawToggleButton(juce::Graphics& g, juce::ToggleButton& btn,
                          bool shouldDrawButtonAsHighlighted, bool shouldDrawButtonAsDown) override;

    // Progress bar styling
    void drawProgressBar(juce::Graphics& g, juce::ProgressBar& bar,
                         int width, int height, double progress,
                         const juce::String& textToShow) override;

    // Tab button styling
    void drawTabButton(juce::Graphics& g, juce::Rectangle<int> bounds,
                       const juce::String& text, bool isActive, bool isHovered);

    // Color constants
    static constexpr juce::uint32 mainBackground = 0xff0a0b0d;      // RGB(10, 11, 13)
    static constexpr juce::uint32 visualizerBackground = 0xff0c0c0e; // RGB(12, 12, 14)
    static constexpr juce::uint32 panelBackground = 0xff141618;      // RGB(20, 22, 24)
    static constexpr juce::uint32 popupBackground = 0xff191b1e;      // RGB(25, 27, 30)

    static constexpr juce::uint32 arcBackgroundColor = 0xff2d3034;   // RGB(45, 48, 52)
    static constexpr juce::uint32 cyanAccent = 0xff00ffff;           // RGB(0, 255, 255) - pure cyan
    static constexpr juce::uint32 orangeAccent = 0xffffa500;         // Orange
    static constexpr juce::uint32 yellowAccent = 0xffffff00;         // Yellow
    static constexpr juce::uint32 purpleAccent = 0xff800080;         // Purple
};
