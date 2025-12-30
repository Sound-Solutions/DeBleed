/*
  ==============================================================================
    DeBleedLookAndFeel.cpp
    Style: FabFilter-inspired (Vector Knobs, Amber Power, Dark Theme)
  ==============================================================================
*/
#include "DeBleedLookAndFeel.h"

DeBleedLookAndFeel::DeBleedLookAndFeel()
{
    // Global Text Colors
    setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.7f));
    setColour(juce::Slider::textBoxTextColourId, juce::Colours::white.withAlpha(0.9f));
    setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);

    // ComboBox Colors
    setColour(juce::ComboBox::backgroundColourId, juce::Colour(panelBackground));
    setColour(juce::ComboBox::outlineColourId, juce::Colour::fromRGB(60, 60, 60));
    setColour(juce::ComboBox::arrowColourId, juce::Colours::white.withAlpha(0.6f));
    setColour(juce::ComboBox::textColourId, juce::Colours::white.withAlpha(0.9f));

    // Popup Menu Colors
    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(popupBackground));
    setColour(juce::PopupMenu::textColourId, juce::Colours::white.withAlpha(0.9f));
    setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colours::white.withAlpha(0.1f));
    setColour(juce::PopupMenu::highlightedTextColourId, juce::Colours::white);

    // TextEditor Colors
    setColour(juce::TextEditor::backgroundColourId, juce::Colour(panelBackground));
    setColour(juce::TextEditor::textColourId, juce::Colours::white.withAlpha(0.8f));
    setColour(juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
}

void DeBleedLookAndFeel::drawButtonBackground(juce::Graphics& g, juce::Button& button,
                                               const juce::Colour& backgroundColour,
                                               bool shouldDrawButtonAsHighlighted,
                                               bool shouldDrawButtonAsDown)
{
    auto bounds = button.getLocalBounds().toFloat().reduced(1.0f);

    // Check if this is a tab button
    if (button.getProperties().contains("isTabButton"))
    {
        bool isActive = button.getToggleState();

        if (isActive)
        {
            g.setColour(juce::Colours::white.withAlpha(0.12f));
            g.fillRoundedRectangle(bounds, 4.0f);
            g.setColour(juce::Colours::white.withAlpha(0.3f));
            g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
        }
        else if (shouldDrawButtonAsHighlighted)
        {
            g.setColour(juce::Colours::white.withAlpha(0.08f));
            g.fillRoundedRectangle(bounds, 4.0f);
        }
        else
        {
            g.setColour(juce::Colours::white.withAlpha(0.04f));
            g.fillRoundedRectangle(bounds, 4.0f);
        }
        return;
    }

    // Standard button style
    if (!button.isEnabled())
    {
        g.setColour(juce::Colour(panelBackground).withAlpha(0.5f));
        g.fillRoundedRectangle(bounds, 4.0f);
        g.setColour(juce::Colours::white.withAlpha(0.08f));
        g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
        return;
    }

    if (shouldDrawButtonAsDown)
    {
        g.setColour(juce::Colours::white.withAlpha(0.15f));
    }
    else if (shouldDrawButtonAsHighlighted)
    {
        g.setColour(juce::Colours::white.withAlpha(0.1f));
    }
    else
    {
        g.setColour(juce::Colour(panelBackground));
    }

    g.fillRoundedRectangle(bounds, 4.0f);
    g.setColour(juce::Colours::white.withAlpha(0.2f));
    g.drawRoundedRectangle(bounds, 4.0f, 1.0f);
}

void DeBleedLookAndFeel::drawButtonText(juce::Graphics& g, juce::TextButton& button,
                                         bool shouldDrawButtonAsHighlighted,
                                         bool shouldDrawButtonAsDown)
{
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));

    if (!button.isEnabled())
        g.setColour(juce::Colours::white.withAlpha(0.25f));  // Clearly disabled
    else if (button.getToggleState() || shouldDrawButtonAsDown || shouldDrawButtonAsHighlighted)
        g.setColour(juce::Colours::white);
    else
        g.setColour(juce::Colours::white.withAlpha(0.6f));

    g.drawText(button.getButtonText(), button.getLocalBounds(), juce::Justification::centred);
}

void DeBleedLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                           float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                           juce::Slider& slider)
{
    auto bounds = juce::Rectangle<float>(x, y, width, height).reduced(2.0f);
    auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
    auto toAngle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
    auto center = bounds.getCentre();

    // A. Background Track
    g.setColour(juce::Colour(arcBackgroundColor));
    juce::Path bgPath;
    bgPath.addCentredArc(center.x, center.y, radius, radius, 0.0f, rotaryStartAngle, rotaryEndAngle, true);
    g.strokePath(bgPath, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // B. Value Arc
    juce::Path valPath;

    // Check if arc should be inverted (fill from current to end instead of start to current)
    bool invertArc = slider.getProperties().contains("invertArc") &&
                     static_cast<bool>(slider.getProperties()["invertArc"]);

    if (invertArc)
        valPath.addCentredArc(center.x, center.y, radius, radius, 0.0f, toAngle, rotaryEndAngle, true);
    else
        valPath.addCentredArc(center.x, center.y, radius, radius, 0.0f, rotaryStartAngle, toAngle, true);

    // Get knob color from properties, default to cyan
    juce::Colour arcColor = juce::Colour::fromRGB(0, 255, 255);  // Pure cyan (DrumCipher style)
    if (slider.getProperties().contains("knobColor"))
    {
        juce::int64 colorVal = static_cast<juce::int64>(slider.getProperties()["knobColor"]);
        arcColor = juce::Colour(static_cast<juce::uint32>(colorVal));
    }

    g.setColour(arcColor);
    g.strokePath(valPath, juce::PathStrokeType(3.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // C. Knob Body
    auto knobRadius = radius - 5.0f;
    juce::ColourGradient knobGrad(juce::Colour::fromRGB(60, 65, 70), center.x, center.y - knobRadius,
                                  juce::Colour::fromRGB(25, 28, 30), center.x, center.y + knobRadius, false);
    g.setGradientFill(knobGrad);
    g.fillEllipse(center.x - knobRadius, center.y - knobRadius, knobRadius * 2.0f, knobRadius * 2.0f);

    // D. Pointer
    juce::Path p;
    p.addRectangle(-1.5f, -knobRadius + 2.0f, 3.0f, 5.0f);
    p.applyTransform(juce::AffineTransform::rotation(toAngle).translated(center));
    g.setColour(juce::Colours::white.withAlpha(0.9f));
    g.fillPath(p);
}

void DeBleedLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& btn,
                                           bool shouldDrawButtonAsHighlighted,
                                           bool shouldDrawButtonAsDown)
{
    auto bounds = btn.getLocalBounds().toFloat().reduced(2.0f);

    // For larger toggle buttons, use default rendering (with text)
    if (bounds.getWidth() > 40)
    {
        LookAndFeel_V4::drawToggleButton(g, btn, shouldDrawButtonAsHighlighted, shouldDrawButtonAsDown);
        return;
    }

    // Small power button style
    bool on = btn.getToggleState();

    // Check for inverted colors (for bypass button: orange when OFF = active)
    bool invertColors = btn.getProperties().contains("invertColors") &&
                        static_cast<bool>(btn.getProperties()["invertColors"]);
    if (invertColors)
        on = !on;  // Invert: show orange when bypass is OFF (active)

    auto center = bounds.getCentre();
    float r = std::min(bounds.getWidth(), bounds.getHeight()) / 2.0f;

    if (on)
    {
        juce::ColourGradient onGrad(juce::Colour::fromRGB(255, 215, 150), bounds.getX(), bounds.getY(),
                                    juce::Colour::fromRGB(200, 100, 20), bounds.getX(), bounds.getBottom(), false);
        g.setGradientFill(onGrad);
    }
    else
    {
        g.setColour(juce::Colour::fromRGB(40, 40, 40));
    }

    g.fillRoundedRectangle(bounds, 4.0f);

    g.setColour(juce::Colours::black.withAlpha(on ? 0.4f : 0.6f));
    g.drawRoundedRectangle(bounds, 4.0f, 1.0f);

    // Power icon (matching Kinetics style - gap at top for the power line)
    juce::Path icon;
    float iconR = r * 0.55f;
    icon.addCentredArc(center.x, center.y, iconR, iconR, 0.0f, 0.64f, 5.64f, true);
    icon.startNewSubPath(center.x, center.y - iconR + 1.0f);
    icon.lineTo(center.x, center.y);

    if (on)
        g.setColour(juce::Colours::white.withAlpha(0.95f));
    else
        g.setColour(juce::Colours::white.withAlpha(0.4f));

    g.strokePath(icon, juce::PathStrokeType(1.5f, juce::PathStrokeType::curved, juce::PathStrokeType::rounded));

    // Highlight on top when ON
    if (on)
    {
        g.setColour(juce::Colours::white.withAlpha(0.2f));
        g.fillRoundedRectangle(bounds.getX(), bounds.getY(), bounds.getWidth(), bounds.getHeight() * 0.4f, 4.0f);
    }
}

void DeBleedLookAndFeel::drawProgressBar(juce::Graphics& g, juce::ProgressBar& bar,
                                          int width, int height, double progress,
                                          const juce::String& textToShow)
{
    auto bounds = juce::Rectangle<float>(0, 0, width, height);

    // Background
    g.setColour(juce::Colour(panelBackground));
    g.fillRoundedRectangle(bounds, 4.0f);

    // Border
    g.setColour(juce::Colours::white.withAlpha(0.1f));
    g.drawRoundedRectangle(bounds, 4.0f, 1.0f);

    // Progress fill
    if (progress > 0.0)
    {
        auto progressBounds = bounds.reduced(2.0f);
        progressBounds.setWidth(progressBounds.getWidth() * static_cast<float>(progress));

        juce::ColourGradient grad(juce::Colour(cyanAccent).withAlpha(0.8f), progressBounds.getX(), progressBounds.getY(),
                                  juce::Colour(cyanAccent).withAlpha(0.5f), progressBounds.getRight(), progressBounds.getY(), false);
        g.setGradientFill(grad);
        g.fillRoundedRectangle(progressBounds, 3.0f);
    }

    // Text
    if (textToShow.isNotEmpty())
    {
        g.setColour(juce::Colours::white.withAlpha(0.8f));
        g.setFont(juce::FontOptions(11.0f));
        g.drawText(textToShow, bounds, juce::Justification::centred);
    }
}

void DeBleedLookAndFeel::drawTabButton(juce::Graphics& g, juce::Rectangle<int> bounds,
                                        const juce::String& text, bool isActive, bool isHovered)
{
    auto boundsF = bounds.toFloat();

    if (isActive)
    {
        g.setColour(juce::Colours::white.withAlpha(0.12f));
        g.fillRoundedRectangle(boundsF, 4.0f);
        g.setColour(juce::Colours::white.withAlpha(0.3f));
        g.drawRoundedRectangle(boundsF, 4.0f, 1.0f);
    }
    else if (isHovered)
    {
        g.setColour(juce::Colours::white.withAlpha(0.08f));
        g.fillRoundedRectangle(boundsF, 4.0f);
    }
    else
    {
        g.setColour(juce::Colours::white.withAlpha(0.04f));
        g.fillRoundedRectangle(boundsF, 4.0f);
    }

    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.setColour(isActive ? juce::Colours::white : juce::Colours::white.withAlpha(0.6f));
    g.drawText(text, bounds, juce::Justification::centred);
}
