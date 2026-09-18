#include "DeBleedLookAndFeel.h"

DeBleedLookAndFeel::DeBleedLookAndFeel()
{
    setColour(juce::Label::textColourId, juce::Colour(labelText));
    setColour(juce::Slider::textBoxTextColourId, juce::Colour(valueText));
    setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    setColour(juce::ComboBox::backgroundColourId, juce::Colour(panelBackground));
    setColour(juce::ComboBox::outlineColourId, juce::Colour(inactiveRing));
    setColour(juce::ComboBox::arrowColourId, juce::Colour(labelText));
    setColour(juce::ComboBox::textColourId, juce::Colour(valueText));
    setColour(juce::PopupMenu::backgroundColourId, juce::Colour(mainBackground));
    setColour(juce::PopupMenu::textColourId, juce::Colour(valueText));
    setColour(juce::PopupMenu::highlightedBackgroundColourId, juce::Colour(arcBackground));
    setColour(juce::PopupMenu::highlightedTextColourId, juce::Colour(valueText));
    setColour(juce::TextEditor::backgroundColourId, juce::Colour(panelBackground));
    setColour(juce::TextEditor::textColourId, juce::Colour(valueText));
    setColour(juce::TextEditor::outlineColourId, juce::Colours::transparentBlack);
}

juce::Font DeBleedLookAndFeel::font(float height, const juce::String& style)
{
    return juce::Font(juce::FontOptions("Helvetica Neue", height, juce::Font::plain).withStyle(style));
}

void DeBleedLookAndFeel::drawBody(juce::Graphics& g, juce::Point<float> centre,
                                 float radius, bool innerRing)
{
    const auto disc = juce::Rectangle<float>(radius * 2.0f, radius * 2.0f).withCentre(centre);
    g.setColour(juce::Colour(rim));
    g.fillEllipse(disc.expanded(1.0f));
    g.setGradientFill(juce::ColourGradient(juce::Colour(bodyCentre), centre.x, centre.y - radius * 0.2f,
                                         juce::Colour(bodyEdge), centre.x, centre.y + radius, true));
    g.fillEllipse(disc);
    if (innerRing)
    {
        g.setColour(juce::Colours::white.withAlpha(0.05f));
        g.drawEllipse(disc.reduced(0.5f), 1.0f);
    }
}

void DeBleedLookAndFeel::drawGlow(juce::Graphics& g, const juce::Path& path,
                                 const juce::PathStrokeType& stroke, juce::Colour colour, float alpha)
{
    juce::Path outline;
    stroke.createStrokedPath(outline, path);
    juce::DropShadow(colour.withAlpha(alpha), 4, {}).drawForPath(g, outline);
}

void DeBleedLookAndFeel::drawTextAtBaseline(juce::Graphics& g, const juce::String& text,
                                           const juce::Font& font, float x, float baseline,
                                           bool centred)
{
    juce::GlyphArrangement glyphs;
    glyphs.addLineOfText(font, text, 0.0f, 0.0f);
    const auto bounds = glyphs.getBoundingBox(0, glyphs.getNumGlyphs(), true);
    glyphs.draw(g, juce::AffineTransform::translation(
        x - (centred ? bounds.getCentreX() : bounds.getX()), baseline));
}

void DeBleedLookAndFeel::drawRotarySlider(juce::Graphics& g, int x, int y, int width, int height,
                                        float sliderPos, float rotaryStartAngle, float rotaryEndAngle,
                                        juce::Slider& slider)
{
    const auto centre = juce::Rectangle<float>(static_cast<float>(x), static_cast<float>(y),
                                              static_cast<float>(width), static_cast<float>(height)).getCentre();
    const bool big = static_cast<bool>(slider.getProperties()["bigKnob"]);
    const float radius = big ? 40.0f : 28.0f;
    const float stroke = big ? 2.5f : 2.0f;
    const float angle = rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
    const auto colour = juce::Colour(static_cast<juce::uint32>(static_cast<juce::int64>(
        slider.getProperties().getWithDefault("knobColor", static_cast<juce::int64>(cyanAccent)))));
    const auto roundStroke = [](float thickness)
    {
        return juce::PathStrokeType(thickness, juce::PathStrokeType::curved, juce::PathStrokeType::rounded);
    };

    juce::Path track;
    track.addCentredArc(centre.x, centre.y, radius, radius, 0.0f,
                       rotaryStartAngle, rotaryEndAngle, true);
    g.setColour(juce::Colour(arcBackground));
    g.strokePath(track, roundStroke(stroke));

    if (sliderPos > 0.002f)
    {
        juce::Path valueArc;
        valueArc.addCentredArc(centre.x, centre.y, radius, radius, 0.0f, rotaryStartAngle, angle, true);
        drawGlow(g, valueArc, roundStroke(stroke + 1.0f), colour, 0.7f);
        g.setColour(colour);
        g.strokePath(valueArc, roundStroke(stroke));
    }

    drawBody(g, centre, radius - 6.0f, true);
    juce::Path pointer;
    pointer.startNewSubPath(centre.x + (radius - 11.0f) * std::sin(angle),
                            centre.y - (radius - 11.0f) * std::cos(angle));
    pointer.lineTo(centre.x + (radius - 7.5f) * std::sin(angle),
                   centre.y - (radius - 7.5f) * std::cos(angle));
    g.setColour(colour);
    g.strokePath(pointer, roundStroke(1.6f));

    const auto number = slider.getTextFromValue(slider.getValue()).trimStart()
                             .initialSectionContainingOnly("+-0123456789.")
                             .trimCharactersAtStart("+");
    g.setColour(juce::Colour(valueText));
    drawTextAtBaseline(g, number, font(big ? 14.0f : 11.0f, "Medium"),
                       centre.x, centre.y + (big ? 5.0f : 4.0f));
}

void DeBleedLookAndFeel::drawToggleButton(juce::Graphics& g, juce::ToggleButton& button,
                                        bool highlighted, bool down)
{
    const auto centre = button.getLocalBounds().toFloat().getCentre();
    if (static_cast<bool>(button.getProperties()["chevron"]))
    {
        const float direction = static_cast<bool>(button.getProperties()["pointsLeft"]) ? 1.0f : -1.0f;
        juce::Path chevron;
        chevron.startNewSubPath(centre.x + direction * 2.5f, centre.y - 4.0f);
        chevron.lineTo(centre.x - direction * 1.5f, centre.y);
        chevron.lineTo(centre.x + direction * 2.5f, centre.y + 4.0f);
        const juce::PathStrokeType stroke(1.6f, juce::PathStrokeType::curved,
                                          juce::PathStrokeType::rounded);
        drawGlow(g, chevron, juce::PathStrokeType(2.6f, juce::PathStrokeType::curved,
                                                 juce::PathStrokeType::rounded),
                 juce::Colour(orangeAccent), 0.8f);
        g.setColour(juce::Colour(orangeAccent));
        g.strokePath(chevron, stroke);
        return;
    }

    if (static_cast<bool>(button.getProperties()["ringStyle"]))
    {
        const auto colour = juce::Colour(static_cast<juce::uint32>(static_cast<juce::int64>(
            button.getProperties()["ringColour"])));
        const auto ring = juce::Rectangle<float>(20.0f, 20.0f).withCentre(centre);
        const bool on = button.getToggleState();
        if (on)
        {
            juce::Path halo;
            halo.addEllipse(ring.expanded(1.0f));
            drawGlow(g, halo, juce::PathStrokeType(3.0f), colour, 0.8f);
        }
        drawBody(g, centre, 10.0f, false);
        g.setColour(on ? colour : juce::Colour(inactiveRing));
        g.drawEllipse(ring, 1.8f);
        g.setGradientFill(juce::ColourGradient(juce::Colour(inactiveRing), centre.x - 1.0f, centre.y - 1.5f,
                                             juce::Colour(meterBackground), centre.x, centre.y + 5.0f, true));
        g.fillEllipse(juce::Rectangle<float>(10.0f, 10.0f).withCentre(centre));
        return;
    }

    LookAndFeel_V4::drawToggleButton(g, button, highlighted, down);
}
