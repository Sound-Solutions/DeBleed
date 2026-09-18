#pragma once
#include <JuceHeader.h>

// Debug-only probe (2026-09-18): logs every left mouse-down in the host process at the AppKit
// level, naming the view chain under the click, so a click the plugin never receives can be
// traced to whatever caught it. Does nothing in release builds.
void installClickMonitor (std::function<void (const juce::String&)> log);
