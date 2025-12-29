#include "AudioDropZone.h"

AudioDropZone::AudioDropZone(const juce::String& label, bool selectDirectory)
    : labelText(label), selectDirectoryMode(selectDirectory)
{
}

void AudioDropZone::paint(juce::Graphics& g)
{
    auto bounds = getLocalBounds().toFloat();

    // Background
    juce::Colour bgColour = isDraggingOver
        ? juce::Colour(0xff3a5a7c)  // Highlighted when dragging
        : juce::Colour(0xff2a3a4c); // Normal dark background

    g.setColour(bgColour);
    g.fillRoundedRectangle(bounds, 8.0f);

    // Border
    juce::Colour borderColour = isDraggingOver
        ? juce::Colour(0xff5a9ad8)
        : (hasValidSelection() ? juce::Colour(0xff4a8a5c) : juce::Colour(0xff4a5a6c));

    g.setColour(borderColour);
    g.drawRoundedRectangle(bounds.reduced(1.0f), 8.0f, 2.0f);

    // Content
    auto contentBounds = bounds.reduced(12.0f);

    // Label at top
    g.setColour(juce::Colours::white.withAlpha(0.9f));
    g.setFont(juce::Font(14.0f, juce::Font::bold));
    g.drawText(labelText, contentBounds.removeFromTop(20.0f),
               juce::Justification::centredLeft, true);

    // Icon or selection info
    if (hasValidSelection())
    {
        // Show selected path
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::Font(12.0f));

        juce::File file(selectedPath);
        juce::String displayName = file.getFileName();
        if (displayName.length() > 30)
            displayName = displayName.substring(0, 27) + "...";

        g.drawText(displayName, contentBounds.removeFromTop(20.0f),
                   juce::Justification::centredLeft, true);

        // Show file count
        g.setColour(juce::Colour(0xff5a9a6c));
        g.drawText(juce::String(audioFileCount) + " audio files",
                   contentBounds.removeFromTop(16.0f),
                   juce::Justification::centredLeft, true);
    }
    else
    {
        // Show drop hint
        g.setColour(juce::Colours::white.withAlpha(0.5f));
        g.setFont(juce::Font(12.0f));

        juce::String hint = selectDirectoryMode
            ? "Drop folder here or click to browse"
            : "Drop audio files here or click to browse";

        g.drawText(hint, contentBounds, juce::Justification::centred, true);

        // Draw drop icon
        auto iconBounds = contentBounds.reduced(30.0f);
        g.setColour(juce::Colours::white.withAlpha(0.3f));

        // Simple folder/file icon
        auto iconRect = juce::Rectangle<float>(
            iconBounds.getCentreX() - 20.0f,
            iconBounds.getCentreY() - 15.0f,
            40.0f, 30.0f
        );

        g.drawRoundedRectangle(iconRect, 4.0f, 1.5f);

        // Folder tab
        if (selectDirectoryMode)
        {
            g.drawLine(iconRect.getX() + 5.0f, iconRect.getY(),
                      iconRect.getX() + 15.0f, iconRect.getY() - 5.0f);
            g.drawLine(iconRect.getX() + 15.0f, iconRect.getY() - 5.0f,
                      iconRect.getX() + 20.0f, iconRect.getY());
        }
    }
}

void AudioDropZone::resized()
{
    // Nothing specific to do here
}

void AudioDropZone::mouseDown(const juce::MouseEvent& /*event*/)
{
    openFileBrowser();
}

bool AudioDropZone::isInterestedInFileDrag(const juce::StringArray& files)
{
    if (files.isEmpty())
        return false;

    if (selectDirectoryMode)
    {
        // Accept directories or audio files
        for (const auto& path : files)
        {
            juce::File file(path);
            if (file.isDirectory() || isAudioFile(file))
                return true;
        }
    }
    else
    {
        // Accept only audio files
        for (const auto& path : files)
        {
            if (isAudioFile(juce::File(path)))
                return true;
        }
    }

    return false;
}

void AudioDropZone::fileDragEnter(const juce::StringArray& /*files*/, int /*x*/, int /*y*/)
{
    isDraggingOver = true;
    repaint();
}

void AudioDropZone::fileDragExit(const juce::StringArray& /*files*/)
{
    isDraggingOver = false;
    repaint();
}

void AudioDropZone::filesDropped(const juce::StringArray& files, int /*x*/, int /*y*/)
{
    isDraggingOver = false;

    if (files.isEmpty())
    {
        repaint();
        return;
    }

    if (selectDirectoryMode)
    {
        // Find first directory or parent of first file
        for (const auto& path : files)
        {
            juce::File file(path);
            if (file.isDirectory())
            {
                setSelectedPath(path);
                if (directorySelectedCallback)
                    directorySelectedCallback(path);
                break;
            }
            else if (isAudioFile(file))
            {
                // Use parent directory
                juce::String parentPath = file.getParentDirectory().getFullPathName();
                setSelectedPath(parentPath);
                if (directorySelectedCallback)
                    directorySelectedCallback(parentPath);
                break;
            }
        }
    }
    else
    {
        // Collect audio files
        juce::StringArray audioFiles;
        for (const auto& path : files)
        {
            juce::File file(path);
            if (isAudioFile(file))
                audioFiles.add(path);
        }

        if (!audioFiles.isEmpty())
        {
            setSelectedPath(audioFiles[0]);
            if (filesSelectedCallback)
                filesSelectedCallback(audioFiles);
        }
    }

    repaint();
}

void AudioDropZone::openFileBrowser()
{
    if (selectDirectoryMode)
    {
        auto chooser = std::make_shared<juce::FileChooser>(
            "Select " + labelText + " Folder",
            juce::File::getSpecialLocation(juce::File::userMusicDirectory),
            "*"
        );

        chooser->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectDirectories,
            [this, chooser](const juce::FileChooser& fc)
            {
                auto result = fc.getResult();
                if (result.isDirectory())
                {
                    setSelectedPath(result.getFullPathName());
                    if (directorySelectedCallback)
                        directorySelectedCallback(result.getFullPathName());
                }
            }
        );
    }
    else
    {
        juce::String wildcard;
        for (const auto& ext : audioExtensions)
            wildcard += "*" + ext + ";";

        auto chooser = std::make_shared<juce::FileChooser>(
            "Select " + labelText + " Files",
            juce::File::getSpecialLocation(juce::File::userMusicDirectory),
            wildcard
        );

        chooser->launchAsync(
            juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles |
            juce::FileBrowserComponent::canSelectMultipleItems,
            [this, chooser](const juce::FileChooser& fc)
            {
                auto results = fc.getResults();
                if (!results.isEmpty())
                {
                    juce::StringArray paths;
                    for (const auto& file : results)
                        paths.add(file.getFullPathName());

                    setSelectedPath(paths[0]);
                    if (filesSelectedCallback)
                        filesSelectedCallback(paths);
                }
            }
        );
    }
}

void AudioDropZone::setSelectedPath(const juce::String& path)
{
    selectedPath = path;
    updateAudioFileCount();
    repaint();
}

void AudioDropZone::clearSelection()
{
    selectedPath.clear();
    audioFileCount = 0;
    repaint();
}

void AudioDropZone::updateAudioFileCount()
{
    audioFileCount = 0;

    if (selectedPath.isEmpty())
        return;

    juce::File file(selectedPath);

    if (file.isDirectory())
    {
        // Count audio files in directory
        for (const auto& ext : audioExtensions)
        {
            auto files = file.findChildFiles(juce::File::findFiles, false, "*" + ext);
            audioFileCount += files.size();

            // Also check uppercase
            auto filesUpper = file.findChildFiles(juce::File::findFiles, false, "*" + ext.toUpperCase());
            audioFileCount += filesUpper.size();
        }
    }
    else if (isAudioFile(file))
    {
        audioFileCount = 1;
    }
}

bool AudioDropZone::isAudioFile(const juce::File& file) const
{
    if (!file.existsAsFile())
        return false;

    juce::String ext = file.getFileExtension().toLowerCase();
    return audioExtensions.contains(ext);
}
