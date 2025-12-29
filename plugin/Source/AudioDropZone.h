#pragma once

#include <JuceHeader.h>

/**
 * AudioDropZone - A drag-and-drop component for selecting audio files/folders.
 *
 * Provides a visual drop target for users to drag audio files or folders.
 * Can also be clicked to open a file browser.
 */
class AudioDropZone : public juce::Component,
                      public juce::FileDragAndDropTarget
{
public:
    /** Callback when files are selected */
    using FilesSelectedCallback = std::function<void(const juce::StringArray& files)>;
    using DirectorySelectedCallback = std::function<void(const juce::String& directory)>;

    AudioDropZone(const juce::String& label, bool selectDirectory = true);
    ~AudioDropZone() override = default;

    // Component overrides
    void paint(juce::Graphics& g) override;
    void resized() override;
    void mouseDown(const juce::MouseEvent& event) override;

    // FileDragAndDropTarget overrides
    bool isInterestedInFileDrag(const juce::StringArray& files) override;
    void fileDragEnter(const juce::StringArray& files, int x, int y) override;
    void fileDragExit(const juce::StringArray& files) override;
    void filesDropped(const juce::StringArray& files, int x, int y) override;

    /** Set callback for when files/directory are selected */
    void setFilesSelectedCallback(FilesSelectedCallback callback) { filesSelectedCallback = std::move(callback); }
    void setDirectorySelectedCallback(DirectorySelectedCallback callback) { directorySelectedCallback = std::move(callback); }

    /** Get the currently selected path */
    juce::String getSelectedPath() const { return selectedPath; }

    /** Set the selected path programmatically */
    void setSelectedPath(const juce::String& path);

    /** Check if a valid path is selected */
    bool hasValidSelection() const { return selectedPath.isNotEmpty(); }

    /** Clear the selection */
    void clearSelection();

    /** Get the number of audio files found in the selection */
    int getAudioFileCount() const { return audioFileCount; }

private:
    void openFileBrowser();
    void updateAudioFileCount();
    bool isAudioFile(const juce::File& file) const;

    juce::String labelText;
    juce::String selectedPath;
    bool isDraggingOver = false;
    bool selectDirectoryMode;
    int audioFileCount = 0;

    FilesSelectedCallback filesSelectedCallback;
    DirectorySelectedCallback directorySelectedCallback;

    // Supported audio extensions
    const juce::StringArray audioExtensions = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioDropZone)
};
