#pragma once

#include <JuceHeader.h>
#include <functional>

/**
 * TrainerProcess - Manages the Python training subprocess.
 *
 * Launches the compiled Python trainer executable as a child process,
 * passes audio file paths as CLI arguments, and parses stdout for
 * progress updates.
 *
 * Output format from trainer:
 * - PROGRESS:XX (0-100)
 * - LOSS:X.XXXXX
 * - EPOCH:X/Y
 * - STATUS:message
 * - ERROR:message
 * - RESULT:SUCCESS/FAILURE
 * - MODEL_PATH:/path/to/model.onnx
 */
class TrainerProcess : public juce::Thread
{
public:
    /** Training state enumeration */
    enum class State
    {
        Idle,
        Preparing,
        Training,
        Exporting,
        Completed,
        Failed
    };

    /** Callback types */
    using ProgressCallback = std::function<void(int progress, const juce::String& status)>;
    using CompletionCallback = std::function<void(bool success, const juce::String& modelPath, const juce::String& error)>;
    using LogCallback = std::function<void(const juce::String& message)>;

    TrainerProcess();
    ~TrainerProcess() override;

    /**
     * Start training with the given audio directories.
     * @param cleanAudioDir Directory containing target vocal audio files
     * @param noiseAudioDir Directory containing bleed/noise audio files
     * @param outputDir Directory for output model and metadata
     * @param modelName Name for the output model file (without .onnx extension)
     * @param epochs Number of training epochs
     * @param continueFromCheckpoint If true, resume from checkpoint.pt in outputDir
     * @return true if training started successfully
     */
    bool startTraining(const juce::String& cleanAudioDir,
                       const juce::String& noiseAudioDir,
                       const juce::String& outputDir,
                       const juce::String& modelName = "model",
                       int epochs = 50,
                       bool continueFromCheckpoint = false);

    /**
     * Cancel the current training process.
     */
    void cancelTraining();

    /**
     * Check if training is currently in progress.
     */
    bool isTraining() const { return isThreadRunning(); }

    /**
     * Get the current training state.
     */
    State getState() const { return currentState.load(); }

    /**
     * Get the current progress (0-100).
     */
    int getProgress() const { return currentProgress.load(); }

    /**
     * Get the current status message.
     */
    juce::String getStatusMessage() const
    {
        juce::ScopedLock lock(statusLock);
        return statusMessage;
    }

    /**
     * Get the last error message.
     */
    juce::String getLastError() const
    {
        juce::ScopedLock lock(statusLock);
        return lastError;
    }

    /**
     * Get the path to the output model (valid after successful completion).
     */
    juce::String getModelPath() const
    {
        juce::ScopedLock lock(statusLock);
        return outputModelPath;
    }

    /**
     * Set the path to the Python trainer executable.
     * If not set, will look for bundled trainer or system Python.
     */
    void setTrainerPath(const juce::String& path) { trainerPath = path; }

    /**
     * Set callbacks for progress, completion, and logging.
     */
    void setProgressCallback(ProgressCallback callback) { progressCallback = std::move(callback); }
    void setCompletionCallback(CompletionCallback callback) { completionCallback = std::move(callback); }
    void setLogCallback(LogCallback callback) { logCallback = std::move(callback); }

private:
    void run() override;
    void parseLine(const juce::String& line);
    juce::String findTrainerExecutable();
    juce::StringArray buildCommandLine();

    // Process management
    std::unique_ptr<juce::ChildProcess> childProcess;

    // Training parameters
    juce::String cleanAudioDir;
    juce::String noiseAudioDir;
    juce::String outputDir;
    juce::String modelName;
    int epochs = 50;
    bool continueFromCheckpoint = false;

    // Trainer executable path
    juce::String trainerPath;

    // State
    std::atomic<State> currentState{State::Idle};
    std::atomic<int> currentProgress{0};

    mutable juce::CriticalSection statusLock;
    juce::String statusMessage;
    juce::String lastError;
    juce::String outputModelPath;

    // Callbacks
    ProgressCallback progressCallback;
    CompletionCallback completionCallback;
    LogCallback logCallback;

    // Cancel flag
    std::atomic<bool> shouldCancel{false};

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(TrainerProcess)
};
