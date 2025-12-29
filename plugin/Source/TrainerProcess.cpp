#include "TrainerProcess.h"

TrainerProcess::TrainerProcess()
    : Thread("DeBleed Trainer")
{
}

TrainerProcess::~TrainerProcess()
{
    cancelTraining();
    stopThread(5000);
}

bool TrainerProcess::startTraining(const juce::String& cleanDir,
                                    const juce::String& noiseDir,
                                    const juce::String& outDir,
                                    int numEpochs)
{
    if (isThreadRunning())
    {
        return false;  // Already training
    }

    // Store parameters
    cleanAudioDir = cleanDir;
    noiseAudioDir = noiseDir;
    outputDir = outDir;
    epochs = numEpochs;

    // Validate directories
    juce::File cleanFile(cleanDir);
    juce::File noiseFile(noiseDir);

    if (!cleanFile.isDirectory())
    {
        juce::ScopedLock lock(statusLock);
        lastError = "Clean audio directory not found: " + cleanDir;
        return false;
    }

    if (!noiseFile.isDirectory())
    {
        juce::ScopedLock lock(statusLock);
        lastError = "Noise audio directory not found: " + noiseDir;
        return false;
    }

    // Create output directory if needed
    juce::File outFile(outDir);
    if (!outFile.exists())
    {
        outFile.createDirectory();
    }

    // Reset state
    shouldCancel.store(false);
    currentProgress.store(0);
    currentState.store(State::Preparing);

    {
        juce::ScopedLock lock(statusLock);
        statusMessage = "Preparing to train...";
        lastError.clear();
        outputModelPath.clear();
    }

    // Start the thread
    startThread();

    return true;
}

void TrainerProcess::cancelTraining()
{
    shouldCancel.store(true);

    if (childProcess && childProcess->isRunning())
    {
        childProcess->kill();
    }

    signalThreadShouldExit();
}

juce::String TrainerProcess::findTrainerExecutable()
{
    // First, check if a custom path was set
    if (trainerPath.isNotEmpty())
    {
        juce::File f(trainerPath);
        if (f.existsAsFile())
            return trainerPath;
    }

    // Look for bundled trainer next to the plugin
    juce::File pluginDir = juce::File::getSpecialLocation(
        juce::File::currentExecutableFile).getParentDirectory();

    // Check various locations
    juce::StringArray searchPaths = {
        pluginDir.getChildFile("trainer").getFullPathName(),
        pluginDir.getChildFile("trainer.exe").getFullPathName(),
        pluginDir.getChildFile("Resources/trainer").getFullPathName(),
        pluginDir.getParentDirectory().getChildFile("Resources/trainer").getFullPathName(),
        pluginDir.getChildFile("../python/trainer.py").getFullPathName(),
    };

    for (const auto& path : searchPaths)
    {
        juce::File f(path);
        if (f.existsAsFile())
            return path;
    }

    // Fall back to system Python with trainer.py
    // Try to find the trainer.py script
    juce::File scriptFile = pluginDir.getParentDirectory()
                                      .getChildFile("python")
                                      .getChildFile("trainer.py");

    if (scriptFile.existsAsFile())
    {
        return scriptFile.getFullPathName();
    }

    return {};
}

juce::StringArray TrainerProcess::buildCommandLine()
{
    juce::StringArray args;

    juce::String executable = findTrainerExecutable();

    if (executable.isEmpty())
    {
        return args;  // Empty = error
    }

    // Check if it's a Python script or compiled executable
    if (executable.endsWith(".py"))
    {
        // Use Python interpreter
#ifdef _WIN32
        args.add("python");
#else
        args.add("python3");
#endif
        args.add(executable);
    }
    else
    {
        // Compiled executable
        args.add(executable);
    }

    // Add arguments
    args.add("--clean_audio_dir");
    args.add(cleanAudioDir);

    args.add("--noise_audio_dir");
    args.add(noiseAudioDir);

    args.add("--output_path");
    args.add(outputDir);

    args.add("--epochs");
    args.add(juce::String(epochs));

    return args;
}

void TrainerProcess::run()
{
    currentState.store(State::Preparing);

    // Build command line
    juce::StringArray args = buildCommandLine();

    if (args.isEmpty())
    {
        juce::ScopedLock lock(statusLock);
        lastError = "Could not find trainer executable";
        currentState.store(State::Failed);

        if (completionCallback)
        {
            juce::MessageManager::callAsync([this]()
            {
                if (completionCallback)
                    completionCallback(false, {}, lastError);
            });
        }
        return;
    }

    // Log the command
    juce::String cmdLine = args.joinIntoString(" ");
    DBG("Starting trainer: " << cmdLine);

    if (logCallback)
    {
        juce::MessageManager::callAsync([this, cmdLine]()
        {
            if (logCallback)
                logCallback("Starting: " + cmdLine);
        });
    }

    // Create and start child process
    childProcess = std::make_unique<juce::ChildProcess>();

    if (!childProcess->start(args))
    {
        juce::ScopedLock lock(statusLock);
        lastError = "Failed to start trainer process";
        currentState.store(State::Failed);

        if (completionCallback)
        {
            juce::MessageManager::callAsync([this]()
            {
                if (completionCallback)
                    completionCallback(false, {}, lastError);
            });
        }
        return;
    }

    currentState.store(State::Training);

    // Read stdout line by line
    juce::String lineBuffer;

    while (!threadShouldExit() && !shouldCancel.load() && childProcess->isRunning())
    {
        // Read available output
        char buffer[256];
        int bytesRead = childProcess->readProcessOutput(buffer, sizeof(buffer) - 1);

        if (bytesRead > 0)
        {
            buffer[bytesRead] = '\0';
            lineBuffer += juce::String::fromUTF8(buffer, bytesRead);

            // Process complete lines
            int newlinePos;
            while ((newlinePos = lineBuffer.indexOf("\n")) >= 0)
            {
                juce::String line = lineBuffer.substring(0, newlinePos).trim();
                lineBuffer = lineBuffer.substring(newlinePos + 1);

                if (line.isNotEmpty())
                {
                    parseLine(line);
                }
            }
        }
        else
        {
            // No data available, wait a bit
            Thread::sleep(50);
        }
    }

    // Process any remaining data
    if (lineBuffer.isNotEmpty())
    {
        parseLine(lineBuffer.trim());
    }

    // Wait for process to finish
    int exitCode = 0;
    if (childProcess->isRunning())
    {
        if (shouldCancel.load())
        {
            childProcess->kill();
        }
        else
        {
            childProcess->waitForProcessToFinish(30000);
        }
    }
    exitCode = static_cast<int>(childProcess->getExitCode());

    // Determine final state
    bool success = false;
    juce::String modelPath;

    {
        juce::ScopedLock lock(statusLock);
        modelPath = outputModelPath;

        if (shouldCancel.load())
        {
            currentState.store(State::Failed);
            lastError = "Training cancelled";
        }
        else if (exitCode == 0 && modelPath.isNotEmpty())
        {
            juce::File modelFile(modelPath);
            if (modelFile.existsAsFile())
            {
                currentState.store(State::Completed);
                success = true;
            }
            else
            {
                currentState.store(State::Failed);
                lastError = "Model file not created";
            }
        }
        else
        {
            currentState.store(State::Failed);
            if (lastError.isEmpty())
                lastError = "Training failed with exit code " + juce::String(exitCode);
        }
    }

    // Call completion callback on message thread
    if (completionCallback)
    {
        juce::String error;
        {
            juce::ScopedLock lock(statusLock);
            error = lastError;
        }

        juce::MessageManager::callAsync([this, success, modelPath, error]()
        {
            if (completionCallback)
                completionCallback(success, modelPath, error);
        });
    }

    childProcess.reset();
}

void TrainerProcess::parseLine(const juce::String& line)
{
    DBG("Trainer: " << line);

    // Log all output
    if (logCallback)
    {
        juce::MessageManager::callAsync([this, line]()
        {
            if (logCallback)
                logCallback(line);
        });
    }

    // Parse known patterns
    if (line.startsWith("PROGRESS:"))
    {
        int progress = line.substring(9).getIntValue();
        progress = juce::jlimit(0, 100, progress);
        currentProgress.store(progress);

        if (progressCallback)
        {
            juce::String status;
            {
                juce::ScopedLock lock(statusLock);
                status = statusMessage;
            }

            juce::MessageManager::callAsync([this, progress, status]()
            {
                if (progressCallback)
                    progressCallback(progress, status);
            });
        }
    }
    else if (line.startsWith("LOSS:"))
    {
        float loss = line.substring(5).getFloatValue();
        juce::ScopedLock lock(statusLock);
        statusMessage = juce::String::formatted("Loss: %.6f", loss);
    }
    else if (line.startsWith("EPOCH:"))
    {
        juce::ScopedLock lock(statusLock);
        statusMessage = "Epoch " + line.substring(6);
    }
    else if (line.startsWith("STATUS:"))
    {
        juce::ScopedLock lock(statusLock);
        statusMessage = line.substring(7);

        // Check for specific states
        if (statusMessage.containsIgnoreCase("export"))
        {
            currentState.store(State::Exporting);
        }
    }
    else if (line.startsWith("ERROR:"))
    {
        juce::ScopedLock lock(statusLock);
        lastError = line.substring(6);
    }
    else if (line.startsWith("MODEL_PATH:"))
    {
        juce::ScopedLock lock(statusLock);
        outputModelPath = line.substring(11);
    }
    else if (line.startsWith("RESULT:"))
    {
        juce::String result = line.substring(7).trim();
        if (result == "SUCCESS")
        {
            currentState.store(State::Completed);
            currentProgress.store(100);
        }
        else
        {
            currentState.store(State::Failed);
        }
    }
}
