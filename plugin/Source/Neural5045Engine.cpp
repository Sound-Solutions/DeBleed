#include "Neural5045Engine.h"

Neural5045Engine::Neural5045Engine()
{
    try
    {
        // Initialize ONNX Runtime environment
        env_ = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "Neural5045");

        // Create session options optimized for real-time
        sessionOptions_ = std::make_unique<Ort::SessionOptions>();
        sessionOptions_->SetIntraOpNumThreads(1);  // Single thread for real-time
        sessionOptions_->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Create memory info for CPU
        memoryInfo_ = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        // Initialize default bypass parameters
        initializeDefaultParams();
    }
    catch (const Ort::Exception& e)
    {
        lastError_ = juce::String("ONNX Runtime init failed: ") + e.what();
        DBG(lastError_);
    }
}

Neural5045Engine::~Neural5045Engine()
{
    unloadModel();
}

void Neural5045Engine::initializeDefaultParams()
{
    defaultParams_.resize(N_PARAMS, 0.5f);

    // Set default values for each parameter type
    // These produce minimal processing (near-unity gain, neutral EQ)

    for (int f = 0; f < N_FILTERS; ++f)
    {
        int baseIdx = f * N_PARAMS_PER_FILTER;

        // Frequency: 0.5 = middle of each filter's range (log scale)
        defaultParams_[baseIdx + 0] = 0.5f;

        // Gain: 0.5 = 0dB (neutral)
        defaultParams_[baseIdx + 1] = 0.5f;

        // Q: 0.5 = moderate Q
        defaultParams_[baseIdx + 2] = 0.5f;
    }

    // Input gain: 1.0 = 0dB (no attenuation)
    // In the biquad chain, input gain maps from [-60, 0]dB, so 1.0 = 0dB
    defaultParams_[N_FILTERS * N_PARAMS_PER_FILTER] = 1.0f;

    // Output gain: 1.0 = 0dB (no attenuation)
    defaultParams_[N_FILTERS * N_PARAMS_PER_FILTER + 1] = 1.0f;
}

void Neural5045Engine::prepare(double sampleRate, int maxBlockSize)
{
    currentSampleRate_ = sampleRate;

    // Calculate maximum frames needed
    maxSamplesAllocated_ = maxBlockSize;
    maxFramesAllocated_ = (maxBlockSize + FRAME_SIZE - 1) / FRAME_SIZE;  // Ceiling division

    // Pre-allocate input buffer: [1, 1, maxSamples] flattened
    inputBuffer_.resize(maxSamplesAllocated_, 0.0f);

    // Pre-allocate output buffer: [N_PARAMS × maxFrames]
    outputBuffer_.resize(N_PARAMS * maxFramesAllocated_, 0.5f);

    // Initialize output with default params
    for (int frame = 0; frame < maxFramesAllocated_; ++frame)
    {
        std::copy(defaultParams_.begin(), defaultParams_.end(),
                  outputBuffer_.begin() + frame * N_PARAMS);
    }

    numFramesOutput_ = 0;

    DBG("Neural5045Engine prepared: " << sampleRate << " Hz, max " << maxBlockSize
        << " samples (" << maxFramesAllocated_ << " frames)");
}

bool Neural5045Engine::loadModel(const juce::String& modelPath)
{
    std::lock_guard<std::mutex> lock(modelMutex_);

    try
    {
        // Check if file exists
        juce::File modelFile(modelPath);
        if (!modelFile.existsAsFile())
        {
            lastError_ = "Model file not found: " + modelPath;
            return false;
        }

        // Unload existing model
        if (session_)
        {
            session_.reset();
            modelLoaded_.store(false);
        }

        // Load new model
#ifdef _WIN32
        std::wstring wpath = modelPath.toWideCharPointer();
        session_ = std::make_unique<Ort::Session>(*env_, wpath.c_str(), *sessionOptions_);
#else
        session_ = std::make_unique<Ort::Session>(*env_, modelPath.toRawUTF8(), *sessionOptions_);
#endif

        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;

        // Input name (expected: "audio")
        auto inputNameAlloc = session_->GetInputNameAllocated(0, allocator);
        inputNameStr_ = inputNameAlloc.get();
        inputNames_ = {inputNameStr_.c_str()};

        // Output name (expected: "params")
        auto outputNameAlloc = session_->GetOutputNameAllocated(0, allocator);
        outputNameStr_ = outputNameAlloc.get();
        outputNames_ = {outputNameStr_.c_str()};

        // Get input shape info - expected [batch, 1, samples]
        auto inputTypeInfo = session_->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape_ = inputTensorInfo.GetShape();

        // Get output shape info - expected [batch, n_params, n_frames]
        auto outputTypeInfo = session_->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape_ = outputTensorInfo.GetShape();

        // Verify input shape
        if (inputShape_.size() != 3)
        {
            lastError_ = juce::String::formatted(
                "Invalid input shape: expected 3 dimensions [batch, channels, samples], got %zu",
                inputShape_.size());
            session_.reset();
            return false;
        }

        // Verify the model expects mono input (1 channel)
        if (inputShape_[1] > 0 && inputShape_[1] != 1)
        {
            lastError_ = juce::String::formatted(
                "Invalid input channels: expected 1 (mono), got %lld",
                inputShape_[1]);
            session_.reset();
            return false;
        }

        // Verify output shape
        if (outputShape_.size() != 3)
        {
            lastError_ = juce::String::formatted(
                "Invalid output shape: expected 3 dimensions [batch, params, frames], got %zu",
                outputShape_.size());
            session_.reset();
            return false;
        }

        // Verify output parameter count (should be 50 or dynamic)
        int64_t outputParams = outputShape_[1];
        if (outputParams > 0 && outputParams != N_PARAMS)
        {
            lastError_ = juce::String::formatted(
                "Output parameter count mismatch: model has %lld, expected %d",
                outputParams, N_PARAMS);
            session_.reset();
            return false;
        }

        currentModelPath_ = modelPath;
        modelLoaded_.store(true);
        lastError_.clear();

        DBG("Neural5045 model loaded: " << modelPath);
        DBG("Input: " << inputNameStr_ << " [batch, 1, samples]");
        DBG("Output: " << outputNameStr_ << " [batch, " << N_PARAMS << ", frames]");

        return true;
    }
    catch (const Ort::Exception& e)
    {
        lastError_ = juce::String("Failed to load Neural5045 model: ") + e.what();
        session_.reset();
        modelLoaded_.store(false);
        return false;
    }
}

void Neural5045Engine::unloadModel()
{
    std::lock_guard<std::mutex> lock(modelMutex_);

    modelLoaded_.store(false);
    session_.reset();
    currentModelPath_.clear();

    // Reset output to default params
    for (int frame = 0; frame < maxFramesAllocated_; ++frame)
    {
        std::copy(defaultParams_.begin(), defaultParams_.end(),
                  outputBuffer_.begin() + frame * N_PARAMS);
    }
    numFramesOutput_ = 0;
}

const float* Neural5045Engine::process(const float* audio, int numSamples)
{
    // Calculate number of frames for this block
    int numFrames = (numSamples + FRAME_SIZE - 1) / FRAME_SIZE;  // Ceiling division
    numFrames = std::max(1, numFrames);

    // If no model loaded, return default parameters
    if (!modelLoaded_.load())
    {
        // Fill output with default params
        for (int frame = 0; frame < numFrames && frame < maxFramesAllocated_; ++frame)
        {
            std::copy(defaultParams_.begin(), defaultParams_.end(),
                      outputBuffer_.begin() + frame * N_PARAMS);
        }
        numFramesOutput_ = numFrames;
        return outputBuffer_.data();
    }

    // Check buffer size
    if (numSamples > maxSamplesAllocated_ || numSamples <= 0)
    {
        DBG("Neural5045Engine: buffer size mismatch - " << numSamples
            << " samples (max: " << maxSamplesAllocated_ << ")");

        for (int frame = 0; frame < numFrames && frame < maxFramesAllocated_; ++frame)
        {
            std::copy(defaultParams_.begin(), defaultParams_.end(),
                      outputBuffer_.begin() + frame * N_PARAMS);
        }
        numFramesOutput_ = numFrames;
        return outputBuffer_.data();
    }

    // Try to acquire the lock without blocking (real-time safe)
    std::unique_lock<std::mutex> lock(modelMutex_, std::try_to_lock);
    if (!lock.owns_lock())
    {
        // Model is being loaded/unloaded, return default params
        for (int frame = 0; frame < numFrames && frame < maxFramesAllocated_; ++frame)
        {
            std::copy(defaultParams_.begin(), defaultParams_.end(),
                      outputBuffer_.begin() + frame * N_PARAMS);
        }
        numFramesOutput_ = numFrames;
        return outputBuffer_.data();
    }

    // Double-check model is still loaded after acquiring lock
    if (!session_ || !modelLoaded_.load())
    {
        for (int frame = 0; frame < numFrames && frame < maxFramesAllocated_; ++frame)
        {
            std::copy(defaultParams_.begin(), defaultParams_.end(),
                      outputBuffer_.begin() + frame * N_PARAMS);
        }
        numFramesOutput_ = numFrames;
        return outputBuffer_.data();
    }

    try
    {
        // Copy audio to input buffer
        std::memcpy(inputBuffer_.data(), audio, numSamples * sizeof(float));

        // Zero-pad if needed (for frame alignment)
        int paddedSamples = numFrames * FRAME_SIZE;
        if (paddedSamples > numSamples)
        {
            std::fill(inputBuffer_.begin() + numSamples,
                      inputBuffer_.begin() + std::min(paddedSamples, maxSamplesAllocated_),
                      0.0f);
        }

        // Create input tensor: [batch=1, channels=1, samples]
        std::vector<int64_t> inputDims = {1, 1, static_cast<int64_t>(paddedSamples)};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo_,
            inputBuffer_.data(),
            paddedSamples,
            inputDims.data(),
            inputDims.size()
        );

        // Run inference
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(inputTensor));

        auto outputTensors = session_->Run(
            Ort::RunOptions{nullptr},
            inputNames_.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNames_.data(),
            outputNames_.size()
        );

        // Copy output parameters
        // Output shape: [batch=1, n_params=50, n_frames]
        if (!outputTensors.empty() && outputTensors[0].IsTensor())
        {
            auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
            auto outputShape = outputInfo.GetShape();

            const float* outputData = outputTensors[0].GetTensorData<float>();

            // Model outputs [batch, params, frames] but we need [frames, params]
            // So we need to transpose while copying
            int modelFrames = static_cast<int>(outputShape[2]);
            int actualFrames = std::min(modelFrames, numFrames);
            actualFrames = std::min(actualFrames, maxFramesAllocated_);

            for (int frame = 0; frame < actualFrames; ++frame)
            {
                for (int param = 0; param < N_PARAMS; ++param)
                {
                    // Source: [batch=0, param, frame] = param * modelFrames + frame
                    // Dest: [frame * N_PARAMS + param]
                    float value = outputData[param * modelFrames + frame];

                    // Clamp to valid range [0, 1]
                    value = std::max(0.0f, std::min(1.0f, value));

                    outputBuffer_[frame * N_PARAMS + param] = value;
                }
            }

            numFramesOutput_ = actualFrames;
        }
        else
        {
            DBG("Neural5045Engine: empty output tensor");
            for (int frame = 0; frame < numFrames && frame < maxFramesAllocated_; ++frame)
            {
                std::copy(defaultParams_.begin(), defaultParams_.end(),
                          outputBuffer_.begin() + frame * N_PARAMS);
            }
            numFramesOutput_ = numFrames;
        }
    }
    catch (const Ort::Exception& e)
    {
        DBG("Neural5045 inference error: " << e.what());
        // On error, return default params
        for (int frame = 0; frame < numFrames && frame < maxFramesAllocated_; ++frame)
        {
            std::copy(defaultParams_.begin(), defaultParams_.end(),
                      outputBuffer_.begin() + frame * N_PARAMS);
        }
        numFramesOutput_ = numFrames;
    }

    return outputBuffer_.data();
}

const float* Neural5045Engine::getFrameParams(int frameIndex) const
{
    if (frameIndex < 0 || frameIndex >= numFramesOutput_ || frameIndex >= maxFramesAllocated_)
    {
        return defaultParams_.data();
    }

    return outputBuffer_.data() + frameIndex * N_PARAMS;
}
