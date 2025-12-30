#include "NeuralGateEngine.h"

NeuralGateEngine::NeuralGateEngine()
{
    try
    {
        // Initialize ONNX Runtime environment
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "DeBleed");

        // Create session options
        sessionOptions = std::make_unique<Ort::SessionOptions>();
        sessionOptions->SetIntraOpNumThreads(1);  // Single thread for real-time
        sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        // Create memory info for CPU
        memoryInfo = std::make_unique<Ort::MemoryInfo>(
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));
    }
    catch (const Ort::Exception& e)
    {
        lastError = juce::String("ONNX Runtime init failed: ") + e.what();
        DBG(lastError);
    }
}

NeuralGateEngine::~NeuralGateEngine()
{
    unloadModel();
}

void NeuralGateEngine::prepare(int maxFrames)
{
    maxFramesAllocated = maxFrames;

    // Pre-allocate input buffer for maximum input size (dual-stream: 257 features)
    inputBuffer.resize(N_MAX_INPUT_FEATURES * maxFrames, 0.0f);

    // Output buffer is always 129 bins (mask for Stream A)
    outputBuffer.resize(N_OUTPUT_BINS * maxFrames, 0.0f);

    // Initialize output to pass-through (mask = 1.0)
    std::fill(outputBuffer.begin(), outputBuffer.end(), 1.0f);
}

bool NeuralGateEngine::loadModel(const juce::String& modelPath)
{
    std::lock_guard<std::mutex> lock(modelMutex);

    try
    {
        // Check if file exists
        juce::File modelFile(modelPath);
        if (!modelFile.existsAsFile())
        {
            lastError = "Model file not found: " + modelPath;
            return false;
        }

        // Unload existing model
        if (session)
        {
            session.reset();
            modelLoaded.store(false);
        }

        // Load new model
#ifdef _WIN32
        std::wstring wpath = modelPath.toWideCharPointer();
        session = std::make_unique<Ort::Session>(*env, wpath.c_str(), *sessionOptions);
#else
        session = std::make_unique<Ort::Session>(*env, modelPath.toRawUTF8(), *sessionOptions);
#endif

        // Get input info
        Ort::AllocatorWithDefaultOptions allocator;

        // Input name
        auto inputNameAlloc = session->GetInputNameAllocated(0, allocator);
        inputNameStr = inputNameAlloc.get();
        inputNames = {inputNameStr.c_str()};

        // Output name
        auto outputNameAlloc = session->GetOutputNameAllocated(0, allocator);
        outputNameStr = outputNameAlloc.get();
        outputNames = {outputNameStr.c_str()};

        // Get input shape info
        auto inputTypeInfo = session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
        inputShape = inputTensorInfo.GetShape();

        // Get output shape info
        auto outputTypeInfo = session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
        outputShape = outputTensorInfo.GetShape();

        // Verify shapes match expected
        if (inputShape.size() != 3)
        {
            lastError = "Invalid input shape: expected 3 dimensions [batch, input_features, frames]";
            session.reset();
            return false;
        }

        // inputShape should be [batch, input_features, frames] or with dynamic dims
        // input_features is either 129 (single-stream) or 257 (dual-stream)
        int64_t inputFeatures = inputShape[1];
        if (inputFeatures > 0)
        {
            if (inputFeatures == N_OUTPUT_BINS)
            {
                modelInputFeatures = N_OUTPUT_BINS;  // Single-stream (legacy)
                DBG("Loaded single-stream model (129 input features)");
            }
            else if (inputFeatures == N_MAX_INPUT_FEATURES)
            {
                modelInputFeatures = N_MAX_INPUT_FEATURES;  // Dual-stream
                DBG("Loaded dual-stream model (257 input features)");
            }
            else
            {
                lastError = juce::String::formatted(
                    "Unexpected input features: model has %lld, expected 129 or 257",
                    inputFeatures);
                session.reset();
                return false;
            }
        }
        else
        {
            // Dynamic dimension - assume dual-stream
            modelInputFeatures = N_MAX_INPUT_FEATURES;
            DBG("Model has dynamic input dimension, assuming dual-stream (257)");
        }

        // Verify output shape - should always be 129 bins
        int64_t outputBins = outputShape.size() >= 2 ? outputShape[1] : -1;
        if (outputBins > 0 && outputBins != N_OUTPUT_BINS)
        {
            lastError = juce::String::formatted(
                "Output bins mismatch: model has %lld, expected %d",
                outputBins, N_OUTPUT_BINS);
            session.reset();
            return false;
        }

        currentModelPath = modelPath;
        modelLoaded.store(true);
        lastError.clear();

        DBG("Model loaded successfully: " << modelPath);
        DBG("Input name: " << inputNameStr << " (" << modelInputFeatures << " features)");
        DBG("Output name: " << outputNameStr);

        return true;
    }
    catch (const Ort::Exception& e)
    {
        lastError = juce::String("Failed to load model: ") + e.what();
        session.reset();
        modelLoaded.store(false);
        return false;
    }
}

void NeuralGateEngine::unloadModel()
{
    std::lock_guard<std::mutex> lock(modelMutex);

    modelLoaded.store(false);
    session.reset();
    currentModelPath.clear();

    // Reset output to pass-through
    std::fill(outputBuffer.begin(), outputBuffer.end(), 1.0f);
}

const float* NeuralGateEngine::process(const float* inputFeatures, int numFrames)
{
    // Calculate number of output elements for pass-through
    int outputElements = std::min(numFrames * N_OUTPUT_BINS,
                                  static_cast<int>(outputBuffer.size()));

    // If no model loaded, return pass-through mask (all 1.0s)
    if (!modelLoaded.load())
    {
        std::fill(outputBuffer.begin(), outputBuffer.begin() + outputElements, 1.0f);
        return outputBuffer.data();
    }

    // Check buffer size
    if (numFrames > maxFramesAllocated || numFrames <= 0)
    {
        std::fill(outputBuffer.begin(), outputBuffer.end(), 1.0f);
        return outputBuffer.data();
    }

    // Try to acquire the lock without blocking (real-time safe)
    // If we can't get the lock, the model is being swapped - return pass-through
    std::unique_lock<std::mutex> lock(modelMutex, std::try_to_lock);
    if (!lock.owns_lock())
    {
        // Model is being loaded/unloaded, return pass-through
        std::fill(outputBuffer.begin(), outputBuffer.begin() + outputElements, 1.0f);
        return outputBuffer.data();
    }

    // Double-check model is still loaded after acquiring lock
    if (!session || !modelLoaded.load())
    {
        std::fill(outputBuffer.begin(), outputBuffer.begin() + outputElements, 1.0f);
        return outputBuffer.data();
    }

    try
    {
        // Copy input data to our buffer
        // Use the model's expected input size (129 for single-stream, 257 for dual-stream)
        int inputElements = numFrames * modelInputFeatures;
        std::memcpy(inputBuffer.data(), inputFeatures, inputElements * sizeof(float));

        // Create input tensor - model expects [batch, input_features, frames]
        std::vector<int64_t> inputDims = {1, static_cast<int64_t>(modelInputFeatures), static_cast<int64_t>(numFrames)};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo,
            inputBuffer.data(),
            inputElements,
            inputDims.data(),
            inputDims.size()
        );

        // Run inference
        std::vector<Ort::Value> inputTensors;
        inputTensors.push_back(std::move(inputTensor));

        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr},
            inputNames.data(),
            inputTensors.data(),
            inputTensors.size(),
            outputNames.data(),
            outputNames.size()
        );

        // Copy output (always N_OUTPUT_BINS = 129 bins)
        if (!outputTensors.empty() && outputTensors[0].IsTensor())
        {
            const float* outputData = outputTensors[0].GetTensorData<float>();
            std::memcpy(outputBuffer.data(), outputData, outputElements * sizeof(float));
        }
    }
    catch (const Ort::Exception& e)
    {
        DBG("Inference error: " << e.what());
        // On error, return pass-through
        std::fill(outputBuffer.begin(), outputBuffer.end(), 1.0f);
    }

    return outputBuffer.data();
}
