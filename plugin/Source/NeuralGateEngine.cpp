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

    // Pre-allocate input and output buffers
    inputBuffer.resize(N_FREQ_BINS * maxFrames, 0.0f);
    outputBuffer.resize(N_FREQ_BINS * maxFrames, 0.0f);

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
            lastError = "Invalid input shape: expected 3 dimensions [batch, freq_bins, frames]";
            session.reset();
            return false;
        }

        // inputShape should be [batch, freq_bins, frames] or with dynamic dims
        // The freq_bins dimension (index 1) should match N_FREQ_BINS
        int64_t freqBins = inputShape[1];
        if (freqBins > 0 && freqBins != N_FREQ_BINS)
        {
            lastError = juce::String::formatted(
                "Frequency bins mismatch: model has %lld, expected %d",
                freqBins, N_FREQ_BINS);
            session.reset();
            return false;
        }

        currentModelPath = modelPath;
        modelLoaded.store(true);
        lastError.clear();

        DBG("Model loaded successfully: " << modelPath);
        DBG("Input name: " << inputNameStr);
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

const float* NeuralGateEngine::process(const float* magnitude, int numFrames)
{
    // If no model loaded, return pass-through mask (all 1.0s)
    if (!modelLoaded.load())
    {
        // Fill output with 1.0 (pass-through)
        int numElements = std::min(numFrames * N_FREQ_BINS,
                                    static_cast<int>(outputBuffer.size()));
        std::fill(outputBuffer.begin(), outputBuffer.begin() + numElements, 1.0f);
        return outputBuffer.data();
    }

    // Check buffer size
    if (numFrames > maxFramesAllocated || numFrames <= 0)
    {
        // Return pass-through for invalid input
        std::fill(outputBuffer.begin(), outputBuffer.end(), 1.0f);
        return outputBuffer.data();
    }

    try
    {
        // Copy input data to our buffer
        int numElements = numFrames * N_FREQ_BINS;
        std::memcpy(inputBuffer.data(), magnitude, numElements * sizeof(float));

        // Create input tensor - model expects [batch, freq_bins, frames]
        std::vector<int64_t> inputDims = {1, N_FREQ_BINS, numFrames};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            *memoryInfo,
            inputBuffer.data(),
            numElements,
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

        // Copy output
        if (!outputTensors.empty() && outputTensors[0].IsTensor())
        {
            const float* outputData = outputTensors[0].GetTensorData<float>();
            std::memcpy(outputBuffer.data(), outputData, numElements * sizeof(float));
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
