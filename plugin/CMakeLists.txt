cmake_minimum_required(VERSION 3.22)

project(DeBleed VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(DEBLEED_BUILD_STANDALONE "Build standalone application" ON)
option(DEBLEED_BUILD_VST3 "Build VST3 plugin" ON)
option(DEBLEED_BUILD_AU "Build AU plugin (macOS only)" ON)

# =============================================================================
# JUCE
# =============================================================================

# Option 1: Use local JUCE installation (set JUCE_DIR to your JUCE path)
# Option 2: Auto-download JUCE (for portability)

set(JUCE_DIR "" CACHE PATH "Path to local JUCE installation (leave empty to auto-download)")

if(JUCE_DIR)
    # Use local JUCE
    message(STATUS "Using local JUCE from: ${JUCE_DIR}")
    add_subdirectory(${JUCE_DIR} ${CMAKE_BINARY_DIR}/JUCE)
else()
    # Auto-download JUCE for portability
    message(STATUS "JUCE_DIR not set, downloading JUCE 7.0.9...")
    include(FetchContent)

    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 7.0.9
        GIT_SHALLOW TRUE
    )

    FetchContent_MakeAvailable(JUCE)
endif()

# =============================================================================
# ONNX Runtime
# =============================================================================

# ONNX Runtime can be provided via:
# 1. System installation (preferred for distribution)
# 2. Downloaded manually to a specified path
# 3. Auto-download (for development)

set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to ONNX Runtime installation")

if(ONNXRUNTIME_ROOT)
    # Use provided path
    set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
    if(APPLE)
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
    elseif(WIN32)
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
    else()
        set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
    endif()
else()
    # Try to find system installation
    find_path(ONNXRUNTIME_INCLUDE_DIR
        NAMES onnxruntime_cxx_api.h
        PATHS
            /usr/local/include/onnxruntime
            /opt/homebrew/include/onnxruntime
            $ENV{HOME}/onnxruntime/include
    )

    find_library(ONNXRUNTIME_LIB
        NAMES onnxruntime
        PATHS
            /usr/local/lib
            /opt/homebrew/lib
            $ENV{HOME}/onnxruntime/lib
    )
endif()

if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
    message(STATUS "ONNX Runtime not found. Downloading...")

    # Determine platform
    if(APPLE)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
            set(ORT_PLATFORM "osx-arm64")
        else()
            set(ORT_PLATFORM "osx-x86_64")
        endif()
        set(ORT_EXT "tgz")
    elseif(WIN32)
        set(ORT_PLATFORM "win-x64")
        set(ORT_EXT "zip")
    else()
        set(ORT_PLATFORM "linux-x64")
        set(ORT_EXT "tgz")
    endif()

    set(ORT_VERSION "1.16.3")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-${ORT_PLATFORM}-${ORT_VERSION}.${ORT_EXT}")

    FetchContent_Declare(
        onnxruntime
        URL ${ORT_URL}
    )
    FetchContent_MakeAvailable(onnxruntime)

    set(ONNXRUNTIME_INCLUDE_DIR "${onnxruntime_SOURCE_DIR}/include")
    if(APPLE)
        set(ONNXRUNTIME_LIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.dylib")
    elseif(WIN32)
        set(ONNXRUNTIME_LIB "${onnxruntime_SOURCE_DIR}/lib/onnxruntime.lib")
    else()
        set(ONNXRUNTIME_LIB "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime.so")
    endif()
endif()

message(STATUS "ONNX Runtime include: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime library: ${ONNXRUNTIME_LIB}")

# Create imported target for ONNX Runtime
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRUNTIME_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
)

if(APPLE)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_SONAME "@rpath/libonnxruntime.dylib"
    )
endif()

# =============================================================================
# Plugin Target
# =============================================================================

set(PLUGIN_SOURCES
    Source/PluginProcessor.cpp
    Source/PluginProcessor.h
    Source/PluginEditor.cpp
    Source/PluginEditor.h
    Source/ControlPanel.cpp
    Source/ControlPanel.h
    Source/DeBleedLookAndFeel.cpp
    Source/DeBleedLookAndFeel.h
    Source/EQCurveDisplay.cpp
    Source/EQCurveDisplay.h
    Source/DifferentiableBiquadChain.cpp
    Source/DifferentiableBiquadChain.h
    Source/Neural5045Engine.cpp
    Source/Neural5045Engine.h
    Source/NeuralGateEngine.cpp
    Source/NeuralGateEngine.h
    Source/STFTProcessor.cpp
    Source/STFTProcessor.h
    Source/TrainerProcess.cpp
    Source/TrainerProcess.h
    Source/AudioDropZone.cpp
    Source/AudioDropZone.h
    Source/EnvelopeFollower.h
)

# Determine formats to build
set(PLUGIN_FORMATS)
if(DEBLEED_BUILD_VST3)
    list(APPEND PLUGIN_FORMATS VST3)
endif()
if(DEBLEED_BUILD_AU AND APPLE)
    list(APPEND PLUGIN_FORMATS AU)
endif()
if(DEBLEED_BUILD_STANDALONE)
    list(APPEND PLUGIN_FORMATS Standalone)
endif()

juce_add_plugin(DeBleed
    COMPANY_NAME "DeBleed Audio"
    PLUGIN_MANUFACTURER_CODE Dbld
    PLUGIN_CODE Dbld
    FORMATS ${PLUGIN_FORMATS}
    PRODUCT_NAME "DeBleed"

    IS_SYNTH FALSE
    NEEDS_MIDI_INPUT FALSE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE

    EDITOR_WANTS_KEYBOARD_FOCUS FALSE

    COPY_PLUGIN_AFTER_BUILD TRUE

    VST3_CATEGORIES "Fx" "Dynamics"
    AU_MAIN_TYPE "kAudioUnitType_Effect"
)

target_sources(DeBleed PRIVATE ${PLUGIN_SOURCES})

target_include_directories(DeBleed PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/Source
    ${ONNXRUNTIME_INCLUDE_DIR}
)

target_compile_definitions(DeBleed PUBLIC
    JUCE_WEB_BROWSER=0
    JUCE_USE_CURL=0
    JUCE_VST3_CAN_REPLACE_VST2=0
    JUCE_DISPLAY_SPLASH_SCREEN=0
)

target_link_libraries(DeBleed
    PRIVATE
        juce::juce_audio_utils
        juce::juce_dsp
        onnxruntime
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# Copy ONNX Runtime library to output directory
if(APPLE)
    add_custom_command(TARGET DeBleed POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ONNXRUNTIME_LIB}"
            "$<TARGET_FILE_DIR:DeBleed>/../Frameworks/"
        COMMENT "Copying ONNX Runtime library"
    )
endif()

# =============================================================================
# Installation
# =============================================================================

install(TARGETS DeBleed
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
