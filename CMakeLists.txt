cmake_minimum_required(VERSION 3.22)

project(DeBleed VERSION 1.0.0 LANGUAGES CXX)

# Build the plugin
add_subdirectory(plugin)

# Create a custom target to bundle the Python trainer
add_custom_target(bundle_trainer
    COMMAND ${CMAKE_COMMAND} -E echo "Bundling Python trainer..."
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/python
        ${CMAKE_BINARY_DIR}/python
    COMMENT "Copying Python trainer files"
)

# Add dependency so trainer is bundled before plugin
add_dependencies(DeBleed bundle_trainer)
