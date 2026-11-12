cmake_minimum_required(VERSION 3.22)

#
# This file is generated only once,
# and is not re-generated if converter is called multiple times.
#
# User is free to modify the file as much as necessary
#

# Setup compiler settings
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Define the build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

message("Build type: " ${CMAKE_BUILD_TYPE})

# Enable compile command to ease indexing with e.g. clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

# Compiler options
set(STM32_MCU_FLAGS  "-mcpu=cortex-m7 -mfpu=fpv5-d16 -mfloat-abi=hard " )

#Linker options
set(STM32_LINKER_SCRIPT stm32h755xx_flash_CM7.ld)
set(STM32_LINKER_OPTION  )

# Set the project name
set(CMAKE_PROJECT_NAME coe_firmware_CM7)
project(${CMAKE_PROJECT_NAME})

# Create an executable object type
add_executable(${CMAKE_PROJECT_NAME})

# Include mx-generated
include("mx-generated.cmake")

# NUCLEO-H755ZI-Q default HSE: 8 MHz from the ST-LINK MCO (UM2408), not the
# 25-MHz HAL template value. Applies to every source, including STM32_Drivers.
target_compile_definitions(stm32cubemx INTERFACE HSE_VALUE=8000000UL)

# Firmware mode. The root ExternalProject does not forward -D options, so the
# selection lives in firmware_mode.cmake. COE_FIRMWARE_MODE_OVERRIDE is only
# for configuring this CM7 project directly (for example, separate check builds).
include("firmware_mode.cmake")
if(DEFINED COE_FIRMWARE_MODE_OVERRIDE)
    set(COE_FIRMWARE_MODE ${COE_FIRMWARE_MODE_OVERRIDE})
endif()
# Numeric values must match App/config/firmware_mode.h.
if(COE_FIRMWARE_MODE STREQUAL "REPLAY")
    set(COE_FIRMWARE_MODE_VALUE 0)
elseif(COE_FIRMWARE_MODE STREQUAL "LIVE_RAW")
    set(COE_FIRMWARE_MODE_VALUE 1)
elseif(COE_FIRMWARE_MODE STREQUAL "LIVE")
    set(COE_FIRMWARE_MODE_VALUE 2)
else()
    message(FATAL_ERROR "Unknown COE_FIRMWARE_MODE '${COE_FIRMWARE_MODE}'; use REPLAY, LIVE_RAW or LIVE")
endif()
message("COE firmware mode: ${COE_FIRMWARE_MODE}")


# In order to use CMake for cross-compiling
set(CMAKE_CXX_COMPILER_FORCED "true")

# Add project symbols (macros)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined symbols
    COE_FIRMWARE_MODE=${COE_FIRMWARE_MODE_VALUE}
)

# Enable CMake support for ASM and C languages
enable_language(C ASM)

# The ESP32 firmware sits next to this project: "esp32" in the team repo (COE/firmware),
# "coe_esp32" in the original local folders. Two shared files come from it.
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../esp32/main/feature_packet.c)
    set(COE_ESP32_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../esp32)
else()
    set(COE_ESP32_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../coe_esp32)
endif()

# Add include paths
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined include paths
    ${CMAKE_CURRENT_SOURCE_DIR}/App/communication
    # Portable ESP decoder used only by communication verification.
    ${COE_ESP32_DIR}/main
    ${COE_ESP32_DIR}/main/transport
    ${CMAKE_CURRENT_SOURCE_DIR}/App/config
    ${CMAKE_CURRENT_SOURCE_DIR}/App/pipeline
    ${CMAKE_CURRENT_SOURCE_DIR}/App/payload
    ${CMAKE_CURRENT_SOURCE_DIR}/App/dsp/bandpass
    ${CMAKE_CURRENT_SOURCE_DIR}/App/dsp/time_features
    ${CMAKE_CURRENT_SOURCE_DIR}/App/dsp/envelope
    ${CMAKE_CURRENT_SOURCE_DIR}/App/dsp/envelope_spectrum
    ${CMAKE_CURRENT_SOURCE_DIR}/App/dsp/frequency_features
    ${CMAKE_CURRENT_SOURCE_DIR}/App/platform/clock
    ${CMAKE_CURRENT_SOURCE_DIR}/App/platform/timing
    ${CMAKE_CURRENT_SOURCE_DIR}/App/acquisition
    ${CMAKE_CURRENT_SOURCE_DIR}/TestData/demos
    ${CMAKE_CURRENT_SOURCE_DIR}/TestData/fixtures
)

# Add sources to executable
target_sources(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user sources here
    App/communication/feature_uart.c
    App/communication/delivery_timing.c
    ${COE_ESP32_DIR}/main/transport/latency_packet.c
    App/dsp/time_features/time_features.c
    App/platform/timing/benchmark.c
    App/dsp/bandpass/bandpass.c
    App/dsp/envelope/envelope.c
    App/dsp/envelope_spectrum/envelope_spectrum.c
    App/dsp/frequency_features/frequency_features.c
    App/payload/feature_payload.c
    App/pipeline/pipeline.c
    App/platform/clock/processing_clock.c
    # Holds the ADC interface pins low in every mode.
    App/acquisition/ad7606_pins.c
)
if(COE_FIRMWARE_MODE STREQUAL "REPLAY")
    # Replay demos own the large AXI SRAM test buffers. The ESP32 packet
    # decoder is compiled only for the communication demo's decoder checks.
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        ${COE_ESP32_DIR}/main/feature_packet.c
        TestData/demos/communication_demo.c
        TestData/demos/time_features_demo.c
        TestData/demos/time_features_reference.c
        TestData/demos/bandpass_demo.c
        TestData/demos/envelope_demo.c
        TestData/demos/final_features_demo.c
        TestData/demos/pipeline_demo.c
        TestData/demos/pipeline_batch_demo.c
    )
else()
    # Live acquisition owns the AXI SRAM ring instead.
    target_sources(${CMAKE_PROJECT_NAME} PRIVATE
        App/acquisition/ad7606.c
        App/acquisition/acquisition_console.c
    )
    if(COE_FIRMWARE_MODE STREQUAL "LIVE_RAW")
        target_sources(${CMAKE_PROJECT_NAME} PRIVATE App/acquisition/raw_monitor.c)
    else()
        target_sources(${CMAKE_PROJECT_NAME} PRIVATE App/acquisition/live_stream.c)
    endif()
endif()

# Optimize the production DSP even while the rest of the project is Debug.
# Keep debug symbols; stepping through optimized functions can skip source lines.
# No fast-math: invalid-input checks and ordinary IEEE arithmetic are required.
option(COE_OPTIMIZE_DSP "Build production DSP with speed optimization" ON)
if(COE_OPTIMIZE_DSP)
    set_source_files_properties(
        App/dsp/time_features/time_features.c
        App/dsp/bandpass/bandpass.c
        App/dsp/envelope/envelope.c
        App/dsp/envelope_spectrum/envelope_spectrum.c
        App/dsp/frequency_features/frequency_features.c
        App/payload/feature_payload.c
        App/pipeline/pipeline.c
        # Preserve the reference function's original arithmetic options after its move.
        TestData/demos/time_features_reference.c
        PROPERTIES COMPILE_OPTIONS "-O2;-ffp-contract=off"
    )
endif()

# The AD7606 interrupts run twice per sample (60,000 times per second).
set_source_files_properties(App/acquisition/ad7606.c PROPERTIES COMPILE_OPTIONS "-O2")

# Arm CMSIS-DSP kernels used by the application.
include("cmake/cmsis_dsp.cmake")
target_link_libraries(${CMAKE_PROJECT_NAME} coe_cmsis_dsp)

# Link directories setup
target_link_directories(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined library search paths
)

# Add linked libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    # Add user defined libraries
)

# Remove wrong libob.a library dependency when using cpp files
list(REMOVE_ITEM CMAKE_C_IMPLICIT_LINK_LIBRARIES ob)

# Validate that STM32CubeMX code is compatible with C standard
if((CMAKE_C_STANDARD EQUAL 90) OR (CMAKE_C_STANDARD EQUAL 99))
    message(ERROR "Generated code requires C11 or higher")
endif()
