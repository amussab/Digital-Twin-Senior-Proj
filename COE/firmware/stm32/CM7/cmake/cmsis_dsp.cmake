# Selected, unmodified Arm kernels from STM32CubeH7 V1.13.0's DSP package.
# List the kernels we use explicitly so firmware does not build the entire DSP.
set(COE_CMSIS_DSP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../Drivers/CMSIS/DSP")
add_library(coe_cmsis_dsp STATIC
    "${COE_CMSIS_DSP_ROOT}/Source/BasicMathFunctions/arm_abs_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/FilteringFunctions/arm_biquad_cascade_df2T_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/FilteringFunctions/arm_biquad_cascade_df2T_init_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/TransformFunctions/arm_rfft_fast_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/TransformFunctions/arm_rfft_fast_init_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/TransformFunctions/arm_cfft_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/TransformFunctions/arm_cfft_init_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/TransformFunctions/arm_cfft_radix8_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/TransformFunctions/arm_bitreversal2.c"
    "${COE_CMSIS_DSP_ROOT}/Source/CommonTables/arm_common_tables.c"
    "${COE_CMSIS_DSP_ROOT}/Source/CommonTables/arm_const_structs.c"
    "${COE_CMSIS_DSP_ROOT}/Source/StatisticsFunctions/arm_mean_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/StatisticsFunctions/arm_rms_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/BasicMathFunctions/arm_offset_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/BasicMathFunctions/arm_mult_f32.c"
    "${COE_CMSIS_DSP_ROOT}/Source/ComplexMathFunctions/arm_cmplx_mag_squared_f32.c"
)
target_include_directories(coe_cmsis_dsp
    PUBLIC "${COE_CMSIS_DSP_ROOT}/Include"
    PRIVATE "${CMAKE_CURRENT_LIST_DIR}/../../Drivers/CMSIS/Include"
)
# Only the tables needed by the fixed 4096-point real FFT are compiled.
target_compile_definitions(coe_cmsis_dsp PRIVATE
    ARM_MATH_LOOPUNROLL ARM_DSP_CONFIG_TABLES ARM_FFT_ALLOW_TABLES
    ARM_TABLE_TWIDDLECOEF_F32_2048 ARM_TABLE_BITREVIDX_FLT_2048
    ARM_TABLE_TWIDDLECOEF_RFFT_F32_4096
)
# Arm recommends Ofast for DSP kernels. Scope it to the library: application
# validation (isfinite) retains ordinary compiler semantics. Allow hardware FMA.
# Cortex-M7 hard-float target flags come from the CM7 project's toolchain.
target_compile_options(coe_cmsis_dsp PRIVATE -Ofast -ffp-contract=fast)
