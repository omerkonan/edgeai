################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/startup_stm32l4r9zijx.s 

S_DEPS += \
./Core/Startup/startup_stm32l4r9zijx.d 

OBJS += \
./Core/Startup/startup_stm32l4r9zijx.o 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/startup_stm32l4r9zijx.o: ../Core/Startup/startup_stm32l4r9zijx.s Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m4 -g3 -c -I"../tensorflow_lite_lib/tensorflow" -I"../tensorflow_lite_lib/third_party/flatbuffers/include" -I"../tensorflow_lite_lib/third_party/gemmlowp" -I"../tensorflow_lite_lib/third_party/ruy" -I"../tensorflow_lite_lib" -x assembler-with-cpp -MMD -MP -MF"Core/Startup/startup_stm32l4r9zijx.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

