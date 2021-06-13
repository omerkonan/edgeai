################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (9-2020-q2-update)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CC_SRCS += \
../tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc \
../tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc 

CC_DEPS += \
./tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/greedy_memory_planner.d \
./tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/linear_memory_planner.d 

OBJS += \
./tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/greedy_memory_planner.o \
./tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/linear_memory_planner.o 


# Each subdirectory must supply rules for building sources it contributes
tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/greedy_memory_planner.o: ../tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L4R9xx -c -I../Core/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I../USB_DEVICE/App -I../USB_DEVICE/Target -I../Middlewares/ST/STM32_USB_Device_Library/Core/Inc -I../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -I"../tensorflow_lite_lib/tensorflow" -I"../tensorflow_lite_lib/third_party/flatbuffers/include" -I"../tensorflow_lite_lib/third_party/gemmlowp" -I"../tensorflow_lite_lib/third_party/ruy" -I"../tensorflow_lite_lib" -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -MMD -MP -MF"tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/greedy_memory_planner.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"
tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/linear_memory_planner.o: ../tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/subdir.mk
	arm-none-eabi-g++ "$<" -mcpu=cortex-m4 -std=gnu++14 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32L4R9xx -c -I../Core/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc -I../Drivers/STM32L4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32L4xx/Include -I../Drivers/CMSIS/Include -I../USB_DEVICE/App -I../USB_DEVICE/Target -I../Middlewares/ST/STM32_USB_Device_Library/Core/Inc -I../Middlewares/ST/STM32_USB_Device_Library/Class/CDC/Inc -I"../tensorflow_lite_lib/tensorflow" -I"../tensorflow_lite_lib/third_party/flatbuffers/include" -I"../tensorflow_lite_lib/third_party/gemmlowp" -I"../tensorflow_lite_lib/third_party/ruy" -I"../tensorflow_lite_lib" -O0 -ffunction-sections -fdata-sections -fno-exceptions -fno-rtti -fno-use-cxa-atexit -Wall -fstack-usage -MMD -MP -MF"tensorflow_lite_lib/tensorflow/lite/micro/memory_planner/linear_memory_planner.d" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

