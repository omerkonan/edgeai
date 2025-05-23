/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2021 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under BSD 3-Clause license,
  * the "License"; You may not use this file except in compliance with the
  * License. You may obtain a copy of the License at:
  *                        opensource.org/licenses/BSD-3-Clause
  *
  ******************************************************************************
  */
/* USER CODE END Header */

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32l4xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */

/* USER CODE END EM */

void HAL_TIM_MspPostInit(TIM_HandleTypeDef *htim);

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);

/* USER CODE BEGIN EFP */

/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define BOOT0_PE0_Pin GPIO_PIN_0
#define BOOT0_PE0_GPIO_Port GPIOE
#define PB9_Pin GPIO_PIN_9
#define PB9_GPIO_Port GPIOB
#define BOOT0_PE0H3_Pin GPIO_PIN_3
#define BOOT0_PE0H3_GPIO_Port GPIOH
#define SPI3_MISO_Pin GPIO_PIN_4
#define SPI3_MISO_GPIO_Port GPIOB
#define SDMMC_D3_Pin GPIO_PIN_11
#define SDMMC_D3_GPIO_Port GPIOC
#define SWDCLK_Pin GPIO_PIN_14
#define SWDCLK_GPIO_Port GPIOA
#define BLE_TEST9_Pin GPIO_PIN_3
#define BLE_TEST9_GPIO_Port GPIOE
#define PB8_Pin GPIO_PIN_8
#define PB8_GPIO_Port GPIOB
#define SPI3_MOSI_Pin GPIO_PIN_5
#define SPI3_MOSI_GPIO_Port GPIOB
#define SPI3_CLK_Pin GPIO_PIN_3
#define SPI3_CLK_GPIO_Port GPIOB
#define USART2_RX_Pin GPIO_PIN_6
#define USART2_RX_GPIO_Port GPIOD
#define SPI2_CLK_Pin GPIO_PIN_1
#define SPI2_CLK_GPIO_Port GPIOD
#define EX_PWM_Pin GPIO_PIN_15
#define EX_PWM_GPIO_Port GPIOA
#define SWDIO_Pin GPIO_PIN_13
#define SWDIO_GPIO_Port GPIOA
#define OTG_FS_DP_Pin GPIO_PIN_12
#define OTG_FS_DP_GPIO_Port GPIOA
#define OTG_FS_DM_Pin GPIO_PIN_11
#define OTG_FS_DM_GPIO_Port GPIOA
#define SAI1_SCK_A_Pin GPIO_PIN_5
#define SAI1_SCK_A_GPIO_Port GPIOE
#define SAI1_MCLK_A_Pin GPIO_PIN_2
#define SAI1_MCLK_A_GPIO_Port GPIOE
#define LED1_Pin GPIO_PIN_1
#define LED1_GPIO_Port GPIOE
#define USART2_RTS_Pin GPIO_PIN_4
#define USART2_RTS_GPIO_Port GPIOD
#define LED2_Pin GPIO_PIN_0
#define LED2_GPIO_Port GPIOD
#define SDMMC_D2_Pin GPIO_PIN_10
#define SDMMC_D2_GPIO_Port GPIOC
#define PA10_Pin GPIO_PIN_10
#define PA10_GPIO_Port GPIOA
#define SDMMC_D1_Pin GPIO_PIN_9
#define SDMMC_D1_GPIO_Port GPIOC
#define OSC32_IN_Pin GPIO_PIN_14
#define OSC32_IN_GPIO_Port GPIOC
#define OSC32_OUT_Pin GPIO_PIN_15
#define OSC32_OUT_GPIO_Port GPIOC
#define SAI1_FS_A_DFSDM_D3_Pin GPIO_PIN_4
#define SAI1_FS_A_DFSDM_D3_GPIO_Port GPIOE
#define SAI1_SD_A_Pin GPIO_PIN_6
#define SAI1_SD_A_GPIO_Port GPIOE
#define DFSDM1_DATIN5_Pin GPIO_PIN_6
#define DFSDM1_DATIN5_GPIO_Port GPIOB
#define PG12_Pin GPIO_PIN_12
#define PG12_GPIO_Port GPIOG
#define USART2_TX_Pin GPIO_PIN_5
#define USART2_TX_GPIO_Port GPIOD
#define SDMMC_CMD_Pin GPIO_PIN_2
#define SDMMC_CMD_GPIO_Port GPIOD
#define SDMMC_CK_Pin GPIO_PIN_12
#define SDMMC_CK_GPIO_Port GPIOC
#define PA9_Pin GPIO_PIN_9
#define PA9_GPIO_Port GPIOA
#define BLE_RST_Pin GPIO_PIN_8
#define BLE_RST_GPIO_Port GPIOA
#define WIFI_RST_Pin GPIO_PIN_6
#define WIFI_RST_GPIO_Port GPIOC
#define I2C2_SMBA_Pin GPIO_PIN_2
#define I2C2_SMBA_GPIO_Port GPIOF
#define I2C2_SDA_Pin GPIO_PIN_1
#define I2C2_SDA_GPIO_Port GPIOF
#define I2C2_SDAF0_Pin GPIO_PIN_0
#define I2C2_SDAF0_GPIO_Port GPIOF
#define RTC_TAMP1_Pin GPIO_PIN_13
#define RTC_TAMP1_GPIO_Port GPIOC
#define CS_WIFI_Pin GPIO_PIN_3
#define CS_WIFI_GPIO_Port GPIOF
#define PG10_Pin GPIO_PIN_10
#define PG10_GPIO_Port GPIOG
#define WIFI_WAKEUP_Pin GPIO_PIN_7
#define WIFI_WAKEUP_GPIO_Port GPIOD
#define I2C3_SDA_Pin GPIO_PIN_8
#define I2C3_SDA_GPIO_Port GPIOG
#define SDMMC_D0_Pin GPIO_PIN_8
#define SDMMC_D0_GPIO_Port GPIOC
#define I2C3_SCL_Pin GPIO_PIN_7
#define I2C3_SCL_GPIO_Port GPIOG
#define C_EN_Pin GPIO_PIN_8
#define C_EN_GPIO_Port GPIOF
#define SW_SEL_Pin GPIO_PIN_6
#define SW_SEL_GPIO_Port GPIOF
#define INT2_DHC_Pin GPIO_PIN_4
#define INT2_DHC_GPIO_Port GPIOF
#define CS_ADWB_Pin GPIO_PIN_5
#define CS_ADWB_GPIO_Port GPIOF
#define PGOOD_Pin GPIO_PIN_7
#define PGOOD_GPIO_Port GPIOF
#define PG9_Pin GPIO_PIN_9
#define PG9_GPIO_Port GPIOG
#define SPI1_MISO_Pin GPIO_PIN_3
#define SPI1_MISO_GPIO_Port GPIOG
#define BLE_SPI_CS_Pin GPIO_PIN_5
#define BLE_SPI_CS_GPIO_Port GPIOG
#define INT_HTS_Pin GPIO_PIN_6
#define INT_HTS_GPIO_Port GPIOG
#define SPI1_MOSI_Pin GPIO_PIN_4
#define SPI1_MOSI_GPIO_Port GPIOG
#define SPI1_CLK_Pin GPIO_PIN_2
#define SPI1_CLK_GPIO_Port GPIOG
#define STSAFE_RESET_Pin GPIO_PIN_10
#define STSAFE_RESET_GPIO_Port GPIOF
#define INT_M_Pin GPIO_PIN_9
#define INT_M_GPIO_Port GPIOF
#define WIFI_BOOT0_Pin GPIO_PIN_12
#define WIFI_BOOT0_GPIO_Port GPIOF
#define SAI1_SD_B_Pin GPIO_PIN_7
#define SAI1_SD_B_GPIO_Port GPIOE
#define CS_DH_Pin GPIO_PIN_15
#define CS_DH_GPIO_Port GPIOD
#define I2C4_SCL_Pin GPIO_PIN_12
#define I2C4_SCL_GPIO_Port GPIOD
#define I2C4_SDA_Pin GPIO_PIN_13
#define I2C4_SDA_GPIO_Port GPIOD
#define EX_RESET_Pin GPIO_PIN_11
#define EX_RESET_GPIO_Port GPIOD
#define OSC_IN_Pin GPIO_PIN_0
#define OSC_IN_GPIO_Port GPIOH
#define OSC_OUT_Pin GPIO_PIN_1
#define OSC_OUT_GPIO_Port GPIOH
#define ADC1_IN1_Pin GPIO_PIN_0
#define ADC1_IN1_GPIO_Port GPIOC
#define SPI2_MISO_Pin GPIO_PIN_2
#define SPI2_MISO_GPIO_Port GPIOC
#define INT2_ADWB_Pin GPIO_PIN_2
#define INT2_ADWB_GPIO_Port GPIOB
#define CHRG_Pin GPIO_PIN_15
#define CHRG_GPIO_Port GPIOF
#define WIFI_DRDY_Pin GPIO_PIN_11
#define WIFI_DRDY_GPIO_Port GPIOE
#define BUTTON_PWR_Pin GPIO_PIN_10
#define BUTTON_PWR_GPIO_Port GPIOD
#define USART3_RX_Pin GPIO_PIN_9
#define USART3_RX_GPIO_Port GPIOD
#define USART3_TX_Pin GPIO_PIN_8
#define USART3_TX_GPIO_Port GPIOD
#define ADC1_IN2_Pin GPIO_PIN_1
#define ADC1_IN2_GPIO_Port GPIOC
#define SPI2_MOSI_Pin GPIO_PIN_3
#define SPI2_MOSI_GPIO_Port GPIOC
#define USART3_RTS_Pin GPIO_PIN_1
#define USART3_RTS_GPIO_Port GPIOB
#define CS_DHC_Pin GPIO_PIN_13
#define CS_DHC_GPIO_Port GPIOF
#define DFSDM1_CKOUT_Pin GPIO_PIN_9
#define DFSDM1_CKOUT_GPIO_Port GPIOE
#define DCDC_2_EN_Pin GPIO_PIN_13
#define DCDC_2_EN_GPIO_Port GPIOE
#define SPI2_MOSI_p2_Pin GPIO_PIN_15
#define SPI2_MOSI_p2_GPIO_Port GPIOB
#define PA0_Pin GPIO_PIN_0
#define PA0_GPIO_Port GPIOA
#define DAC1_OUT1_Pin GPIO_PIN_4
#define DAC1_OUT1_GPIO_Port GPIOA
#define PC5_WKUP5_Pin GPIO_PIN_5
#define PC5_WKUP5_GPIO_Port GPIOC
#define INT1_DHC_Pin GPIO_PIN_8
#define INT1_DHC_GPIO_Port GPIOE
#define INT_STT_Pin GPIO_PIN_15
#define INT_STT_GPIO_Port GPIOE
#define PB11_Pin GPIO_PIN_11
#define PB11_GPIO_Port GPIOB
#define PB14_Pin GPIO_PIN_14
#define PB14_GPIO_Port GPIOB
#define PA1_Pin GPIO_PIN_1
#define PA1_GPIO_Port GPIOA
#define INT2_DH_Pin GPIO_PIN_2
#define INT2_DH_GPIO_Port GPIOA
#define EX_ADC_Pin GPIO_PIN_5
#define EX_ADC_GPIO_Port GPIOA
#define uC_ADC_BATT_Pin GPIO_PIN_4
#define uC_ADC_BATT_GPIO_Port GPIOC
#define SEL1_2_Pin GPIO_PIN_0
#define SEL1_2_GPIO_Port GPIOG
#define BLE_TEST8_Pin GPIO_PIN_10
#define BLE_TEST8_GPIO_Port GPIOE
#define DFSDM1_D7_Pin GPIO_PIN_10
#define DFSDM1_D7_GPIO_Port GPIOB
#define SD_DETECT_Pin GPIO_PIN_12
#define SD_DETECT_GPIO_Port GPIOB
#define CHRGB0_Pin GPIO_PIN_0
#define CHRGB0_GPIO_Port GPIOB
#define SEL3_4_Pin GPIO_PIN_14
#define SEL3_4_GPIO_Port GPIOF
#define BLE_INT_Pin GPIO_PIN_1
#define BLE_INT_GPIO_Port GPIOG
#define PE12_Pin GPIO_PIN_12
#define PE12_GPIO_Port GPIOE
#define INT1_ADWB_Pin GPIO_PIN_14
#define INT1_ADWB_GPIO_Port GPIOE
#define USART3_CTS_Pin GPIO_PIN_13
#define USART3_CTS_GPIO_Port GPIOB
/* USER CODE BEGIN Private defines */

/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
