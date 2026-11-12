/* Host-test button/tick inputs; never included by the firmware build. */
#include "stm32h7xx_hal.h"
#define BUTTON_USER 0
extern int fake_pressed;
static inline int BSP_PB_GetState(int button) { (void)button; return fake_pressed; }
