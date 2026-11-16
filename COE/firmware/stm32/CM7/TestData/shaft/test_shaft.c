/* PC entry point for the shaft_demo.c known-answer checks. */
#include "shaft_demo.h"
#include <stdio.h>

int main(void)
{
    run_shaft_demo();
    printf("shaft checks: %u cases, first failure %u, %s\n", (unsigned)shaft_checked_cases,
           (unsigned)shaft_failed_check, shaft_checks_passed ? "PASS" : "FAIL");
    printf("1750 RPM window: %.4f RPM, PROX 1 1X %.3f um pp at %.3f deg (true 50 um at 70 deg)\n",
           (double)shaft_demo_rpm_1750, (double)shaft_demo_x1_um_pp, (double)shaft_demo_x1_phase_deg);
    return shaft_checks_passed ? 0 : 1;
}
