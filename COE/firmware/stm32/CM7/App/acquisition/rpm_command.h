/**
 * @file rpm_command.h
 * Pure text handling for the laptop's commands: no hardware, no pipeline. Case does not matter.
 *   RPM <n>                              set the shaft speed
 *   BEARING <ftf> <bsf> <bpfo> <bpfi> [half_width]   set the bearing orders (multiples of shaft speed)
 *   RPM?   CONFIG?                       ask (both give the full @CONFIG line)
 * (File name kept from the first, RPM-only version.)
 */
#ifndef COE_RPM_COMMAND_H
#define COE_RPM_COMMAND_H

#include <stdbool.h>
#include <stdint.h>

#define RPM_LINE_MAX 96U

typedef enum
{
    RPM_CMD_BAD,   /* Not a valid command. */
    RPM_CMD_SET,   /* rpm holds the requested speed (range not checked here). */
    RPM_CMD_QUERY, /* RPM? or CONFIG? */
    RPM_CMD_BEARING, /* orders[] (and half_width when has_half_width) hold the request. */
} rpm_command_kind_t;

typedef struct
{
    rpm_command_kind_t kind;
    uint32_t rpm;
    float orders[4];       /* FTF, BSF, BPFO, BPFI. */
    float half_width;
    bool has_half_width;
} rpm_command_t;

/** Collects bytes into one line; \r or \n ends it. Too long a line is dropped whole. */
typedef struct
{
    char text[RPM_LINE_MAX];
    uint32_t length;
    bool overflow;
} rpm_line_t;

/** Add one byte. True when a non-empty line is complete: read line->text, then call
 * rpm_line_clear(). Empty lines (the \n after a \r) are ignored. */
bool rpm_line_feed(rpm_line_t *line, char byte);
void rpm_line_clear(rpm_line_t *line);

/** Parse a finished line. RPM: digits only, 1..5 of them. Orders: plain decimals
 * (digits, optional '.', no sign/exponent), each > 0 and < 100; a number of its own, not "1e3". */
rpm_command_t rpm_command_parse(const char *line);

#endif
