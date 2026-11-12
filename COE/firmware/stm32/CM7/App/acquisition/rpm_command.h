/**
 * @file rpm_command.h
 * Pure text handling for the laptop's speed commands: no hardware, no pipeline.
 * "RPM <n>" sets the speed, "RPM?" asks for it. Case does not matter.
 */
#ifndef COE_RPM_COMMAND_H
#define COE_RPM_COMMAND_H

#include <stdbool.h>
#include <stdint.h>

#define RPM_LINE_MAX 24U

typedef enum
{
    RPM_CMD_BAD,   /* Not a valid command. */
    RPM_CMD_SET,   /* rpm holds the requested speed (range not checked here). */
    RPM_CMD_QUERY,
} rpm_command_kind_t;

typedef struct
{
    rpm_command_kind_t kind;
    uint32_t rpm;
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

/** Parse a finished line. Digits only, 1..5 of them, so "RPM 3600x" and "RPM -5" are BAD. */
rpm_command_t rpm_command_parse(const char *line);

#endif
