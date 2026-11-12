/**
 * @file rpm_command.c
 * Line assembly and parsing of the "RPM <n>" / "RPM?" commands.
 */
#include "rpm_command.h"

static char upper(char c)
{
    return (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
}

void rpm_line_clear(rpm_line_t *line)
{
    line->length = 0U;
    line->overflow = false;
    line->text[0] = '\0';
}

bool rpm_line_feed(rpm_line_t *line, char byte)
{
    if (byte == '\r' || byte == '\n')
    {
        const bool complete = line->length != 0U && !line->overflow;
        if (complete)
            line->text[line->length] = '\0';
        else
            rpm_line_clear(line);
        return complete;
    }
    if (line->length + 1U < RPM_LINE_MAX)
        line->text[line->length++] = byte;
    else
        line->overflow = true;
    return false;
}

rpm_command_t rpm_command_parse(const char *line)
{
    rpm_command_t command = {RPM_CMD_BAD, 0U};
    while (*line == ' ' || *line == '\t')
        ++line;
    if (upper(line[0]) != 'R' || upper(line[1]) != 'P' || upper(line[2]) != 'M')
        return command;
    line += 3;
    if (*line == '?')
    {
        ++line;
        while (*line == ' ' || *line == '\t')
            ++line;
        if (*line == '\0')
            command.kind = RPM_CMD_QUERY;
        return command;
    }
    if (*line != ' ' && *line != '\t')
        return command;
    while (*line == ' ' || *line == '\t')
        ++line;
    uint32_t value = 0U, digits = 0U;
    while (*line >= '0' && *line <= '9' && digits < 6U)
    {
        value = value * 10U + (uint32_t)(*line - '0');
        ++digits;
        ++line;
    }
    while (*line == ' ' || *line == '\t')
        ++line;
    if (digits == 0U || digits > 5U || *line != '\0')
        return command;
    command.kind = RPM_CMD_SET;
    command.rpm = value;
    return command;
}
