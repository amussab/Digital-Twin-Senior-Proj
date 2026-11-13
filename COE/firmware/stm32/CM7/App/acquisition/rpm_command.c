/**
 * @file rpm_command.c
 * Line assembly and parsing of the "RPM <n>", "BEARING ...", "RPM?" and "CONFIG?" commands.
 */
#include "rpm_command.h"

static char upper(char c)
{
    return (c >= 'a' && c <= 'z') ? (char)(c - 'a' + 'A') : c;
}

static bool blank(char c)
{
    return c == ' ' || c == '\t';
}

static const char *skip_blanks(const char *s)
{
    while (blank(*s))
        ++s;
    return s;
}

/** True when s starts with the (upper-case) word, ignoring case. */
static bool starts_with(const char *s, const char *word, const char **after)
{
    while (*word != '\0')
    {
        if (upper(*s) != *word)
            return false;
        ++s;
        ++word;
    }
    *after = s;
    return true;
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

/** Plain decimal: up to 9 digits, optional '.', at least one digit. Advances *s past it. */
static bool parse_decimal(const char **s, float *out)
{
    const char *p = *s;
    double value = 0.0, scale = 1.0;
    uint32_t digits = 0U;
    bool fraction = false;
    while ((*p >= '0' && *p <= '9') || (*p == '.' && !fraction))
    {
        if (*p == '.')
            fraction = true;
        else
        {
            if (++digits > 9U)
                return false;
            if (fraction)
            {
                scale *= 0.1;
                value += (double)(*p - '0') * scale;
            }
            else
                value = value * 10.0 + (double)(*p - '0');
        }
        ++p;
    }
    if (digits == 0U || (*p != '\0' && !blank(*p)))
        return false;
    *out = (float)value;
    *s = p;
    return true;
}

static rpm_command_t parse_bearing(const char *line)
{
    rpm_command_t command = {RPM_CMD_BAD, 0U, {0.0f, 0.0f, 0.0f, 0.0f}, 0.0f, false};
    float values[5];
    uint32_t count = 0U;
    line = skip_blanks(line);
    while (*line != '\0')
    {
        if (count == 5U || !parse_decimal(&line, &values[count]) || !(values[count] > 0.0f)
            || values[count] >= 100.0f)
            return command;
        ++count;
        line = skip_blanks(line);
    }
    if (count != 4U && count != 5U)
        return command;
    for (uint32_t i = 0U; i < 4U; ++i)
        command.orders[i] = values[i];
    command.has_half_width = count == 5U;
    command.half_width = count == 5U ? values[4] : 0.0f;
    command.kind = RPM_CMD_BEARING;
    return command;
}

rpm_command_t rpm_command_parse(const char *line)
{
    rpm_command_t command = {RPM_CMD_BAD, 0U, {0.0f, 0.0f, 0.0f, 0.0f}, 0.0f, false};
    const char *p;
    line = skip_blanks(line);
    if (starts_with(line, "CONFIG?", &p) || starts_with(line, "RPM?", &p))
    {
        if (*skip_blanks(p) == '\0')
            command.kind = RPM_CMD_QUERY;
        return command;
    }
    if (starts_with(line, "BEARING", &p))
    {
        if (!blank(*p))
            return command;
        return parse_bearing(p);
    }
    if (!starts_with(line, "RPM", &p) || !blank(*p))
        return command;
    p = skip_blanks(p);
    uint32_t value = 0U, digits = 0U;
    while (*p >= '0' && *p <= '9' && digits < 6U)
    {
        value = value * 10U + (uint32_t)(*p - '0');
        ++digits;
        ++p;
    }
    p = skip_blanks(p);
    if (digits == 0U || digits > 5U || *p != '\0')
        return command;
    command.kind = RPM_CMD_SET;
    command.rpm = value;
    return command;
}
