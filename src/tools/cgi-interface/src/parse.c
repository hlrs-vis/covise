/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ----------------------------------------------------------------------------
0123456789012345678901234567890123456789012345678901234567890123456789012345678
---------------------------------------------------------------------------- */
/* ============================================================================
                                   Includes 
============================================================================ */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parse.h"

/* ============================================================================
                              Globale Variablen
============================================================================ */

/* ============================================================================
                                 Prototypes
============================================================================ */

char *read_file(char *name);
static long count_blocks(char *pos);

/* ============================================================================
                                  Funktions
============================================================================ */

/* ------------------------------------------------------------------------- */

char *read_file(char *name)
{
    FILE *handle;
    long len;
    char *buffer;

    if (!(handle = fopen(name, "r")))
        return 0;
    if ((fseek(handle, 0, SEEK_END)))
    {
        fclose(handle);
        return 0;
    }
    if (0 == (len = ftell(handle)))
    {
        fclose(handle);
        return 0;
    }
    if ((fseek(handle, 0, SEEK_SET)))
    {
        fclose(handle);
        return 0;
    }
    if (NULL == (buffer = malloc((size_t)len + 1)))
    {
        fclose(handle);
        return 0;
    }
    if (0 == (fread(buffer, len, 1, handle)))
    {
        free(buffer);
        fclose(handle);
        return 0;
    }
    fclose(handle);
    buffer[len] = 0;
    return buffer;
}

char *substitute(char *text, char *pattern, char *subst) /* Marc Necker April 1998 */
{
    char *pat;
    char *newtext;
    char *textptr;
    char *tmpptr;
    char *patstart;

    newtext = text;
    textptr = text;
    while ((*textptr) != '\0')
    {
        if ((*textptr) == (*pattern))
        {
            pat = pattern;
            patstart = textptr;
            while (((*pat) == (*textptr)) && ((*pat) != '\0'))
            {
                pat++;
                textptr++;
            }
            if ((*pat) == '\0')
            {
                newtext = malloc((sizeof(char) * (strlen(text) + strlen(subst) - strlen(pattern) + 1)));
                (*patstart) = '\0';
                strcpy(newtext, text);
                tmpptr = newtext + strlen(text);
                strcpy(tmpptr, subst);
                tmpptr += strlen(subst);
                strcpy(tmpptr, textptr);
                textptr = newtext + strlen(text);
                free(text);
                text = newtext;
            }
        }
        textptr++;
    }
    return newtext;
}

/* ------------------------------------------------------------------------- */

static long count_blocks(char *pos)
{
    long count = 0;

    do
    {
        pos++;
        count++;
        pos = strstr(pos, "\n# BLOCK");
    } while (pos);
    return count - 1;
}

/* ------------------------------------------------------------------------- */

BLOCK *InitCgiErrorFile(char *filename)
{
    char *file; /* Das ErrorFile im Speicher        */
    char *pos; /* Ein Temp-Pointer in den Speicher */

    char **names = NULL; /* argv aller Namen                 */
    char **types = NULL; /* argv aller Typen                 */
    char **message = NULL; /* argv fuer Werte ...              */
    char **pattern = NULL; /* argv fuer Werte ...              */
    char **maillist = NULL; /* argv fuer Werte ...              */

    long count = 0; /* ein armer kleiner counter        */
    BLOCK *block; /* Die grosse Datenstruktur         */

    /* das ERRORFILE laden */
    if (NULL == (pos = file = read_file(filename)))
    {
        exit(0);
    }

    /* die Zeilenanzahl bestimmen */
    if (0 == (count = count_blocks(pos)))
    {
        free(file);
        exit(0);
    }

    /* Seicher fuer die Blockliste holen */
    types = (char **)malloc(sizeof(void **) * count);
    names = (char **)malloc(sizeof(void **) * count);
    message = (char **)malloc(sizeof(void **) * count);
    pattern = (char **)malloc(sizeof(void **) * count);
    maillist = (char **)malloc(sizeof(void **) * count);
    block = (BLOCK *)malloc(sizeof(BLOCK));

    /* kam genug Speicher */
    if (0 == (names && types && pattern && maillist && message && block))
    {
        if (types)
            free(types);
        if (names)
            free(names);
        if (message)
            free(message);
        if (pattern)
            free(pattern);
        if (maillist)
            free(maillist);
        if (block)
            free(block);
    }

    /* Alles schon einmal eintragen */
    block->bl_types = types;
    block->bl_names = names;
    block->bl_message = message;
    block->bl_pattern = pattern;
    block->bl_maillist = maillist;
    block->bl_number = count;

    /* Alle Elemente uebernehmen */
    count = 0;
    while (count < block->bl_number)
    {
        pos = strstr(pos, "\n# BLOCK") + 8; /* die naechste Eintragung stellen */

        pos = strstr(pos, "# TYPE") + 6; /* auf den Datentyp    */
        while (*pos <= 32)
            pos++; /* no whitespaces      */
        block->bl_types[count] = pos; /* die Stelle merken   */
        while (*pos > 32)
            pos++; /* Das Wortende suchen */
        *pos++ = 0;

        pos = strstr(pos, "# NAME") + 6; /* auf den Datentyp    */
        while (*pos <= 32)
            pos++; /* no whitespaces      */
        block->bl_names[count] = pos; /* die Stelle merken   */
        while (*pos > 32)
            pos++; /* Das Wortende suchen */
        *pos++ = 0;

        /* Ist es ein Block vom Typ VAR ? */
        if (!strncmp(block->bl_types[count], "VAR", 3))
        {
            /* den Pruef-Pattern uebernehmen */
            pos = strstr(pos, "# PATTERN") + 9;
            while (*pos != '"')
                pos++;
            pos++;
            block->bl_pattern[count] = pos;
            while (*pos != '"')
                pos++;
            *pos++ = 0;

            /* den das ausdruckformat feststellen */
            pos = strstr(pos, "# FORMAT") + 8;
            while (*pos != '"')
                pos++;
            pos++;
            block->bl_maillist[count] = pos;
            while (*pos != '"')
                pos++;
            *pos++ = 0;
        }

        /* dieser Teil ist in jedem Zweig, darum als "Invariante" */
        pos = strstr(pos, "# PRINT") + 7;
        while (*pos <= 32)
            pos++;
        block->bl_message[count] = pos;
        pos = strstr(pos, "# END");
        *pos++ = 0;
        count++;
    }
    return block;
}

/* ------------------------------------------------------------------------- */

void FreeCgiErrorFile(BLOCK *block)
{
    free(block->bl_types);
    free(block->bl_names);
    free(block->bl_message);
    free(block->bl_pattern);
    free(block->bl_maillist);
    free(block);
}

/* ------------------------------------------------------------------------- */

char *GetElementPattern(BLOCK *block, char *name, char *type)
{
    long index = 0;

    while (index < block->bl_number)
    {
        if (!strcmp(name, block->bl_names[index]))
        {
            if (!strcmp(type, block->bl_types[index]))
            {
                return block->bl_pattern[index];
            }
        }
        index++;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */

char *GetElementMaillist(BLOCK *block, char *name, char *type)
{
    long index = 0;

    while (index < block->bl_number)
    {
        if (!strcmp(name, block->bl_names[index]))
        {
            if (!strcmp(type, block->bl_types[index]))
            {
                return block->bl_maillist[index];
            }
        }
        index++;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */

char *GetElementMessage(BLOCK *block, char *name, char *type)
{
    long index = 0;

    while (index < block->bl_number)
    {
        if (0 == strcmp(name, block->bl_names[index]))
        {
            if (0 == strcmp(type, block->bl_types[index]))
            {
                return block->bl_message[index];
            }
        }
        index++;
    }
    return NULL;
}

/* ------------------------------------------------------------------------- */
