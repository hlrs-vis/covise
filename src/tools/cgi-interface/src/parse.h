/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* ----------------------------------------------------------------------------
0123456789012345678901234567890123456789012345678901234567890123456789012345678
---------------------------------------------------------------------------- */
/* ============================================================================
                                   includes
============================================================================ */

#ifndef _PARSE_H
#define _PARSE_H

/*=============================================================================
                                  structures
============================================================================ */

struct block
{
    char **bl_types; /* types-list of elements */
    char **bl_names; /* list of all names */
    char **bl_message; /* list of Errormessages ?! */
    char **bl_pattern; /* list of patterns */
    char **bl_maillist; /* Maillist */
    long bl_number; /* Anzahl */
};

typedef struct block BLOCK;

/*=============================================================================
                                  Prototypes
============================================================================ */

extern BLOCK *InitCgiErrorFile(char *filename);
extern char *GetElementPattern(BLOCK *block, char *name, char *type);
extern char *GetElementMaillist(BLOCK *block, char *name, char *type);
extern char *GetElementMessage(BLOCK *block, char *name, char *type);

extern char *read_file(char *filename);
/* Marc Necker April 1998 */
extern char *substitute(char *text, char *pattern, char *subst);
#endif
