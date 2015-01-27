/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NEWICK_BUFFER_H
#define _NEWICK_BUFFER_H

#include <stdio.h>

#define NEWICK_BUFFER_UNSPECIFIED_MODE 0x0000
#define NEWICK_BUFFER_FILE_MODE 0x1000
#define NEWICK_BUFFER_STRING_MODE 0x2000

struct _Newick_buffer
{
    int max_size;
    int size;
    int pos;
    FILE *file;
    char *buffer;
    int mode;
};
typedef struct _Newick_buffer Newick_buffer;

extern Newick_buffer *Newick_buffer_new();
extern void Newick_buffer_delete(Newick_buffer **nb);

extern void Newick_buffer_init_values(Newick_buffer *nb);

extern void Newick_buffer_attach(Newick_buffer *nb, FILE *file);
extern void Newick_buffer_attach_str(Newick_buffer *nb, char *str);

extern char Newick_buffer_peek(Newick_buffer *nb, int *state);
extern int Newick_buffer_advance(Newick_buffer *nb, int *state);
extern char Newick_buffer_next_char(Newick_buffer *nb, int *state);

#endif
