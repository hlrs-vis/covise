/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
	Example 1
	Storing the header and a data element NPCO

	This program generates a databus file with a data element NPCO.
*/

#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" {
int for_cpstr(char *d, int sd, char *s, int ss);
int for_cpystr(char *d, int sd, char *s, int ss);
void *_intel_fast_memcpy(void *dest, const void *src, size_t n);
int _intel_fast_memcmp(const void *s1, const void *s2, size_t n);
void *_intel_fast_memset(void *s, int c, size_t n);
}

int for_cpstr(char *d, int sd, char *s, int ss)
{
    //fprintf(stderr,"for_cpstr %s %d %s %d\n",d, sd, s, ss);
    if (sd >= ss)
        memcpy(d, s, ss);
    else
        memcpy(d, s, sd);
    if (sd > ss)
    {
        memset(d + ss, ' ', sd - ss);
    }
    return 0;
}
int for_cpystr(char *d, int sd, char *s, int ss)
{
    //fprintf(stderr,"for_cpystr\n");
    //fprintf(stderr,"for_cpystr %d %s %d\n",sd, s, ss);
    if (s == NULL)
    {
        ss = 0;
    }
    else
    {
        if (sd >= ss)
            memcpy(d, s, ss);
        else
            memcpy(d, s, sd);
    }
    if (sd > ss)
    {
        memset(d + ss, ' ', sd - ss);
    }
    return ss;
}

void *_intel_fast_memcpy(void *dest, const void *src, size_t n)
{
    return memcpy(dest, src, n);
}

int _intel_fast_memcmp(const void *s1, const void *s2, size_t n)
{
    fprintf(stderr, "_intel_fast_memcmp\n");
    return memcmp(s1, s2, n);
}
void *_intel_fast_memset(void *s, int c, size_t n)
{
    return memset(s, c, n);
}
