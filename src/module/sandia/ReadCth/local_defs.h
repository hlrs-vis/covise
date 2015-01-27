/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 *  localdefs.h include file.
 */

/*  Include File SCCS header
 *  "@(#)SCCSID: local_defs.h 1.1"
 *  "@(#)SCCSID: Version Created: 11/18/92 20:43:03"
 *
 *   Commonly used definitions.
 */

#ifndef LOCAL_DEFS_H
#define LOCAL_DEFS_H

#ifdef Boolean
#undef Boolean
#endif
typedef int Boolean;

#ifndef NULL
#define NULL 0
#endif

#ifndef FALSE
#define FALSE 0
#define TRUE 1
#endif

#ifndef False
#define False 0
#define True 1
#endif

/*
 * The following make the use of malloc much easier.  You generally want
 * to use UALLOC() to dynamically allocate space and UFREE() to free
 * the space up.
 *
 * UALLOC       - Allocate permanent memory.
 * UCALLOC      - Allocate permanent memory and initializes it to zeroes.
 * UALLOCA      - Allocate temporary memory that is automatically free'd
 *                when calling procedure returns.
 * UREALLOC     - Change the size of allocated memory from a previous
 *                UALLOC() call (only on permanent memory).
 * STRALLOC     - Returns a pointer to the string copied into permanent
 *                memory.
 * STRALLOCA    - Same as STRALLOC except uses temporary memory.
 * UFREE        - Free permanent memory and set pointer to NULL.
 */

#define UALLOC(type, count) (type *) malloc((unsigned)(count) * (sizeof(type)))
#define UCALLOC(type, count) (type *) calloc((unsigned)(count), (sizeof(type)))
#define UALLOCA(type, count) (type *) alloca(count * sizeof(type))
#define UREALLOC(ptr, type, count) (type *) REALLOC((char *)ptr, (unsigned)sizeof(type) * count)
#define STRALLOC(string) strcpy(UALLOC(char, strlen(string) + 1), string)
#define STRALLOCA(string) strcpy(UALLOCA(char, strlen(string) + 1), string)
#define UFREE(ptr)                  \
    if (ptr)                        \
    {                               \
        (void) free((char *)(ptr)); \
        (ptr) = NULL;               \
    }

/* 
 *  Return the number of elements in an array.
 */

#define DIM(array) (sizeof(array) / sizeof(*(array)))

/* 
 *  Are two strings equal?
 */

#define STREQ(a, b) (strcmp((a), (b)) == 0)
#endif /* LOCAL_DEFS_H */
