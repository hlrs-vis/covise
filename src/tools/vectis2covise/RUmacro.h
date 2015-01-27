/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*RICARDO SQA =========================================================
 * Module Name   : RUmacro
 * Subject       : Public macro definitions for RUtil routines.
 * Language      : C
 * Requires      :
 * Documentation : RUtil program file.  Software.
 * Filename      : RUmacro.h
 * Author        : F R Jeske
 * Creation Date : 23-Jun-97
 * Last Modified : $Date: 1999/08/09 10:32:29 $
 * Version       : $Revision: /main/10 $
 * Status        : Current
 * Modified      : Revision, date, author, reason.
 * See ClearCase history.
 *======================================================================
 */

#ifndef _RUMACRO_H
#define _RUMACRO_H

/***** Errors *****/
#define RU_OK 0

#define RU_ERROR_MIN 1000
#define RU_ERROR_MAX 42

/* File Errors */
#define RU_CANT_OPEN_FILE 1001
#define RU_UNEXPECTED_EOF 1002
#define RU_UNKNOWN_AUXFORMAT 1003

/* Unit conversion Errors */
#define RU_UNIT_NOTFOUND 1037
#define RU_PREFIX_NOTFOUND 1038
#define RU_INVALID_UNITCONV 1039

/* Fatal runtime Errors */
#define RU_MALLOC_FAILURE 1040
#define RU_INTERNAL_ERROR 1041

/***** Severities *****/
#define RU_NOTICE 1
#define RU_MESSAGE 1
#define RU_WARNING 2
#define RU_FATAL 3
#define RU_ERROR 3
#define RU_BUG 4
#define RU_MAXSEV 4

/***** Decode types ******/
#define RU_NULL_T 0
#define RU_CHAR_T -1
#define RU_INT_T 1
#define RU_REAL_T 2
#define RU_DBLE_T 3

/***** Pipe modes ******/
#define RU_PIPE_READ 0
#define RU_PIPE_WRITE 1
#define RU_PIPE_BLOCK 0
#define RU_PIPE_NONBLOCK 2
#define RU_PIPE_WAITING -1
#define RU_PIPE_ENDED -2
#define RU_PIPE_OK 0
#define RU_PIPE_INVALID 1
#define RU_PIPE_SYSERR 2
#define RU_PIPE_NOPIPE 3

/***** Buffer Sizes ******/
#define RU_NAMESIZE 256

/***** Math Stuff */
#if !defined(M_PI)
#define M_PI ((double)3.141592653589)
#endif

#ifndef FALSE
#define FALSE 0
#define TRUE 1
#endif

#if !defined(MAX)
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif
#if !defined(MIN)
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif
#define SQR(x) ((x) * (x))
#define NINT(d) (int)(d >= 0 ? (d + 0.5) : (d - 0.5))

#define FREE(x)        \
    if (x)             \
    {                  \
        RU_freeMem(x); \
        (x) = 0;       \
    }

#define ALL(list, type, ptr) (ptr = (type *)(list)->head; ((type *)ptr)->node.next; ptr = (type *)((type *)ptr)->node.next)
#define REV_ALL(list, type, ptr) (ptr = (type *)(list)->tailprev; ((type *)ptr)->node.prev; ptr = (type *)((type *)ptr)->node.prev)
#define HEAD(list) ((void *)(((Node *)(list)->head)->next ? (list)->head : NULL))
#define TAIL(list) ((void *)(((Node *)(list)->tailprev)->prev ? (list)->tailprev : NULL))
#define NEXT(node) ((void *)(((Node *)node)->next->next ? ((Node *)node)->next : NULL))
#define PREV(node) ((void *)(((Node *)node)->prev->prev ? ((Node *)node)->prev : NULL))

#define RU_BIGENDIAN 1
#define RU_LITTLEENDIAN 2
#endif /* _RUMACRO_H */
