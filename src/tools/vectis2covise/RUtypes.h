/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/* 
 *  ================ Ricardo Software SQA =================
 *
 *  Filename:          RUtypes.h
 *  Version:           $Revision: /main/5 $
 *  Language:          C
 *  Subject:           RUtil public type definitions
 *  Required Software: None
 *  Documentation:     RUtil Programmer's Reference
 *  Author:            frj
 *  Creation Date:     23-Jun-1997
 *  Status:            Commercial
 *
 *  $Log$
 *  Revision 1.7  1998/03/06 22:27:57  frj
 *  (CR-7) made atexit() usage optional
 *
 *  Revision 1.6  1997/06/24  13:38:16  frj
 *  added log to SQA header
 *
 *  =======================================================
ENDSQA */

#ifndef _RUTYPES_H
#define _RUTYPES_H

typedef struct _node
{
    struct _node *next;
    struct _node *prev;
} Node;

typedef struct
{
    Node *head;
    Node *tail;
    Node *tailprev;
} List;

typedef struct
{
    List memblocklist;
    size_t memblocksize;
    size_t curmemblocksize;
    void *curmem;
} MemoryBlock;

typedef struct
{
    Node node;
    char *name;
    float add, mult;
    int mass, length, time, angle, temp;
} RU_Unit;

typedef struct
{
    Node node;
    char name;
    float order;
} RU_Prefix;

/*************/
/* rucvtunit */
/*************/
typedef struct
{
    char *str; /* Unit or prefix name - e.g. "inch", "mega" */
    char *exp; /* Name expansion - e.g. "(0.0254*metre)", "1.0e+6" */
} RU_Cpair;

/*************/
/* ruerror.c */
/*************/
typedef struct
{
    Node node; /* list node */
    char *name; /* pointer to routine name */
    char *mesg; /* pointer to error message */
    int err; /* error code */
    int sev; /* error severity */
} RU_Err;

/**************/
/* rudecode.c */
/**************/
typedef struct
{
    char *name; /* pointer to item name or null string if none */
    int type; /* item type */
    int ival; /* integer value or length of string type item */
    float rval; /* real (float) floating point value */
    double dval; /* double precision floating point value */
    char *sval; /* pointer to string value */
    char *units; /* pointer to units or null string if none */
} RU_Item;
#endif /* _RUTYPES_H */
