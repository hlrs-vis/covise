/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _LIBAPPL_APPL_DATATYPE_H
#define _LIBAPPL_APPL_DATATYPE_H

/* Datatype identifiers follow the following principle:
   Scalar types (excluding pointers) have simple number up to 0x0ff
   (That should be enough for most languages today...)

   Pointer to the corresponding types get an additional 0x100

   Arrary of datatypes have set the flag corresponding to 0x1000

   A group of data is indicated by 0x2000
   
   MPI Datatypes start from 0x3000

   User datatypes start from 0xe000                                  */

#define APPL_DATATYPE_BOOL 0x0001
#define APPL_DATATYPE_CHAR 0x0002
#define APPL_DATATYPE_STRING 0x0003
#define APPL_DATATYPE_INT 0x0004
#define APPL_DATATYPE_REAL 0x0005

#define APPL_DATATYPE_PTR 0x0100

#define APPL_DATATYPE_BOOL_PTR 0x0101
#define APPL_DATATYPE_CHAR_PTR 0x0102
#define APPL_DATATYPE_INT_PTR 0x0104
#define APPL_DATATYPE_REAL_PTR 0x0105

#define APPL_DATATYPE_ARRAY 0x1000

#define APPL_DATATYPE_BOOL_ARRAY 0x1001
#define APPL_DATATYPE_CHAR_ARRAY 0x1002
#define APPL_DATATYPE_INT_ARRAY 0x1004
#define APPL_DATATYPE_REAL_ARRAY 0x1005

#define APPL_DATATYPE_GROUP 0x2000
#define APPL_DATATYPE_DISTRIBUTED 0x3000

/* Derived datatypes: This must be changed to a more
   generic form!                                     */

#define APPL_DATATYPE_SCALARFIELD2D 0xe000
#define APPL_DATATYPE_SCALARFIELD2D_PTR 0xe001

#define APPL_DATATYPE_TREE 0xe100
#define APPL_DATATYPE_TREE_PTR 0xe101

#define APPL_DATATYPE_UNKNOWN 0xffff

typedef int appl_datatype_t;

#endif
