/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 *  plotread.h include file.
 */

/*  Include File SCCS Plotread
 *  "@(#)SCCSID: plotread.h 1.1"
 *  "@(#)SCCSID: Version Created: 11/18/92 20:38:28"
 */

#ifndef PLOTREAD
#define PLOTREAD

/* plotread.h */

typedef enum
{
    VIZ_FILE_OBJ,
    VAR_OBJ,
    TIMESLICE_OBJ,
    DATA_LIST_OBJ,
    STRUCT_BLK_OBJ,
    STRUCT_DOMAIN_OBJ
} Plotfile_Obj;
typedef enum
{
    NATIVE,
    XDR,
    BIT16
} Encoding;
typedef enum
{
    IJK,
    KJI
} Ordering;
typedef enum
{
    RECTANGULAR,
    CYLINDRICAL,
    SPHERICAL
} Geometry;
typedef enum
{
    REAL,
    DOUBLE,
    INTEGER,
    BYTE
} Variable_Type;

typedef struct _List_Node_
{
    void *data;
    struct _List_Node_ *next;
} List_Node;

typedef struct
{
    List_Node *front;
    List_Node *end;
} PlotList;

typedef struct
{
    FILE *file;
    Encoding encoding;
    char *info;
    long var_list_ptr;
    long timeslice_list_ptr;
} Plotfile;

typedef struct
{
    char *var_name;
    Variable_Type type;
    char *units;
} Variable;

typedef struct
{
    float time;
    int cycle;
    long data_obj_list_ptr;
} Timeslice;

typedef struct
{
    char *obj_name;
    char *var_name;
    long obj_ptr;
} Data_Obj;

typedef struct
{
    int ndim;
    int size[3];
    Geometry geom;
    char *units[3];
    float *coords;
} Struct_Domain;

typedef struct
{
    Struct_Domain *domain;
    void *values;
} Struct_Block;
#ifdef __cplusplus
extern "C" {
#endif
#ifdef __STDC__
Plotfile *Open_Plotfile(char *filename);
Plotfile *Set_Plotfile(FILE *fp);
void Close_Plotfile();
PlotList *Load_Variable_List(long var_list_ptr);
PlotList *Load_Timeslice_List(long timeslice_list_ptr);
PlotList *Load_Data_Obj_List(long data_obj_list_ptr);
Struct_Block *Load_Struct_Block(Variable *obj_ptr, Data_Obj *dobj);

void Free_Plotfile(Plotfile *p);
void Free_VAR_List(PlotList *v_list);
void Free_TS_List(PlotList *ts_list);
void Free_DO_List(PlotList *do_list);
void Free_SB(Struct_Block *sb);

#else /* ! __STDC__ */
Plotfile *Open_Plotfile();
void Close_Plotfile();
PlotList *Load_Variable_List();
PlotList *Load_Timeslice_List();
PlotList *Load_Data_Obj_List();
Struct_Block *Load_Struct_Block();
void Free_Plotfile();
void Free_VAR_List();
void Free_TS_List();
void Free_DO_List();
void Free_SB();
#endif /* __STDC__ */
#ifdef __cplusplus
}
#endif
#endif /* PLOTREAD_H */
