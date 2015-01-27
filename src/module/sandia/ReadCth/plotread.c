/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef lint
static char *sccsid[] = {
    "@(#)SCCSID: SCCS/s.plotread.c 1.1",
    "@(#)SCCSID: Version Created: 11/18/92 20:37:32"
};
#endif
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * NAME
 *      plotread -
 *
 * SECURITY CLASSIFICATION
 *      Unclassified
 *
 * DESCRIPTION
 *	Top level structure for controlling the reading of
 *	visualization files.  Uses LEX/YACC to parse the viz file.
 *
 *     RETURNS:
 *
 * DIAGNOSTICS
 *
 * LIMITATIONS
 *      None
 *
 * FILES
 *
 * NOTES
 *
 * SEE ALSO
 *
 * AUTHOR
 *     DL Campbell/CJ Pavlakos
 *     Division 1431/1425
 *     Sandia National Laboratories
 *     Albuquerque, NM  87185
 * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include "plotread.h"
#include "local_defs.h"

/* ----- Globals ------------ */

Plotfile_Obj expected_obj;
void *return_obj = NULL;
FILE *pfile = NULL;
Variable_Type current_type;
Encoding current_encoding;

int yywrap()
{
    return (1);
}

/* ----- Routines ----------- */

/*
   Open_Plotfile -- opens plotfile and loads header data 
*/
Plotfile *Open_Plotfile(filename) char *filename;
{
    Plotfile *p;

    /* open plotfile */
    if ((pfile = fopen(filename, "r")) == (FILE *)NULL)
    {
        printf("** Open_Plotfile: error opening file %s\n", filename);
        return (NULL);
    }

    /* parse header */
    expected_obj = VIZ_FILE_OBJ;
    yyparse();
    p = (Plotfile *)return_obj;
    current_encoding = p->encoding;
    return (p);
}

Plotfile *Set_Plotfile(FILE *fp)
{
    Plotfile *p;

    /* open plotfile */
    if ((pfile = fp) == (FILE *)NULL)
    {
        return (NULL);
    }

    /* parse header */
    expected_obj = VIZ_FILE_OBJ;
    yyparse();
    p = (Plotfile *)return_obj;
    if (p)
        current_encoding = p->encoding;
    return (p);
}

/*
   Close_Plotfile -- closes plotfile and cleans up as necessary
*/
void Close_Plotfile()
{
    fclose(pfile);
}

/*
   Load_Variable_List -- load variables found at file location
	var_list_ptr.  (returns list of Variable)
*/
PlotList *Load_Variable_List(var_list_ptr) long var_list_ptr;
{
    PlotList *v_list;

    /* seek file location */
    fseek(pfile, var_list_ptr, 0);

    /* parse variable list */
    expected_obj = VAR_OBJ;
    yyparse();
    v_list = (PlotList *)return_obj;
    return (v_list);
}

/* 
   Load_Timeslice_List -- loads lists of timeslices starting at file location 
      timeslice_list_ptr (returns list of Timeslice)
*/
PlotList *Load_Timeslice_List(timeslice_list_ptr) long timeslice_list_ptr;
{
    PlotList *ts_list;

    /* seek file location */
    fseek(pfile, timeslice_list_ptr, 0);

    /* parse timeslice list */
    expected_obj = TIMESLICE_OBJ;
    yyparse();
    ts_list = (PlotList *)return_obj;
    return (ts_list);
}

/*
   Load_Data_Obj_List -- load list of data objects for timeslice at file
      location data_obj_list_ptr
*/
PlotList *Load_Data_Obj_List(data_obj_list_ptr) long data_obj_list_ptr;
{
    PlotList *do_list;

    /* seek file location */
    fseek(pfile, data_obj_list_ptr, 0);

    /* parse data object */
    expected_obj = DATA_LIST_OBJ;
    yyparse();
    do_list = (PlotList *)return_obj;
    return (do_list);
}

/*
   Load_Struct_Block -- load a structured block (data and corresponding
      domain) from file location obj_ptr
*/
#ifdef __STDC__
Struct_Block *Load_Struct_Block(Variable *var, Data_Obj *dobj)

#else /* ! __STDC__ */
Struct_Block *Load_Struct_Block(var, dobj)
    Variable *var;
Data_Obj *dobj;

#endif /* __STDC__ */
{
    Struct_Block *sb;

    current_type = var->type;

    /* seek file location */
    fseek(pfile, dobj->obj_ptr, 0);

    /* parse structured block */
    expected_obj = STRUCT_BLK_OBJ;
    yyparse();
    sb = (Struct_Block *)return_obj;
    return (sb);
}

/*
   Free_Plotfile -- free memory associated with loaded Plotfile
*/
void Free_Plotfile(p)
    Plotfile *p;
{
    /* free info string */
    UFREE(p->info);

    /* free Plotfile */
    UFREE(p);
}

/*
   Free_Var_List
*/
void Free_VAR_List(v_list)
    PlotList *v_list;
{
    List_Node *ptr, *temp;
    Variable *var;

    ptr = v_list->front;
    while (ptr != NULL)
    {
        temp = ptr;
        ptr = ptr->next;
        var = (Variable *)temp->data;
        UFREE(var->var_name);
        UFREE(var->units);
        UFREE(var); /* free timeslice */
        UFREE(temp); /* free list node */
    }
    UFREE(v_list); /* free list */
}

/*
   Free_TS_List
*/
void Free_TS_List(ts_list)
    PlotList *ts_list;
{
    List_Node *ptr, *temp;
    Timeslice *ts;

    ptr = ts_list->front;
    while (ptr != NULL)
    {
        temp = ptr;
        ptr = ptr->next;
        ts = (Timeslice *)temp->data;
        UFREE(ts); /* free timeslice */
        UFREE(temp); /* free list node */
    }
    UFREE(ts_list); /* free list */
}

/*
   Free_DO_List
*/
void Free_DO_List(do_list)
    PlotList *do_list;
{
    List_Node *ptr, *temp;
    Data_Obj *dobj;

    ptr = do_list->front;
    while (ptr != NULL)
    {
        temp = ptr;
        ptr = ptr->next;
        dobj = (Data_Obj *)temp->data;
        UFREE(dobj->obj_name); /* free object name string */
        UFREE(dobj->var_name); /* free var name string */
        UFREE(dobj); /* free data object */
        UFREE(temp); /* free list node */
    }
    UFREE(do_list); /* free list */
}

/*
   Free_SB
*/
void Free_SB(sb)
    Struct_Block *sb;
{
    /* note: don't free domain */
    UFREE(sb->values); /* free values */
    UFREE(sb); /* free structured block */
}
