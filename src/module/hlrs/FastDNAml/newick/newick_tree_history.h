/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NEWICK_TREE_HISTORY_H
#define _NEWICK_TREE_HISTORY_H

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

/******************************************************************************
 *   A history entry of newick trees.                                         *
 ******************************************************************************/

struct _Newick_tree_history_entry
{
    char dummy;
};
typedef struct _Newick_tree_history_entry Newick_tree_history_entry;

EXTERN Newick_tree_history_entry *Newick_tree_history_entry_new();
EXTERN void Newick_tree_history_entry_delete(Newick_tree_history_entry **nthe);

/******************************************************************************
 *   Complete history                                                         *
 ******************************************************************************/

struct _Newick_tree_history
{
    char dummy;
};
typedef struct _Newick_tree_history Newick_tree_history;

EXTERN Newick_tree_history *Newick_tree_history_new();
EXTERN void Newick_tree_hostory_delete(Newick_tree_history **nth);

#endif
