/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _NEWICK_PARSER_H
#define _NEWICK_PARSER_H

#include <stdlib.h>
#include <stdio.h>

#include <network/network.h>

#ifdef __cplusplus
#ifndef EXTERN
#define EXTERN extern "C"
#endif
#else
#ifndef EXTERN
#define EXTERN extern
#endif
#endif

#define PARSE_OK 0x0000

#define PARSE_START 0x1000
#define PARSE_FINISHED 0x1fff

#define PARSE_ROOT_LABEL 0x2000

#define PARSE_LABEL 0x2010
#define PARSE_QUOTED_LABEL 0x2011
#define PARSE_UNQUOTED_LABEL 0x2012

#define PARSE_TREE 0x2500
#define PARSE_SUBTREE 0x2501
#define PARSE_DESCENDANT_LIST 0x2502
#define PARSE_FURTHER_ENTRIES 0x2503
#define PARSE_INTERNAL_NODE_LABEL 0x2504
#define PARSE_LEAF_LABEL 0x2505
#define PARSE_SUBTREE_FINISHED 0x2510

#define PARSE_BRANCH_LENGTH 0x3000
#define PARSE_LEAF_BRANCH_LENGTH 0x3001

#define PARSE_NUMBER 0x3010
#define PARSE_DECIMAL 0x3100
#define PARSE_NACHKOMMASTELLEN 0x3101
#define PARSE_EXPONENT 0x3102

#define PARSE_TREE_FINAL 0x4000

#define PARSE_ERROR 0xffff

/******************************************************************************
 *   Newick tree entry                                                        *
 ******************************************************************************/

struct _Newick_tree_entry
{
    char *label;
    double length;

    int max_num_subtrees;
    int num_subtrees;
    struct _Newick_tree_entry **next;
};
typedef struct _Newick_tree_entry Newick_tree_entry;

EXTERN Newick_tree_entry *Newick_tree_entry_new();
EXTERN void Newick_tree_entry_delete(Newick_tree_entry **nte);

EXTERN void Newick_tree_entry_delete_rec(Newick_tree_entry **nte);

EXTERN void Newick_tree_entry_add(Newick_tree_entry *nte, Newick_tree_entry *new_entry);

EXTERN void Newick_tree_entry_print_rec(Newick_tree_entry *nte, FILE *file, int off);

/******************************************************************************
 *   Newick tree                                                              *
 ******************************************************************************/

struct _Newick_tree
{

    Newick_tree_entry *start;
};
typedef struct _Newick_tree Newick_tree;

EXTERN Newick_tree *Newick_tree_new();
EXTERN void Newick_tree_delete(Newick_tree **nt);

EXTERN double Newick_tree_len(Newick_tree *nt);
EXTERN int Newick_tree_num_leafs(Newick_tree *nt);

EXTERN void Newick_tree_merge(Newick_tree_entry *nte);

EXTERN Newick_tree *Newick_tree_parse_file(FILE *file, int *status);
EXTERN int Newick_tree_parse_file_ip(Newick_tree *nt, FILE *file, int *status);

EXTERN Newick_tree *Newick_tree_parse_string(char *str, int *status);
EXTERN int Newick_tree_parse_string_ip(Newick_tree *nt, char *str, int *status);

EXTERN char *Newick_tree_create_string(Newick_tree *nt);
EXTERN int Newick_tree_print(Newick_tree *nt, FILE *file);

EXTERN int Newick_tree_write(Newick_tree *nt, sock_t sock);
EXTERN int Newick_tree_read_ip(Newick_tree *nt, sock_t);
EXTERN Newick_tree *Newick_tree_read(sock_t sock);

#endif
