/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*   Author: Geoff Leach, Department of Computer Science, RMIT.
 *   email: gl@cs.rmit.edu.au
 *
 *   Date: 6/10/93
 *
 *   Version 1.0
 *   
 *   Copyright (c) RMIT 1993. All rights reserved.
 *
 *   License to copy and use this software purposes is granted provided 
 *   that appropriate credit is given to both RMIT and the author.
 *
 *   License is also granted to make and use derivative works provided
 *   that appropriate credit is given to both RMIT and the author.
 *
 *   RMIT makes no representations concerning either the merchantability 
 *   of this software or the suitability of this software for any particular 
 *   purpose.  It is provided "as is" without express or implied warranty 
 *   of any kind.
 *
 *   These notices must be retained in any copies of any part of this software.
 */

/* divide_and_conquer.c */
void divide(point *p_sorted[], int l, int r, edge **l_ccw, edge **r_cw);
boolean isConnected(point *u, point *v);
bool checkAndRemove(point *origin, edge *e1, edge *e2, float max_angle);

/* edge.c */
edge *join(edge *a, point *u, edge *b, point *v, side s);
void delete_edge(edge *e);
void splice(edge *a, edge *b, point *v);
edge *make_edge(point *u, point *v);

/* error.c */
void panic(char *m);

/* i_o.c */
void read_points(int n);
void print_results(int n, char o);

/* memory.c */
void alloc_memory(int n);
void free_memory();
edge *get_edge();
void free_edge(edge *e);

/* sort.c */
void merge_sort(point *p[], point *p_temp[], int l, int r);
