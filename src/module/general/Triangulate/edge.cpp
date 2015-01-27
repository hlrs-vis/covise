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

#include "defs.h"
#include "decl.h"
#include "edge.h"

extern point *p_array;

/* 
 *  Creates a new edge and adds it to two rings of edges.
 */
edge *join(edge *a, point *u, edge *b, point *v, side s)
{
    edge *e;

    /* u and v are the two vertices which are being joined.
     a and b are the two edges associated with u and v res.  */

    e = make_edge(u, v);

    if (s == left)
    {
        if (Org(a) == u)
            splice(Oprev(a), e, u);
        else
            splice(Dprev(a), e, u);
        splice(b, e, v);
    }
    else
    {
        splice(a, e, u);
        if (Org(b) == v)
            splice(Oprev(b), e, v);
        else
            splice(Dprev(b), e, v);
    }

    return e;
}

/* 
 *  Remove an edge.
 */
void delete_edge(edge *e)
{
    point *u, *v;

    /* Cache origin and destination. */
    u = Org(e);
    v = Dest(e);

    /* Adjust entry points. */
    if (u->entry_pt == e)
        u->entry_pt = e->onext;
    if (v->entry_pt == e)
        v->entry_pt = e->dnext;

    /* Four edge links to change */
    if (Org(e->onext) == u)
        e->onext->oprev = e->oprev;
    else
        e->onext->dprev = e->oprev;

    if (Org(e->oprev) == u)
        e->oprev->onext = e->onext;
    else
        e->oprev->dnext = e->onext;

    if (Org(e->dnext) == v)
        e->dnext->oprev = e->dprev;
    else
        e->dnext->dprev = e->dprev;

    if (Org(e->dprev) == v)
        e->dprev->onext = e->dnext;
    else
        e->dprev->dnext = e->dnext;

    free_edge(e);
}

/* 
 *  Add an edge to a ring of edges. 
 */
void splice(edge *a, edge *b, point *v)
{
    edge *next;

    /* b must be the unnattached edge and a must be the previous 
     ccw edge to b. */

    if (Org(a) == v)
    {
        next = Onext(a);
        Onext(a) = b;
    }
    else
    {
        next = Dnext(a);
        Dnext(a) = b;
    }

    if (Org(next) == v)
        Oprev(next) = b;
    else
        Dprev(next) = b;

    if (Org(b) == v)
    {
        Onext(b) = next;
        Oprev(b) = a;
    }
    else
    {
        Dnext(b) = next;
        Dprev(b) = a;
    }
}

/*
 *  Initialise a new edge.
 */
edge *make_edge(point *u, point *v)
{
    edge *e;

    e = get_edge();

    e->onext = e->oprev = e->dnext = e->dprev = e;
    e->org = u;
    e->dest = v;
    if (u->entry_pt == NULL)
        u->entry_pt = e;
    if (v->entry_pt == NULL)
        v->entry_pt = e;

    return e;
}
