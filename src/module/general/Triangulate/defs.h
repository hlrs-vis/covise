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

#define SYSV
#define OUTPUT
#define TIME

#ifndef NULL
#define NULL 0
#endif
#define TRUE 1
#define FALSE 0

/* Edge sides. */
typedef enum
{
    right,
    left
} side;

/* Geometric and topological entities. */
//typedef  float  real;
//typedef  float  ordinate;
typedef unsigned char boolean;
typedef struct point point;
typedef struct edge edge;

struct point
{
    float x;
    float y;
    edge *entry_pt;
};

struct edge
{
    point *org;
    point *dest;
    edge *onext;
    edge *oprev;
    edge *dnext;
    edge *dprev;
};
