/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _HPHMACROS_H_
#define _HPHMACROS_H_

#define MIN2(X, Y) ((X) < (Y) ? (X) : (Y))
#define MAX2(X, Y) ((X) > (Y) ? (X) : (Y))
#define MIN3(A, B, C) MIN2(A, MIN2(B, C))
#define MAX3(A, B, C) MAX2(A, MAX2(B, C))
#define MIN4(A, B, C, D) MIN2(MIN2((A), (B)), MIN2((C), (D)))
#define MAX4(A, B, C, D) MAX2(MAX2((A), (B)), MAX2((C), (D)))
#define TORANGE(X, A, B) (MIN2(MAX2(X, A), B))
#define CNT(X) (sizeof(X) / sizeof(X[0]))

#endif
