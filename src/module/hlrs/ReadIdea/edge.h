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

#define Org(e) ((e)->org)
#define Dest(e) ((e)->dest)
#define Onext(e) ((e)->onext)
#define Oprev(e) ((e)->oprev)
#define Dnext(e) ((e)->dnext)
#define Dprev(e) ((e)->dprev)

#define Other_point(e, p) ((e)->org == p ? (e)->dest : (e)->org)
#define Next(e, p) ((e)->org == p ? (e)->onext : (e)->dnext)
#define Prev(e, p) ((e)->org == p ? (e)->oprev : (e)->dprev)

#define Visited(p) ((p)->f)

#define Identical_refs(e1, e2) bool(e1 == e2)
