/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coRunRec.h"

using namespace std;
using namespace covise;

/* Inline fuer Microschrott

/// get element with certain ID 
coRunRecList::Iter coRunRecList::getID(int runID)
{
   Iter rec(*this);
   while ( (rec) && ((*rec).runID!=runID) )
      rec++;
   return rec;
}
      
/// get element with certain Contents 
coRunRecList::Iter coRunRecList::getElem(const coRunRec &searchRec)
{
   Iter rec(*this);
   while ( (rec) && !((*rec)==searchRec) )
      rec++;
   return rec;
}
   
/// add a record, replace existing equal ones
void coRunRecList::add(const coRunRec &addRec)
{
   
   Iter rec(*this);
   while ( (rec) && !((*rec)==addRec) )
   {
      // if we got it, we needn't appeand it twice
      if ( (*rec) == addRec)
         return;
      rec++;
   }
   append(addRec);

}
      
*/
