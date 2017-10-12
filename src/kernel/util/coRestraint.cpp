/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\
 **                                                                        **
 **                                                                        **
 ** Description: Interface classes for application modules to the COVISE   **
 **              software environment                                      **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                                                                        **
 **                             (C)1997 RUS                                **
 **                Computer Center University of Stuttgart                 **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author:                                                                **
 ** Date:                                                                  **
\**************************************************************************/

#include "coRestraint.h"

#include <sstream>
#include <cstdio>
#include <cctype>

using namespace covise;

//==========================================================================
//
//==========================================================================
coRestraint::coRestraint()
: all(false)
, changed(true)
, stringCurrent(false)
{
}

//==========================================================================
//
//==========================================================================
coRestraint::~coRestraint()
{
}

//==========================================================================
//
//==========================================================================
void coRestraint::add(ssize_t mi, ssize_t ma)
{
   stringCurrent = false;
   changed = true;
   all = false;
   min.push_back(mi);
   max.push_back(ma);
}


//==========================================================================
//
//==========================================================================
void coRestraint::add(ssize_t val)
{
   stringCurrent = false;
   changed = true;
   all = false;
   min.push_back(val);
   max.push_back(val);
}


//==========================================================================
//
//==========================================================================
void coRestraint::add(const std::string &selection)
{
   stringCurrent = false;
   changed = true;

   if (selection == "all") {
      min.clear();
      max.clear();
      all = true;
      return;
   }

   all = false;

   const char *c=selection.c_str();
   while(*c && !isdigit(*c))
      ++c;

   while (*c) {
      int inc=0;
      ssize_t dumMax, dumMin;
      ssize_t numNumbers = sscanf(c,"%zd-%zd%n",&dumMin,&dumMax,&inc);
      if(numNumbers>0) {
         if(numNumbers==1) {
            dumMax=dumMin;
            if(inc == 0) {
               // inc is 0 at least on windows if only one number is read
               while(*c && isdigit(*c))
                  ++c;
            }
         }
         min.push_back(dumMin);
         max.push_back(dumMax);
      }
        else
        {
            fprintf(stderr, "error parsing string: %s in coRestraint::add", selection);
            inc = 1;
        }
      c += inc;
      while(*c && !isdigit(*c))
         ++c;
   }
}


//==========================================================================
//
//==========================================================================
void coRestraint::clear()
{
   stringCurrent = false;
   changed = true;
   all = false;
   min.clear();
   max.clear();
}

//==========================================================================
//
//==========================================================================
void coRestraint::cut()
{
   stringCurrent = false;
   changed = true;
   all = false;
    min.pop_back();
    max.pop_back();
}

//==========================================================================
//
//==========================================================================
ssize_t coRestraint::lower() const
{
   size_t i=0;
   ssize_t low;
   if (!min.empty())
      low = min[0];
   else
      return -1;
   while (i<min.size())
   {
      if ( (low>=min[i]) )
      {
         low = min[i];
      }
      i++;
   }
   return low;
}


//==========================================================================
//
//==========================================================================
ssize_t coRestraint::upper() const
{
	size_t i = 0;
	ssize_t up;
   if (!max.empty())
      up = max[0];
   else
      return -1;
   while (i<max.size())
   {
      if ( (up<=max[i]) )
      {
         up = max[i];
      }
      ++i;
   }
   return up;
}


//==========================================================================
//
//==========================================================================
bool coRestraint::operator ()(ssize_t val) const
{
   if (all)
      return true;

   size_t i=0;
   while (i<min.size())
   {
      if ( (val>=min[i]) && (val<=max[i]) )
         return true;
      i++;
   }
   return false;
}


//==========================================================================
//
//==========================================================================
bool coRestraint::get(ssize_t val, ssize_t &group) const
{
   if (all) {
      group = -1;
      return true;
   }

   group=0;
    while (size_t(group) < min.size())
   {
      if ( (val>=min[group]) && (val<=max[group]) )
         return true;
      group++;
   }
   return false;
}

//==========================================================================
//
//==========================================================================

const std::string &coRestraint::getRestraintString() const
{
    if (!stringCurrent)
   {
        stringCurrent = true;
      restraintString = getRestraintString(getValues());
   }
   return restraintString;
}

const std::string coRestraint::getRestraintString(std::vector<ssize_t> sortedValues) const
{
    const size_t size = sortedValues.size();
   if (size == 0)
      return "";

   std::ostringstream restraintStream;
    ssize_t old = sortedValues[0];
    restraintStream << old;

    bool inSequence = false;
    for (size_t i = 1; i < size; ++i)
      {
        const ssize_t cur = sortedValues[i];
        if (cur == old + 1)
      {
            if (i == size-1)
         {
                restraintStream << "-" << cur;
      }
            inSequence = true;
      }
      else 
      {
            if (inSequence)
                restraintStream << "-" << old;
            restraintStream << ", " << cur;
            inSequence = false;
      }
        old = cur;
   }
   return restraintStream.str();
}

//==========================================================================
//
//==========================================================================
// function returns vector containing all integer indices
// that are specified by the string added to this coRestraint object
//
// returns an empty vector, if the evaluation of char array is 
// not successful
// 
const std::vector<ssize_t> &coRestraint::getValues() const
{
   if (changed)
   {
      changed = false;
      values.clear();
      //getting the indices
      ssize_t counter = lower();
      ssize_t limit = upper();
      if (limit == -1 || counter == -1)
      {
         values.clear();
      }
      else
      {
         while( counter <= limit )
         {
            if( operator()(counter) )
            {
               values.push_back(counter);
            }
            ++counter;
         }
      }
   }

   return values;
}

