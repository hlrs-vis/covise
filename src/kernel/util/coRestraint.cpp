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
#include <cassert>

using namespace covise;

//==========================================================================
//
//==========================================================================
coRestraint::coRestraint()
: all(false)
, globalStep(1)
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

size_t coRestraint::getNumGroups() const
{
    assert(min.size() == max.size());
    assert(min.size() == step.size());
    return min.size();
}

//==========================================================================
//
//==========================================================================
void coRestraint::add(ssize_t mi, ssize_t ma, ssize_t st)
{
   stringCurrent = false;
   changed = true;
   all = false;
   min.push_back(mi);
   max.push_back(ma);
   step.push_back(st);
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
   step.push_back(1);
}


//==========================================================================
//
//==========================================================================
void coRestraint::add(const std::string &selection)
{
   stringCurrent = false;
   changed = true;

   const char *c=selection.c_str();

   if (selection.substr(0,3) == "all") {
      min.clear();
      max.clear();
      step.clear();
      globalStep = 1;
      all = true;
      ssize_t dumStep=1;
      ssize_t numNumbers = sscanf(c,"all/%zd",&dumStep);
      if (numNumbers == 1)
          globalStep = (int)dumStep;
      if (globalStep <= 0)
          globalStep = 1;
      return;
   }

   all = false;

   while(*c && !isdigit(*c))
      ++c;
   while (*c) {
      int inc=0;
      ssize_t dumMax, dumMin, dumStep=1;
      ssize_t numNumbers = sscanf(c,"%zd-%zd/%zd%n",&dumMin,&dumMax,&dumStep,&inc);
      if(numNumbers>0) {
         if(numNumbers==1) {
            dumMax=dumMin;
         }
         if(inc == 0) {
             // inc is 0 at least on windows if only one number is read
             while(*c && (isdigit(*c) || *c=='-' || *c=='/'))
                 ++c;
         }
         if (numNumbers<3)
             dumStep=1;
         min.push_back(dumMin);
         max.push_back(dumMax);
         step.push_back(dumStep);
      }
      else
      {
          fprintf(stderr, "error parsing string: %s in coRestraint::add", selection.c_str());
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
   globalStep = 1;
   min.clear();
   max.clear();
   step.clear();
}

//==========================================================================
//
//==========================================================================
void coRestraint::cut()
{
   stringCurrent = false;
   changed = true;
   all = false;
   if (!min.empty())
   {
       min.pop_back();
       max.pop_back();
       step.pop_back();
   }
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
   ssize_t group = -1;
   return get(val, group);
}


//==========================================================================
//
//==========================================================================
bool coRestraint::get(ssize_t val, ssize_t &group) const
{
   if (all) {
      group = -1;
      return (val % globalStep) == 0;
   }

   for (group=0; size_t(group) < min.size(); ++group)
   {
      if ( (val>=min[group]) && (val<=max[group]) )
      {
         return (val - (min[group]-1)) % step[group] == 0;
      }
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

