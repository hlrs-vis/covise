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
#include "wce_Restraint.h"

using namespace covise;
//==========================================================================
//
//==========================================================================
void coRestraint::add(int mi, int ma)
{
    stringChanged = false;
    changed = true;
    min.push_back(mi);
    max.push_back(ma);
}

//==========================================================================
//
//==========================================================================
void coRestraint::add(int val)
{
    stringChanged = false;
    changed = true;
    min.push_back(val);
    max.push_back(val);
}

//==========================================================================
//
//==========================================================================
void coRestraint::add(const char *selection)
{
    stringChanged = false;
    changed = true;
    const char *c = selection;
    while (*c && (*c < '0' || *c > '9'))
        c++;
    while (*c)
    {
        int inc = 0, dumMax, dumMin;
        int numNumbers = sscanf(c, "%d-%d%n", &dumMin, &dumMax, &inc);
        if (numNumbers > 0)
        {
            if (numNumbers == 1)
            {
                dumMax = dumMin;
                if (inc == 0) // inc is 0 at least on windows if only one number is read
                {
                    while (*c && (*c >= '0' && *c <= '9'))
                        c++;
                }
            }
            min.push_back(dumMin);
            max.push_back(dumMax);
        }
        c += inc;
        while (*c && (*c < '0' || *c > '9'))
            c++;
    }
}

//==========================================================================
//
//==========================================================================
void coRestraint::clear()
{
    stringChanged = false;
    changed = true;
    min.clear();
    max.clear();
}

//==========================================================================
//
//==========================================================================
int coRestraint::lower() const
{
    int i = 0, low;
    if (!min.empty())
        low = min[0];
    else
        return -1;
    while (i < min.size())
    {
        if ((low >= min[i]))
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
int coRestraint::upper() const
{
    int i = 0, up;
    if (!max.empty())
        up = max[0];
    else
        return -1;
    while (i < max.size())
    {
        if ((up <= max[i]))
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
int coRestraint::operator()(int val) const
{
    int i = 0;
    while (i < min.size())
    {
        if ((val >= min[i]) && (val <= max[i]))
            return 1;
        i++;
    }
    return 0;
}

//==========================================================================
//
//==========================================================================
int coRestraint::get(int val, int &group) const
{
    group = 0;
    while (group < min.size())
    {
        if ((val >= min[group]) && (val <= max[group]))
            return 1;
        group++;
    }
    return 0;
}

//==========================================================================
//
//==========================================================================

const std::string &coRestraint::getRestraintString() const
{
    if (!stringChanged)
    {
        stringChanged = true;
        restraintString = getRestraintString(getValues());
    }
    return restraintString;
}

const std::string coRestraint::getRestraintString(std::vector<int> sortedValues) const
{
    std::ostringstream restraintStream;
    int old = -1, size = sortedValues.size();
    bool started = false, firsttime = true;
    if (size == 0)
        return "";
    for (int i = 0; i < size; ++i)
    {
        int actual = sortedValues[i];
        if (firsttime)
        {
            firsttime = false;
            restraintStream << actual;
            old = actual;
            continue;
        }
        else if (actual == old + 1 && i < size - 1)
        {
            if (!started)
            {
                restraintStream << "-";
                started = true;
            }
            old = actual;
            continue;
        }
        else if (started)
        {
            started = false;
            restraintStream << old << ", " << actual;
        }
        else
        {
            restraintStream << ", " << actual;
        }
        old = actual;
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
const std::vector<int> &coRestraint::getValues() const
{
    if (changed)
    {
        changed = false;
        values.clear();
        //getting the indices
        int counter = lower();
        int limit = upper();
        if (limit == -1 || counter == -1)
        {
            values.clear();
        }
        else
        {
            while (counter <= limit)
            {
                if (operator()(counter))
                {
                    values.push_back(counter);
                }
                ++counter;
            }
        }
    }

    return values;
}
