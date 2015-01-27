/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WCE_RESTARINT_H
#define WCE_RESTARINT_H
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
 **                Computing Center University of Stuttgart                **
 **                            Allmandring 30                              **
 **                            70550 Stuttgart                             **
 ** Author:                                                                **
 ** Date:                                                                  **
\**************************************************************************/

#include <util/common.h>
#include <vector>

namespace covise
{
class coRestraint
{
private:
    mutable std::vector<int> values, min, max;
    mutable bool changed, stringChanged;
    mutable std::string restraintString;

public:
    coRestraint()
        : changed(true)
        , stringChanged(false){};
    ~coRestraint(){};

    void add(int mi, int ma);
    void add(int val);
    void add(const char *selection);
    int get(int val, int &group) const;
    int getNumGroups() const
    {
        return min.size();
    };
    void clear();
    const std::vector<int> &getValues() const;
    int lower() const;
    int upper() const;
    const std::string &getRestraintString() const;
    const std::string getRestraintString(std::vector<int>) const;

    // operators
    int operator()(int val) const;
};
}
#endif // COVISE_RESTRAINT_H
