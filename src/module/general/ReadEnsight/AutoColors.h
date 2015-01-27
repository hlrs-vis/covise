/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    AutoColors
//
// Description: add colors automatically to 2d parts of a given model
//
// Initial version: dd.mm.2005
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2005 by VISENSO GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//

#ifndef AUTOCOLORS_H
#define AUTOCOLORS_H

#include <string>
#include <vector>

using namespace std;

class AutoColors
{
public:
    /// The one and only access to this object.
    //
    /// @return        pointer to the one and only instance
    static AutoColors *instance();

    string next();

    void reset();

private:
    /// default CONSTRUCTOR
    AutoColors();

    /// DESTRUCTOR
    ~AutoColors();

    static AutoColors *instance_;
    int idx_;
    vector<string> colors_;
};

#endif
