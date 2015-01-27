/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    PyFile
//
// Description: convert a Covise net-file to a python script
//
// Initial version: 21.03.2003
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002/2003 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//  Changes:
//

#ifndef PYFILE_H
#define PYFILE_H

#include "NetFile.h"

class PyFile : public NetFile
{
public:
    /// default CONSTRUCTOR
    PyFile();
    PyFile(Skeletons *skels, ostream &sDiag = cerr);

    /// DESTRUCTOR
    ~PyFile();

    // write the converted output to s
    friend ostream &operator<<(ostream &s, const PyFile &f);

private:
    // helper: create argument string for a given paramter
    std::string createArgumentStr(const ParamSkel &p) const;

    std::string pyNetName_;
};
#endif
