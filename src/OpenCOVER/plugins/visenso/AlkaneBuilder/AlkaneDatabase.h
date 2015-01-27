/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ALKANE_DATABASE_H
#define _ALKANE_DATABASE_H

#include <iostream>
#include <vector>

struct Alkane
{
    Alkane(std::string f, std::string n, bool l, int c, int h);

    std::string formula, name;
    bool linear;
    int carbons, hydrogens;
};

class AlkaneDatabase
{
public:
    static AlkaneDatabase *Instance();
    Alkane findByFormula(std::string formula);
    Alkane findByAtoms(int nc, int nh);
    Alkane findByAtoms(int nc, int nh, bool linear);
    std::vector<Alkane> alkanes;

private:
    AlkaneDatabase(); // constructor
    static AlkaneDatabase *instance_;
};

#endif
