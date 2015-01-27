/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    EnElement
//
// Description: general data class for the handling of ensight geometry elements
//
// CLASS    EnPart
//
// Description: general data class for the handling of parts of Ensight geometry

//
// Initial version: 05.06.2002
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//
#ifndef ENELEMENT_H
#define ENELEMENT_H

#include <util/coviseCompat.h>

class EnPart;

typedef vector<EnPart> PartList;

// helper strip off spaces
string strip(const string &str);

//
// describes a Ensight element as class of elements
// it contains NO specific information about real elements (like corner indices)
// in out model but serves as a skelton for the elemts in a given Ensight part
//
class EnElement
{
public:
    enum
    {
        D0,
        D1,
        D2,
        D3
    };
    enum
    {
        point,
        bar2,
        bar3,
        tria3,
        tria6,
        quad4,
        quad8,
        tetra4,
        tetra10,
        pyramid5,
        pyramid13,
        hexa8,
        hexa20,
        penta6,
        penta15,
        nsided,
        nfaced
    };

    /// default CONSTRUCTOR
    EnElement();

    EnElement(const string &name);

    /// copy constructor
    EnElement(const EnElement &e);

    const EnElement &operator=(const EnElement &e);

    /// DESTRUCTOR
    ~EnElement();

    // return the dimensionality of element
    // i.e. 2D 3D
    int getDim() const;

    // is it a valid ENSIGHT element
    bool valid() const;

    // return the number of corners
    int getNumberOfCorners() const;

    // return COVISE type
    int getCovType() const;

    bool empty() const;

    // remap: either resort element corners or make new connectivity
    int remap(int *cornIn, int *cornOut);

    // return ENSIGHT type as a string
    string getEnTypeStr() const;

    // returns true if cell is fully degenerated i.e. a point
    int distinctCorners(const int *ci, int *co) const;

    void setBlacklist(const vector<int> &bl);

    vector<int> getBlacklist() const;

private:
    bool valid_;
    bool empty_;
    int numCorn_;
    int dim_;
    int covType_;
    int enType_;

    int startIdx_;
    int endIdx_;

    string enTypeStr_;

    vector<int> dataBlacklist_;
};

#endif
