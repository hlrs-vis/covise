/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class EnElement                       ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 05.06.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "EnElement.h"
#include "GeoFileAsc.h"
#include <do/coDoUnstructuredGrid.h>
//
// Constructor
//
EnElement::EnElement()
    : valid_(false)
    , empty_(true)
{
}

EnElement::EnElement(const string &name)
    : valid_(false)
    , empty_(false)
    , enTypeStr_(strip(name.substr(0, 10))) // the max. length of an ensight cell type is 10 (pyramid13)
{
    if (name.find("point") != string::npos)
    {
        valid_ = true;
        numCorn_ = 1;
        dim_ = D0;
        covType_ = TYPE_POINT;
        enType_ = point;
        //cerr << "EnElement::EnElement(..) point found" << endl;
    }
    else if (name.find("bar2") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) bar2 found" << endl;
        numCorn_ = 2;
        dim_ = D1;
        covType_ = TYPE_BAR;
        enType_ = bar2;
    }
    else if (name.find("bar3") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) bar3 found" << endl;
        numCorn_ = 3;
        dim_ = D1;
        enType_ = bar3;
    }
    else if (name.find("tria3") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) tria3 found" << endl;
        numCorn_ = 3;
        dim_ = D2;
        covType_ = TYPE_TRIANGLE;
        enType_ = tria3;
    }
    else if (name.find("tria6") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) tria6 found" << endl;
        numCorn_ = 6;
        dim_ = D2;
        enType_ = tria6;
    }
    else if (name.find("quad4") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) quad4 found" << endl;
        numCorn_ = 4;
        dim_ = D2;
        covType_ = TYPE_QUAD;
        enType_ = quad4;
    }
    else if (name.find("quad8") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) quad8 found" << endl;
        numCorn_ = 8;
        dim_ = D2;
        covType_ = TYPE_POINT;
        enType_ = quad8;
    }
    else if (name.find("tetra4") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) tetra4 found" << endl;
        numCorn_ = 4;
        dim_ = D3;
        covType_ = TYPE_TETRAHEDER;
        enType_ = tetra4;
    }
    else if (name.find("tetra10") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) tetra10 found" << endl;
        numCorn_ = 10;
        dim_ = D3;
        enType_ = tetra10;
    }
    else if (name.find("pyramid5") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) pyramid5 found" << endl;
        numCorn_ = 5;
        dim_ = D3;
        covType_ = TYPE_PYRAMID;
    }
    else if (name.find("pyramid13") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) pyramid13 found" << endl;
        numCorn_ = 13;
        dim_ = D3;
        covType_ = TYPE_POINT;
        enType_ = pyramid13;
    }
    else if (name.find("hexa8") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) hexa8  found" << endl;
        numCorn_ = 8;
        dim_ = D3;
        covType_ = TYPE_HEXAEDER;
        enType_ = hexa8;
    }
    else if (name.find("hexa20") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) hexa20 found" << endl;
        numCorn_ = 20;
        dim_ = D3;
        covType_ = TYPE_POINT;
        enType_ = hexa20;
    }
    else if (name.find("penta6") != string::npos)
    {
        valid_ = true;
        //cerr << "EnElement::EnElement(..) penta6  found" << endl;
        numCorn_ = 6;
        dim_ = D3;
        covType_ = TYPE_PRISM;
        enType_ = penta6;
    }
    else if (name.find("penta15") != string::npos)
    {
        valid_ = false;
        cerr << "EnElement::EnElement(..) pyramid15 found" << endl;
        numCorn_ = 15;
        dim_ = D3;
        covType_ = TYPE_POINT;
        enType_ = penta15;
    }
    else if (name.find("nsided") != string::npos)
    {
        valid_ = true;
        numCorn_ = 0; // not constant among the elements
        dim_ = D2;
        covType_ = TYPE_POINT; // not nescessary, as all 2D elements go into DO_Polygons
        enType_ = nsided;
    }
    else if (name.find("nfaced") != string::npos)
    {
        valid_ = true;
        numCorn_ = 0; // not constant among the elements
        dim_ = D3;
        covType_ = TYPE_POLYHEDRON;
        enType_ = nfaced;
    }
    // we allow a dummy element thats the only one which is empty
    else if (name.find("dummy") != string::npos)
    {
        valid_ = false;
        empty_ = true;
    }
}

EnElement::EnElement(const EnElement &e)
    : valid_(e.valid_)
    , empty_(e.empty_)
    , numCorn_(e.numCorn_)
    , dim_(e.dim_)
    , covType_(e.covType_)
    , enType_(e.enType_)
    , startIdx_(e.startIdx_)
    , endIdx_(e.endIdx_)
    , enTypeStr_(strip(e.enTypeStr_))
    , dataBlacklist_(e.dataBlacklist_)
{
}

const EnElement &
    EnElement::
    operator=(const EnElement &e)
{
    if (this == &e)
        return *this;

    valid_ = e.valid_;
    empty_ = e.empty_;
    numCorn_ = e.numCorn_;
    dim_ = e.dim_;
    covType_ = e.covType_;
    enType_ = e.enType_;
    startIdx_ = e.startIdx_;
    endIdx_ = e.endIdx_;
    enTypeStr_ = strip(e.enTypeStr_);
    dataBlacklist_ = e.dataBlacklist_;

    return *this;
}

// extend it later to Ensight elements which are not present in COVISE
// like hexa20...
// EnElement::remap(int *eleIn, int *eleOut, int *newTypes, int *newElem...)
int
EnElement::remap(int *cornIn, int *cornOut)
{
    // only 2D elements have to be remaped
    // at the first stage
    if (cornOut != NULL)
    {
        switch (enType_)
        {
        case tria3:
            cornOut[0] = cornIn[0];
            cornOut[1] = cornIn[2];
            cornOut[2] = cornIn[1];
            break;
        case quad4:
            cornOut[0] = cornIn[0];
            cornOut[1] = cornIn[3];
            cornOut[2] = cornIn[2];
            cornOut[3] = cornIn[1];
            break;
        default:
            memcpy(cornOut, cornIn, numCorn_ * sizeof(int));
            break;
        }
    }
    return numCorn_;
}

bool
EnElement::valid() const
{
    return valid_;
}

bool
EnElement::empty() const
{
    return empty_;
}

int
EnElement::getNumberOfCorners() const
{
    return numCorn_;
}

int
EnElement::getDim() const
{
    return dim_;
}

int
EnElement::getCovType() const
{
    return covType_;
}

string
EnElement::getEnTypeStr() const
{
    return enTypeStr_;
}

// return number and array of distinct corners
int
EnElement::distinctCorners(const int *ci, int *co) const
{
    if (ci != NULL)
    {
        int i, j;
        int distinct(numCorn_);
        bool fnd;
        for (i = 0; i < numCorn_; ++i)
        {
            fnd = true;
            for (j = i + 1; j < numCorn_; ++j)
            {
                fnd = fnd && (ci[i] != ci[j]);
            }
            if (!fnd)
                distinct--;
            else
                co[i] = ci[i];
        }
        // 	if ( distinct != numCorn_ ) {
        //  	    cerr << "EnElement::distinctCorners(..) expected "
        //  		 << numCorn_
        //  		 << " got " << distinct << " corners" << endl;
        // 	}

        return distinct;
    }
    return false;
}

void
EnElement::setBlacklist(const vector<int> &bl)
{
    dataBlacklist_ = bl;
}

vector<int>
EnElement::getBlacklist() const
{
    return dataBlacklist_;
}

//
// Destructor
//
EnElement::~EnElement()
{
}
