/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// CLASS    EnPart
//
// Description: general data class for the handling of parts of Ensight geometry

//
// Initial version: 27.06.2005
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2005 by VISENSO GmbH
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//
#ifndef ENPART_H
#define ENPART_H

#include <util/coviseCompat.h>
#include <do/coDistributedObject.h>
#include "EnElement.h"

using namespace covise;

class EnPart;
typedef vector<EnPart> PartList;

//
// object to allow bookkeeping of parts
//
class EnPart
{
public:
    EnPart(const int &pNum, const string &comment = "");
    EnPart();
    EnPart(const EnPart &p);

    const EnPart &operator=(const EnPart &p);

    // add elemet and number of elements to part data
    void addElement(const EnElement &ele, const int &anz);

    // print out part object to stderr - debug
    void print() const;

    // returns an string of the form
    // #part |  part comment | number of elements
    string partInfoString(const int &ref = 0) const;

    void setPartNum(const int &partNum);
    // returns part number
    int getPartNum() const;

    // a part object is empty is it got not filled yet
    bool isEmpty() const;

    // find Element by name
    // we assume that each element occurs only once
    // in a given part
    EnElement findElement(const string &name) const;

    // returns number of cells per element
    uint64_t getElementNum(const string &name) const;

    // return the total number of corners in *this
    uint64_t getTotNumberOfCorners() const;
    uint64_t getTotNumCorners2d();
    uint64_t getTotNumCorners3d();

    // return the total number of elements contained in *this
    uint64_t getTotNumEle() const;
    uint64_t getTotNumEle2d() const;
    uint64_t getTotNumEle3d() const;

    // return the total number of elements contained in *this
    // havig a dimensionality dim
    uint64_t getTotNumEle(const int &dim);

    // return the number of different element-types contained in *this
    size_t getNumEle() const;

    void setComment(const string &comm);
    void setComment(const char *ch);
    string comment() const;

    void activate(const bool &b = true);

    bool isActive() const;

    // set number of Coords - needed for GOLD
    void setNumCoords(const uint64_t &n);

    uint64_t numCoords();

    uint64_t numEleRead2d() const;
    uint64_t numEleRead3d() const;
    void setNumEleRead2d(const uint64_t &n);
    void setNumEleRead3d(const uint64_t &n);

    uint64_t numConnRead2d() const;
    uint64_t numConnRead3d() const;
    void setNumConnRead2d(const uint64_t &n);
    void setNumConnRead3d(const uint64_t &n);

    // delete all fields (see below) and reset pointers to NULL
    void clearFields();

    // these pointers have to be used with care
    float *arr1_, *arr2_, *arr3_;
    float *d2dx_, *d2dy_, *d2dz_;
    float *d3dx_, *d3dy_, *d3dz_;
    float *x2d_, *y2d_, *z2d_;
    float *x3d_, *y3d_, *z3d_;

    int *el2d_, *cl2d_, *tl2d_;
    int *el3d_, *cl3d_, *tl3d_;
    int *indexMap2d_, *indexMap3d_;
    coDistributedObject *distGeo2d_;
    coDistributedObject *distGeo3d_;

    vector<int> subParts_numElem, subParts_numConn, subParts_numCoord;
    vector<const int*>  subParts_IndexList;

private:
    int partNum_;
    vector<EnElement> elementList_;
    vector<int> numList_;
    vector<int> numList2d_;
    vector<int> numList3d_;
    bool empty_;
    int counter_;
    string comment_;
    // if true part will be used to build up covise data
    bool active_;
    uint64_t numCoords_;

    uint64_t numEleRead2d_;
    uint64_t numEleRead3d_;
    uint64_t numConnRead2d_;
    uint64_t numConnRead3d_;
};
#endif
