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
    int getElementNum(const string &name) const;

    // return the total number of corners in *this
    int getTotNumberOfCorners() const;
    int getTotNumCorners2d();
    int getTotNumCorners3d();

    // return the total number of elements contained in *this
    int getTotNumEle() const;
    int getTotNumEle2d() const;
    int getTotNumEle3d() const;

    // return the total number of elements contained in *this
    // havig a dimensionality dim
    int getTotNumEle(const int &dim);

    // return the number of different element-types contained in *this
    int getNumEle() const;

    void setComment(const string &comm);
    void setComment(const char *ch);
    string comment() const;

    void activate(const bool &b = true);

    bool isActive() const;

    // set number of Coords - needed for GOLD
    void setNumCoords(const int &n);

    int numCoords();

    int numEleRead2d() const;
    int numEleRead3d() const;
    void setNumEleRead2d(const int &n);
    void setNumEleRead3d(const int &n);

    int numConnRead2d() const;
    int numConnRead3d() const;
    void setNumConnRead2d(const int &n);
    void setNumConnRead3d(const int &n);

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

    vector<int> subParts_numElem, subParts_numConn;

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
    int numCoords_;

    int numEleRead2d_;
    int numEleRead3d_;
    int numConnRead2d_;
    int numConnRead3d_;
};
#endif
