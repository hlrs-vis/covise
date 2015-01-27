/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                               (C)2005 VISENSO GmbH  ++
// ++ Description:                                                        ++
// ++             Implementation of class EnPart                          ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@visenso.de)                            ++
// ++                                                                     ++
// ++               VISENSO GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 05.06.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "EnPart.h"
#include <numeric>

EnPart::EnPart()
    : arr1_(NULL)
    , arr2_(NULL)
    , arr3_(NULL)
    , d2dx_(NULL)
    , d2dy_(NULL)
    , d2dz_(NULL)
    , d3dx_(NULL)
    , d3dy_(NULL)
    , d3dz_(NULL)
    , x2d_(NULL)
    , y2d_(NULL)
    , z2d_(NULL)
    , x3d_(NULL)
    , y3d_(NULL)
    , z3d_(NULL)
    , el2d_(NULL)
    , cl2d_(NULL)
    , tl2d_(NULL)
    , el3d_(NULL)
    , cl3d_(NULL)
    , tl3d_(NULL)
    , indexMap2d_(NULL)
    , indexMap3d_(NULL)
    , distGeo2d_(NULL)
    , distGeo3d_(NULL)
    , partNum_(-1)
    , empty_(true)
    , active_(true)
    , numCoords_(0)
    , numEleRead2d_(0)
    , numEleRead3d_(0)
    , numConnRead2d_(0)
    , numConnRead3d_(0)
{
}

EnPart::EnPart(const int &pNum, const string &comment)
    : arr1_(NULL)
    , arr2_(NULL)
    , arr3_(NULL)
    , d2dx_(NULL)
    , d2dy_(NULL)
    , d2dz_(NULL)
    , d3dx_(NULL)
    , d3dy_(NULL)
    , d3dz_(NULL)
    , x2d_(NULL)
    , y2d_(NULL)
    , z2d_(NULL)
    , x3d_(NULL)
    , y3d_(NULL)
    , z3d_(NULL)
    , el2d_(NULL)
    , cl2d_(NULL)
    , tl2d_(NULL)
    , el3d_(NULL)
    , cl3d_(NULL)
    , tl3d_(NULL)
    , indexMap2d_(NULL)
    , indexMap3d_(NULL)
    , distGeo2d_(NULL)
    , distGeo3d_(NULL)
    , partNum_(pNum)
    , empty_(true)
    , comment_(comment)
    , active_(true)
    , numCoords_(0)
    , numEleRead2d_(0)
    , numEleRead3d_(0)
    , numConnRead2d_(0)
    , numConnRead3d_(0)
{
}

EnPart::EnPart(const EnPart &p)
    : arr1_(p.arr1_)
    , arr2_(p.arr2_)
    , arr3_(p.arr3_)
    , d2dx_(p.d2dx_)
    , d2dy_(p.d2dy_)
    , d2dz_(p.d2dz_)
    , d3dx_(p.d3dx_)
    , d3dy_(p.d3dy_)
    , d3dz_(p.d3dz_)
    , x2d_(p.x2d_)
    , y2d_(p.y2d_)
    , z2d_(p.z2d_)
    , x3d_(p.x3d_)
    , y3d_(p.y3d_)
    , z3d_(p.z3d_)
    , el2d_(p.el2d_)
    , cl2d_(p.cl2d_)
    , tl2d_(p.tl2d_)
    , el3d_(p.el3d_)
    , cl3d_(p.cl3d_)
    , tl3d_(p.tl3d_)
    , indexMap2d_(p.indexMap2d_)
    , indexMap3d_(p.indexMap3d_)
    , distGeo2d_(p.distGeo2d_)
    , distGeo3d_(p.distGeo3d_)
    , partNum_(p.partNum_)
    , empty_(p.empty_)
    , comment_(p.comment_)
    , active_(p.active_)
    , numCoords_(p.numCoords_)
    , numEleRead2d_(p.numEleRead2d_)
    , numEleRead3d_(p.numEleRead3d_)
    , numConnRead2d_(p.numConnRead2d_)
    , numConnRead3d_(p.numConnRead3d_)
{
    unsigned int i;
    for (i = 0; i < p.elementList_.size(); ++i)
    {
        elementList_.push_back(p.elementList_[i]);
        numList_.push_back(p.numList_[i]);
    }
    for (i = 0; i < p.subParts_numElem.size(); ++i)
        subParts_numElem.push_back(p.subParts_numElem[i]);
    for (i = 0; i < p.subParts_numConn.size(); ++i)
        subParts_numConn.push_back(p.subParts_numConn[i]);
}

const EnPart &
    EnPart::
    operator=(const EnPart &p)
{
    if (this == &p)
        return *this;

    partNum_ = p.partNum_;
    empty_ = p.empty_;
    arr1_ = p.arr1_;
    arr2_ = p.arr2_;
    arr3_ = p.arr3_;
    comment_ = p.comment_;
    active_ = p.active_;
    numCoords_ = p.numCoords_;
    numEleRead2d_ = p.numEleRead2d_;
    numConnRead2d_ = p.numConnRead2d_;
    numEleRead3d_ = p.numEleRead3d_;
    numConnRead3d_ = p.numConnRead3d_;

    distGeo2d_ = p.distGeo2d_;
    distGeo3d_ = p.distGeo3d_;

    el2d_ = p.el2d_;
    cl2d_ = p.cl2d_;
    tl2d_ = p.tl2d_;

    el3d_ = p.el3d_;
    cl3d_ = p.cl3d_;
    tl3d_ = p.tl3d_;

    x2d_ = p.x2d_;
    y2d_ = p.y2d_;
    z2d_ = p.z2d_;
    x3d_ = p.x3d_;
    y3d_ = p.y3d_;
    z3d_ = p.z3d_;

    d2dx_ = p.d2dx_;
    d2dy_ = p.d2dy_;
    d2dz_ = p.d2dz_;
    d3dx_ = p.d3dx_;
    d3dy_ = p.d3dy_;
    d3dz_ = p.d3dz_;

    indexMap2d_ = p.indexMap2d_;
    indexMap3d_ = p.indexMap3d_;

    elementList_.clear();
    numList_.clear();

    unsigned int i;
    for (i = 0; i < p.elementList_.size(); ++i)
    {
        elementList_.push_back(p.elementList_[i]);
        numList_.push_back(p.numList_[i]);
    }

    subParts_numElem.clear();
    for (i = 0; i < p.subParts_numElem.size(); ++i)
        subParts_numElem.push_back(p.subParts_numElem[i]);
    subParts_numConn.clear();
    for (i = 0; i < p.subParts_numConn.size(); ++i)
        subParts_numConn.push_back(p.subParts_numConn[i]);

    return *this;
}

void
EnPart::setComment(const string &c)
{
    comment_ = c;
}

void
EnPart::setComment(const char *ch)
{
    comment_ = ch;
}

string
EnPart::comment() const
{
    return comment_;
}

void
EnPart::addElement(const EnElement &ele, const int &anz)
{
    // we add only valid elements
    if (ele.valid())
    {
        elementList_.push_back(ele);
        numList_.push_back(anz);
        if (ele.getDim() == EnElement::D2)
            numList2d_.push_back(anz);
        if (ele.getDim() == EnElement::D3)
            numList3d_.push_back(anz);
        empty_ = false;
    }
}

void
EnPart::print() const
{
    unsigned int i;

    cerr << "PART " << partNum_ << endl;
    cerr << "   COMMENT: " << comment_ << endl;

    for (i = 0; i < elementList_.size(); ++i)
    {
        cerr << "   " << numList_[i] << " elements of type " << elementList_[i].getCovType() << endl;
    }
    cerr << " active " << active_ << endl;

    cerr << "   arr1 " << arr1_ << endl;
    cerr << "   arr2 " << arr2_ << endl;
    cerr << "   arr3 " << arr3_ << endl;

    cerr << "   el2d   " << el2d_ << endl;
    cerr << "   cl2d   " << cl2d_ << endl;

    cerr << "   numEleRead3d  " << numEleRead3d_
         << "   numEleRead2d  " << numEleRead2d_ << endl;

    cerr << "   numConnRead3d " << numConnRead3d_
         << "   numConnRead2d " << numConnRead2d_ << endl;
}

void
EnPart::setPartNum(const int &partNum)
{
    partNum_ = partNum;
}

int
EnPart::getPartNum() const
{
    return partNum_;
}

EnElement
EnPart::findElement(const string &name) const
{
    unsigned int i;
    for (i = 0; i < elementList_.size(); ++i)
    {
        if (elementList_[i].getEnTypeStr() == name)
        {
            return elementList_[i];
        }
    }
    EnElement e;
    return e;
}

int
EnPart::getElementNum(const string &name) const
{
    unsigned int i;
    int ret(0);
    for (i = 0; i < elementList_.size(); ++i)
    {
        if (elementList_[i].getEnTypeStr() == name)
        {
            ret = numList_[i];
            break;
        }
    }
    return ret;
}

int
EnPart::getTotNumEle() const
{
    return std::accumulate(numList_.begin(), numList_.end(), 0);
}

int
EnPart::getTotNumEle2d() const
{
    return std::accumulate(numList2d_.begin(), numList2d_.end(), 0);
}

int
EnPart::getTotNumEle3d() const
{
    return std::accumulate(numList3d_.begin(), numList3d_.end(), 0);
}

int
EnPart::getTotNumCorners2d() // doesnt work with nsided since that element type doesnt have a specific number of corners
{
    int ret = 0, anz = 0, nc = 0;
    vector<int>::iterator it(numList2d_.begin());
    vector<EnElement>::iterator itEle(elementList_.begin());
    while (it != numList2d_.end())
    {
        anz = *it;
        nc = itEle->getNumberOfCorners();
        ret += nc * anz;
        ++it;
        ++itEle;
    }
    return ret;
}

int
EnPart::getTotNumCorners3d() // doesnt work with nfaced since that element type doesnt have a specific number of corners
{
    int ret = 0, anz = 0, nc = 0;
    vector<int>::iterator it(numList3d_.begin());
    vector<EnElement>::iterator itEle(elementList_.begin());
    while (it != numList3d_.end())
    {
        anz = *it;
        nc = itEle->getNumberOfCorners();
        ret += nc * anz;
        ++it;
        ++itEle;
    }
    return ret;
}

int
EnPart::getNumEle() const
{
    return elementList_.size();
}

string
EnPart::partInfoString(const int &ref) const
{
    string infoStr;
    // ref No.
    char nStr[32];
    sprintf(nStr, "%8d", ref);
    infoStr += nStr;
    infoStr += " | ";

    // part No.
    char rStr[32];
    sprintf(rStr, "%8d", partNum_);
    infoStr += rStr;
    infoStr += " | ";

    // comment
    unsigned int j;
    string outComment(comment_);
    const int ensightMaxStringLen(80); // max. length of ensight comment is 80

    for (j = outComment.size(); j < ensightMaxStringLen; ++j)
        outComment += " ";

    infoStr += outComment;
    infoStr += " | ";

    // total number of elements
    int nTot(0);
    for (j = 0; j < elementList_.size(); ++j)
        nTot += numList_[j];
    sprintf(nStr, "%8d", nTot);
    infoStr += nStr;
    infoStr += " | ";
    if ((numEleRead2d_ > 0) && (numEleRead3d_ > 0))
        infoStr += "    2D/3D    ";
    else if ((numEleRead2d_ == 0) && (numEleRead3d_ == 0) && (numCoords_ > 0))
        infoStr += "     2D     "; // HACK: if no elements read, output coordinates as points at the 2D port
    else if (numEleRead2d_ > 0)
        infoStr += "     2D     ";
    else if (numEleRead3d_ > 0)
        infoStr += "     3D     ";
    return infoStr;
}

void
EnPart::activate(const bool &b)
{
    active_ = b;
}

bool
EnPart::isActive() const
{
    return active_;
}

int
EnPart::getTotNumberOfCorners() const
{
    unsigned int i;
    int numCorn(0), nc;
    for (i = 0; i < elementList_.size(); ++i)
    {
        nc = elementList_[i].getNumberOfCorners();
        nc *= numList_[i];
        numCorn += nc;
    }
    return numCorn;
}

void
EnPart::setNumCoords(const int &n)
{
    numCoords_ = n;
}

int
EnPart::numCoords()
{
    return numCoords_;
}

int
EnPart::numEleRead2d() const
{
    return numEleRead2d_;
}

int
EnPart::numEleRead3d() const
{
    return numEleRead3d_;
}

void
EnPart::setNumEleRead2d(const int &n)
{
    numEleRead2d_ = n;
}

void
EnPart::setNumEleRead3d(const int &n)
{
    numEleRead3d_ = n;
}

int
EnPart::numConnRead2d() const
{
    return numConnRead2d_;
}

int
EnPart::numConnRead3d() const
{
    return numConnRead3d_;
}

void
EnPart::setNumConnRead2d(const int &n)
{
    numConnRead2d_ = n;
}

void
EnPart::setNumConnRead3d(const int &n)
{
    numConnRead3d_ = n;
}

void
EnPart::clearFields()
{
    if (arr1_ != NULL)
        delete[] arr1_;
    arr1_ = NULL;
    if (arr2_ != NULL)
        delete[] arr2_;
    arr2_ = NULL;
    if (arr3_ != NULL)
        delete[] arr3_;
    arr3_ = NULL;
    if (d2dx_ != NULL)
        delete[] d2dx_;
    d2dx_ = NULL;
    if (d2dy_ != NULL)
        delete[] d2dy_;
    d2dy_ = NULL;
    if (d2dz_ != NULL)
        delete[] d2dz_;
    d2dz_ = NULL;
    if (d3dx_ != NULL)
        delete[] d3dx_;
    d3dx_ = NULL;
    if (d3dy_ != NULL)
        delete[] d3dy_;
    d3dy_ = NULL;
    if (d3dz_ != NULL)
        delete[] d3dz_;
    d3dz_ = NULL;
    if (x2d_ != NULL)
        delete[] x2d_;
    x2d_ = NULL;
    if (y2d_ != NULL)
        delete[] y2d_;
    y2d_ = NULL;
    if (z2d_ != NULL)
        delete[] z2d_;
    z2d_ = NULL;
    if (x3d_ != NULL)
        delete[] x3d_;
    x3d_ = NULL;
    if (y3d_ != NULL)
        delete[] y3d_;
    y3d_ = NULL;
    if (z3d_ != NULL)
        delete[] z3d_;
    z3d_ = NULL;
    if (el2d_ != NULL)
        delete[] el2d_;
    el2d_ = NULL;
    if (cl2d_ != NULL)
        delete[] cl2d_;
    cl2d_ = NULL;
    if (tl2d_ != NULL)
        delete[] tl2d_;
    tl2d_ = NULL;
    if (el3d_ != NULL)
        delete[] el3d_;
    el3d_ = NULL;
    if (cl3d_ != NULL)
        delete[] cl3d_;
    cl3d_ = NULL;
    if (tl3d_ != NULL)
        delete[] tl3d_;
    tl3d_ = NULL;
    if (indexMap2d_ != NULL)
        delete[] indexMap2d_;
    indexMap2d_ = NULL;
    if (indexMap3d_ != NULL)
        delete[] indexMap3d_;
    indexMap3d_ = NULL;
}
