/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#define COIDENT "$Header: /vobs/covise/src/application/general/READ_PAM/aux.cpp /main/vir_main/1 18-Dec-2001.11:14:10 we_te $"
#include <util/coIdent.h>

#include "auxiliary.h"
#include <limits.h>

void coStringObj::fill_index(const INDEX ind)
{
    int i;
    for (i = 0; i < 9; ++i)
    {
        index_[i] = ind[i];
    }
}

coStringObj::coStringObj(const coStringObj &input)
{
    ObjType_ = input.ObjType_;
    ObjName_ = input.ObjName_;
    component_ = input.component_;
    position_ = input.position_;
    ElemType_ = input.ElemType_;
    fill_index(input.index_);
}

coStringObj &coStringObj::operator=(const coStringObj &rhs)
{
    if (this == &rhs)
        return *this;
    ObjType_ = rhs.ObjType_;
    ObjName_ = rhs.ObjName_;
    component_ = rhs.component_;
    position_ = rhs.position_;
    fill_index(rhs.index_);
    ElemType_ = rhs.ElemType_;
    return *this;
}

void whichContents::eliminate(int k) // eliminate element k
{
    int l;
    if (number_ <= 0)
        return;
    if (k < 0 || k >= number_)
        return;
    for (l = k + 1; l < number_; ++l)
    {
        theContents_[l - 1] = theContents_[l];
    }
    --number_;
}

int whichContents::operator==(whichContents &rhs)
{
    if (number_ != rhs.number_)
        return 0;
    int i;
    for (i = 0; i < number_; ++i)
        if (!(theContents_[i] == rhs.theContents_[i]))
            return 0;

    return 1;
}

void whichContents::reset()
{
    delete[] theContents_;
    theContents_ = new coStringObj[1];
    theContents_[0].ObjName_ = "none";
    theContents_[0].ObjType_ = coStringObj::NONE;
    theContents_[0].component_ = '0';
    theContents_[0].index_[0] = 0;
    theContents_[0].position_ = 0;
    length_ = 1;
    number_ = 1;
}

// before compression vector and tensor entries are only
// present in the form of single components
void whichContents::compress(int base)
{
    int i;
    for (i = 0; i < number_; ++i) // find where additional vars start
    {
        if (theContents_[i].index_[0] == base)
            break;
    }
    if (i == number_)
        return; // no additional vars
    int j, k;
    int compressed;
    for (j = i; j < number_; ++j)
    {
        if (theContents_[j].ObjType_ == coStringObj::SCALAR || theContents_[j].ObjType_ == coStringObj::NONE)
            continue;
        compressed = 0;
        int component_back = atoi(&theContents_[j].component_);
        if (component_back <= 0 || component_back > theContents_[j].ObjType_)
        {
            Covise::sendError("stale data");
            continue; // in fact this is an error!!!
        }
        theContents_[j].index_[component_back - 1] = theContents_[j].index_[0];
        for (k = j + 1; k < number_; ++k)
        {
            // look for other components of the same object
            if (theContents_[j].ObjName_ == theContents_[k].ObjName_ && theContents_[j].ElemType_ == theContents_[k].ElemType_ && theContents_[j].ObjType_ == theContents_[k].ObjType_)
            {

                if (atoi(&theContents_[k].component_) <= 0 || atoi(&theContents_[k].component_) > theContents_[j].ObjType_)
                {
                    Covise::sendError("stale data");
                    continue; // in fact this is an error!!!
                }

                theContents_[j].index_[atoi(&theContents_[k].component_) - 1] = theContents_[k].index_[0];
                ++compressed;
            }
        }
        if (compressed != theContents_[j].ObjType_ - 1 || theContents_[j].ObjType_ == coStringObj::TENSOR_2D || theContents_[j].ObjType_ == coStringObj::TENSOR_3D)
        {
            // var is not complete
            // do not execute compression
            // turn type to scalar because
            // we do not have all components
            // or var is a tensor -> not yet implemented
            theContents_[j].ObjType_ = coStringObj::SCALAR;
            /*
                  theContents_[j].ObjName_ += " Comp. ";
                  theContents_[j].ObjName_ += theContents_[j].component_;
         */
            // restore index
            theContents_[j].index_[0] = theContents_[j].index_[component_back - 1];
        } // do compress: eliminate compressed vars
        else
        {
            theContents_[j].component_ = '0'; // we have the full
            // object, not an isolated component
            for (k = j + 1; k < number_;)
            {
                if (theContents_[j].ObjName_ == theContents_[k].ObjName_ && theContents_[j].ElemType_ == theContents_[k].ElemType_ && theContents_[j].ObjType_ == theContents_[k].ObjType_)
                {
                    eliminate(k);
                }
                else
                {
                    ++k;
                }
            }
        }
    }
}

void whichContents::add(const char *new_option1, coStringObj::Type new_type,
                        coStringObj::ElemType new_elem_type, INDEX ind,
                        char component, int position)
{
    char *new_option = new char[strlen(new_option1) + 1];
    strcpy(new_option, new_option1);

    if ((new_type == coStringObj::VECTOR && (atoi(&component) < 0 || atoi(&component) > 3)) || (new_type == coStringObj::TENSOR_2D && (atoi(&component) < 0 || atoi(&component) > 4)) || (new_type == coStringObj::TENSOR_3D && (atoi(&component) < 0 || atoi(&component) > 9)))
        return;

    // do not add a repeated object
    int j;
    for (j = 0; j < number_; ++j)
    {
        if (strcmp(theContents_[j].ObjName_.c_str(), new_option) == 0 && theContents_[j].ObjType_ == new_type && theContents_[j].ElemType_ == new_elem_type && theContents_[j].component_ == component)
            return;
    }

    // ensure that "none" is always an option
    if (length_ == 0)
    {
        reset();
    }

    // prevent overflow
    if (number_ == length_)
    {
        coStringObj *tmp = new coStringObj[2 * number_];
        length_ = 2 * number_;
        int i;
        for (i = 0; i < number_; ++i)
        {
            tmp[i] = theContents_[i];
        }
        delete[] theContents_;
        theContents_ = tmp;
    }

    // add
    // search for '\0' char in new_option
    int i;
    for (i = 0; new_option[i] != '\0'; ++i)
        ;
    // substitite new lines by spaces
    for (j = 0; j < i; ++j)
    {
        if (new_option[j] == '\n')
            new_option[j] = ' ';
    }
    // eliminate trailing spaces
    for (j = i - 1; j >= 0; --j)
    {
        if (new_option[j] == ' ')
            new_option[j] = '\0';
        else
            break;
    }
    theContents_[number_].ObjName_ = new_option;
    theContents_[number_].ObjType_ = new_type;
    theContents_[number_].ElemType_ = new_elem_type;
    theContents_[number_].component_ = component;
    theContents_[number_].position_ = position;
    theContents_[number_++].fill_index(ind);
}

whichContents::whichContents(const whichContents &input)
{
    int i;
    number_ = input.number_;
    length_ = number_;
    theContents_ = new coStringObj[number_];
    for (i = 0; i < number_; ++i)
    {
        theContents_[i] = input.theContents_[i];
    }
}

whichContents &whichContents::operator=(const whichContents &rhs)
{
    if (this == &rhs)
        return *this;
    delete[] theContents_;
    int i;
    number_ = rhs.number_;
    length_ = number_;
    theContents_ = new coStringObj[number_];
    for (i = 0; i < number_; ++i)
    {
        theContents_[i] = rhs.theContents_[i];
    }
    return *this;
}

int
compar(const void *ope1, const void *ope2)
{
    int *elem1 = (int *)ope1;
    int *elem2 = (int *)ope2;
    if (*elem1 < *elem2)
    {
        return -1;
    }
    else if (*elem1 == *elem2)
    {
        return 0;
    }
    return 1;
}

void Map1D::setMap(int l, int *list)
{
    delete[] mapping_;
    mapping_ = 0;
    min_ = INT_MAX;
    max_ = INT_MIN;
    if (l <= 0)
        return;
    int i;
    for (i = 0; i < l; ++i)
    {
        if (list[i] < min_)
            min_ = list[i];
        if (list[i] > max_)
            max_ = list[i];
    }
    length_ = l;
    if (max_ - min_ < TRIVIAL_LIMIT)
    {
        method_ = TRIVIAL;
        mapping_ = new int[max_ - min_ + 1];
        for (i = 0; i < l; ++i)
        {
            mapping_[list[i] - min_] = i;
        }
    }
    else
    {
        int i;
        method_ = HASHING;
        labels_.clear();
        for (i = 0; i < l; ++i)
        {
            labels_.insert(list[i], i);
        }
    }
}

Map1D::Map1D(int l, int *list)
    : mapping_(0)
    , labels_(INT_MAX)
{
    setMap(l, list);
}

const int &Map1D::operator[](int i)
{
    const int *ret = 0;
    switch (method_)
    {
    case TRIVIAL:
        if (min_ > max_)
        {
            Covise::sendWarning("Unexpected error: Map1D[] may not be used");
            return min_;
        }
        if (i < min_)
        {
            i = min_;
            Covise::sendWarning("Unexpected error: Map1D[] with forbidden index");
        }
        if (i > max_)
        {
            i = max_;
            Covise::sendWarning("Unexpected error: Map1D[] with forbidden index");
        }
        ret = &mapping_[i - min_];
        break;
    case HASHING:
        ret = &labels_.find(i);
        break;
        /*
               ret=(int *)bsearch(&i,mapping_,length_,2*sizeof(int),compar);
               ret += 1;
               break;
         */
    }
    return *ret;
}
