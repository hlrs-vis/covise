/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class  FileItem                       ++
// ++             Implementation of class  PortItem                       ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 15.04.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "Items.h"
#include <api/coChoiceParam.h>
#include <api/coModule.h>

using namespace covise;

///////////////////////////////////////////////////////////////////
//   simple data class FileItem
///////////////////////////////////////////////////////////////////
FileItem::FileItem()
    : empty_(true)
{
}

FileItem::FileItem(const string &name,
                   const string &desc,
                   const string &value,
                   const string &mask)
    : name_(name)
    , desc_(desc)
    , value_(value)
    , mask_(mask)
    , empty_(false)
    , browserPtr_(NULL)
{
}

bool
FileItem::empty()
{
    return empty_;
}

///////////////////////////////////////////////////////////////////
//   simple data class PortItem
///////////////////////////////////////////////////////////////////
PortItem::PortItem()
    : empty_(true)
{
}

PortItem::PortItem(const string &name,
                   const string &type,
                   const string &desc,
                   const bool &ifChoice)
    : name_(name)
    , type_(type)
    , desc_(desc)
    , empty_(false)
    , portPtr_(NULL)
    , ifChoice_(ifChoice)
    , maxChoiceVal_(0)
{
}

bool
PortItem::empty()
{
    return empty_;
}

void
PortItem::cleanChoice()
{
    choiceLabels_.erase(choiceLabels_.begin(), choiceLabels_.end());
}

void
PortItem::fillChoice(const string &it)
{

    if (!it.empty())
    {
        choiceLabels_.push_back(it);
    }
    else
    {
        cerr << "PortItem::fillChoice(..) it empty ?????" << endl;
        return;
    }

    syncChoice(false);
}

void
PortItem::fillChoice(vector<string> &it, bool do_update)
{
    //cleanChoice();
    choiceLabels_ = it;
    syncChoice(do_update);
}

void
PortItem::syncChoice(bool do_update)
{
    if (choicePtr_ != NULL)
    {
        int num = (int)choiceLabels_.size();
        int i;
        char **lables = new char *[num];
        for (i = 0; i < num; ++i)
        {
            lables[i] = const_cast<char *>(choiceLabels_[i].c_str());
        }

        if (do_update)
        {
            choicePtr_->updateValue(num, lables, 0);
        }
        else
        {
            choicePtr_->setValue(num, lables, 0);
        }

        // don't dare to delete the array elements
        delete[] lables;
    }
}

int
PortItem::getChoice()
{
    if (choicePtr_ != NULL)
    {
        int ret(choicePtr_->getValue());
        //	cerr << "PortItem::getChoice() found: " << ret << endl;
        return ret;
    }
    return -1;
}

void
PortItem::setPortPtr(coOutputPort *ptr)
{
    portPtr_ = ptr;
}

coOutputPort *
PortItem::getPortPtr() const
{
    return portPtr_;
}

CompVecPortItem::CompVecPortItem()
    : PortItem()
{
}

CompVecPortItem::CompVecPortItem(const string &name,
                                 const string &type,
                                 const string &desc)
    : PortItem(name, type, desc, false)
{
    string chName = name + "_x";
    addItem(chName, type, desc);

    chName = name + "_y";
    addItem(chName, type, desc);

    chName = name + "_z";
    addItem(chName, type, desc);
}

void
CompVecPortItem::addItem(const string &chName, const string &type, const string &desc)
{
    PortItem *comp = new PortItem(chName, type, desc, true);
    choices_.push_back(comp);
}

void
CompVecPortItem::setChoicePtrs(vector<coChoiceParam *> ptrs)
{
    int i = 0;
    for (i = 0; i < 3; i++)
    {
        choices_[i]->setChoicePtr(ptrs[i]);
    }
}

void
CompVecPortItem::fillChoice(vector<string> &it, bool do_update)
{
    int i = 0;
    for (i = 0; i < 3; i++)
    {
        choices_[i]->fillChoice(it, do_update);
    }
}

vector<int>
CompVecPortItem::getChoice()
{
    vector<int> results;

    int i = 0;
    for (i = 0; i < 3; i++)
    {
        results.push_back(choices_[i]->getChoice());
    }

    return results;
}
