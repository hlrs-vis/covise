/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    FileItem
// CLASS    PortItem
//
// Description:
//
// Initial version: 2001-
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2001 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef ITEMS_H
#define ITEMS_H

#include <covise/covise.h>
#include <map>

#ifdef __sgi
using namespace std;
#endif
#include <api/coChoiceParam.h>
#include <api/coFileBrowserParam.h>
#include <api/coOutputPort.h>

///////////////////////////////////////////////////////////////////
//
//   simple data class FileItem to store information about
//   filebrowser parameters
//
///////////////////////////////////////////////////////////////////

namespace covise
{

class READEREXPORT FileItem
{
public:
    FileItem();
    FileItem(const string &name,
             const string &desc,
             const string &value,
             const string &mask);

    string getName()
    {
        return name_;
    };
    string getDesc()
    {
        return desc_;
    };
    string getValue()
    {
        return value_;
    };
    string getMask()
    {
        return mask_;
    };

    // BE AWARE: allows user to change data but this behaviour is needed here
    coFileBrowserParam *getBrowserPtr()
    {
        return browserPtr_;
    };
    void setBrowserPtr(coFileBrowserParam *ptr)
    {
        browserPtr_ = ptr;
    };

    bool empty();

private:
    string name_;
    string desc_;
    string value_;
    string mask_;
    bool empty_;
    coFileBrowserParam *browserPtr_;
};

///////////////////////////////////////////////////////////////////
//
//   simple data class PortItem to store information about
//   filebrowser ports
//
///////////////////////////////////////////////////////////////////
class READEREXPORT PortItem
{

public:
    PortItem();
    PortItem(const string &name,
             const string &type,
             const string &desc,
             const bool &ifChoice);

    virtual ~PortItem()
    {
    }

    string getName()
    {
        return name_;
    };
    string getType()
    {
        return type_;
    };
    string getDesc()
    {
        return desc_;
    };
    coOutputPort *getPortPtr() const;

    void setPortPtr(coOutputPort *ptr);

    // BE AWARE: allows user to change data but this behaviour is needed here
    coChoiceParam *getChoicePtr()
    {
        return choicePtr_;
    };
    void setChoicePtr(coChoiceParam *ptr)
    {
        choicePtr_ = ptr;
    };
    //
    void fillChoice(const string &it);
    virtual void fillChoice(vector<string> &it, bool do_update = false);

    virtual void updateChoice(vector<string> &it)
    {
        fillChoice(it, true);
    };

    int getChoice();
    void cleanChoice();

    bool hasChoice()
    {
        return ifChoice_;
    };
    bool empty();

private:
    string name_;
    string type_;
    string desc_;
    bool empty_;
    coChoiceParam *choicePtr_;
    coOutputPort *portPtr_;
    bool ifChoice_;
    int maxChoiceVal_;
    vector<string> choiceLabels_;

    // synchronise choices with COVISE API
    void syncChoice(bool do_update);
};

typedef vector<PortItem *> PortItemList;

///////////////////////////////////////////////////////////////////
//
//   simple data class CompVecPortItem to store information about
//   vector ports that are composed by scalar values in the reader
//
///////////////////////////////////////////////////////////////////
class READEREXPORT CompVecPortItem : public PortItem
{

public:
    CompVecPortItem();
    CompVecPortItem(const string &name,
                    const string &type,
                    const string &desc);

    virtual ~CompVecPortItem()
    {
    }

    string getCompName(int comp)
    {
        return choices_[comp]->getName();
    };

    void setChoicePtrs(vector<coChoiceParam *> ptrs);
    void fillChoice(vector<string> &it, bool do_update = false);

    void updateChoice(vector<string> &it)
    {
        fillChoice(it, true);
    };

    vector<int> getChoice();

private:
    PortItemList choices_;

    void addItem(const string &chName, const string &type, const string &desc);
};
}
#endif
