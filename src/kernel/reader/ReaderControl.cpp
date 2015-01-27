/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2001 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of class ReaderControl                   ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 12.04.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ReaderControl.h"
#include "CoviseIO.h"
#include <api/coChoiceParam.h>
#include <api/coOutputPort.h>
#include <iostream>

using namespace covise;

// initialize instance_
ReaderControl *ReaderControl::instance_ = NULL;

ReaderControl::ReaderControl()
{
}

//
//  access method
//
ReaderControl *
ReaderControl::instance()
{
    if (!instance_)
    {
        instance_ = new ReaderControl;
    }
    return instance_;
}

void
ReaderControl::addFile(const int &tok,
                       const string &name,
                       const string &desc,
                       const string &value,
                       const string &mask)

{

    FileItem *fi = new FileItem(name, desc, value, mask);
    files_[tok] = fi;
    fileNames_[name] = tok;
}

void
ReaderControl::addOutputPort(const int &tok,
                             const string &name,
                             const string &type,
                             const string &desc,
                             const bool &noChoice)

{
    PortItem *pi = new PortItem(name, type, desc, noChoice);
    ports_[tok] = pi;
}

void
ReaderControl::addCompVecOutPort(const int &tok,
                                 const string &name,
                                 const string &type,
                                 const string &desc)

{
    CompVecPortItem *pi = new CompVecPortItem(name, type, desc);
    cvec_ports_[tok] = pi;
    ports_[tok] = pi;
}

string
ReaderControl::getFileVal(const int &tok)
{
    FileList::iterator it = files_.find(tok);

    if (it == files_.end())
    {
        string ret;
        return ret;
    }
    else
    {
        return (*it).second->getValue();
    }
}

FileItem *
ReaderControl::getFileItem(const int &tok)
{
    FileList::iterator it = files_.find(tok);

    if (it == files_.end())
    {
        FileItem *ret = NULL;
        return ret;
    }
    else
    {
        return (*it).second;
    }
}

FileItem *
ReaderControl::getFileItem(const string &name)
{

    map<string, int>::iterator it = fileNames_.find(name);

    int tok((*it).second);

    FileList::iterator itT = files_.find(tok);

    if (itT == files_.end())
    {
        FileItem *ret = NULL;
        return ret;
    }
    else
    {
        return (*itT).second;
    }
}

PortItem *
ReaderControl::getPortItem(const int &tok)
{
    PortList::iterator it = ports_.find(tok);

    if (it == ports_.end())
    {
        PortItem *ret = NULL;
        return ret;
    }
    else
    {
        return (*it).second;
    }
}

bool
ReaderControl::isCompVecPort(const int &tok)
{
    CompVecPortList::iterator it_v = cvec_ports_.find(tok);
    if (it_v == cvec_ports_.end())
    {
        return false;
    }
    else
    {
        return true;
    }
}

int
ReaderControl::fillPortChoice(const int &tok, const string &item)
{
    // introduce initCheck();
    // check if port has choice and exit if not

    int ret(0);

    PortItem *pI = getPortItem(tok);

    if (pI != NULL)
    {
        pI->fillChoice(item);
    }
    else
    {
        cerr << "ReaderControl::fillPortChoice(..) PortItem NULL" << endl;
    }
    return ret;
}

void
ReaderControl::fillPortChoice(const int &tok, vector<string> &item, bool do_update)
{
    // introduce initCheck();
    // check if port has choice and exit if not

    //cerr << "ReaderControl::fillPortChoice(..) " << tok << " item: " << item << endl;

    PortItem *pI = getPortItem(tok);

    if (pI != NULL)
    {
        if (isCompVecPort(tok))
        {
            ((CompVecPortItem *)pI)->fillChoice(item, do_update);
        }
        else
        {
            pI->fillChoice(item, do_update);
        }
    }
    else
    {
        cerr << "ReaderControl::fillPortChoice(..) PortItem NULL" << endl;
    }
}

void
ReaderControl::fillPortChoice(const int &tok, vector<string> &item)
{
    fillPortChoice(tok, item, false);
}

void
ReaderControl::updatePortChoice(const int &tok, vector<string> &item)
{
    fillPortChoice(tok, item, true);
}

int
ReaderControl::getPortChoice(const int &tok)
{
    PortItem *pI = getPortItem(tok);

    if (pI != NULL)
    {
        return pI->getChoice();
    }
    else
    {
        cerr << "ReaderControl::getPortChoice(..) PortItem NULL" << endl;
        return -1;
    }
}

vector<int>
ReaderControl::getCompVecChoices(const int &tok)
{
    if (isCompVecPort(tok))
    {
        CompVecPortItem *pI = (CompVecPortItem *)getPortItem(tok);
        if (pI != NULL)
        {
            return pI->getChoice();
        }
        else
        {
            cerr << "ReaderControl::getCompVecPortChoice(..) PortItem NULL" << endl;
        }
    }
    vector<int> vec;
    vec.push_back(1);
    vec.push_back(1);
    vec.push_back(1);
    return vec;
}

void
ReaderControl::cleanPortChoice(const int &tok)
{
    PortItem *pI = getPortItem(tok);

    if (pI != NULL)
    {
        pI->cleanChoice();
    }
    else
    {
        cerr << "ReaderControl::cleanPortChoice(..) PortItem NULL" << endl;
    }
}

string
ReaderControl::getAssocObjName(const int &tok)
{
    PortItem *pI = getPortItem(tok);

    string ret;
    if (pI != NULL)
    {
        coOutputPort *ptr = pI->getPortPtr();
        ret = string(ptr->getObjName());
    }
    else
    {
        cerr << "ReaderControl::getAssocObjName(..) PortItem NULL" << endl;
    }
    return ret;
}

int
ReaderControl::setAssocPortObj(const int &tok, coDistributedObject *obj)
{
    PortItem *pI = getPortItem(tok);

    int ret;
    if (pI != NULL)
    {
        coOutputPort *ptr = pI->getPortPtr();
        ptr->setCurrentObject(obj);
        ret = 1;
    }
    else
    {
        cerr << "ReaderControl::setAssocPortObj(..) PortItem NULL" << endl;
        ret = -1;
    }
    return ret;
}

string
ReaderControl::getPortFileNm(const int &)
{

    string ret;

    return ret;
}

bool
ReaderControl::storePortObj(string dir, string grpName, map<int, string> &labels)
{
    CoviseIO io;
    coOutputPort *ptr;
    PortList::iterator it;
    string grpFileName = dir + string("/") + grpName;

    // read in existing covgrp file
    string oldfiles, line;
    ifstream oldGrp(grpFileName.c_str());
    while (getline(oldGrp, line))
    {
        oldfiles += line;
    }

    ofstream grpFile(grpFileName.c_str(), ios::app);
    for (it = ports_.begin(); it != ports_.end(); it++)
    {
        ptr = (*it).second->getPortPtr();
        string file = dir + string("/") + labels[(*it).first] + ".covise";
        if (ptr->getCurrentObject())
        {
            // check if already stored
            if (oldfiles.find(file) == string::npos)
            {
                if (io.WriteFile((char *)file.c_str(), ptr->getCurrentObject()))
                {
                    grpFile << labels[(*it).first] << ":" << file << endl;
                }
                else
                    return false;
            }
        }
    }
    return true;
}

//
// Destructor
//
ReaderControl::~ReaderControl()
{
}
