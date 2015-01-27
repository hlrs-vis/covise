/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS    ReaderControl
//
// Description: container for all control information for readers
//
// Initial version: 22.04.2004
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// Changes:
//

#ifndef READERCONTROL_H
#define READERCONTROL_H

#include <covise/covise.h>
#include <map>

#ifdef _STANDARD_C_PLUS_PLUS
using std::map;
#endif

#include "Items.h"

namespace covise
{

#define READER_CONTROL ReaderControl::instance()

typedef map<int, FileItem *> FileList;
typedef map<int, PortItem *> PortList;
typedef map<int, CompVecPortItem *> CompVecPortList;

///////////////////////////////////////////////////////////////////
//
//   Singleton: ReaderControl - provide global information about
//                              current reader
//
///////////////////////////////////////////////////////////////////

//template class READEREXPORT map<string,int>;

class READEREXPORT ReaderControl
{

public:
    /// the one and only acess
    static ReaderControl *instance();

    void addFile(const int &tok,
                 const string &name,
                 const string &desc,
                 const string &value,
                 const string &mask = string("*"));

    void addOutputPort(const int &tok,
                       const string &name,
                       const string &type,
                       const string &desc,
                       const bool &ifChoice = true);

    void addCompVecOutPort(const int &tok,
                           const string &name,
                           const string &type,
                           const string &desc);

    // returns an int which is associated to the string item
    int fillPortChoice(const int &tok, const string &item);

    // fill all choices with one call
    void fillPortChoice(const int &tok, vector<string> &item);

    // same as fillPortChoice but keeps content of choice if possible
    void updatePortChoice(const int &tok, vector<string> &item);

    // returns the current value of the choice parameter of port
    // with token tok
    int getPortChoice(const int &tok);

    // returns the current value of the choice parameter of a composed vector port
    // with token tok
    vector<int> getCompVecChoices(const int &tok);

    void cleanPortChoice(const int &tok);

    // returns the COVISE object name associated with a port
    string getAssocObjName(const int &tok);

    int setAssocPortObj(const int &tok, coDistributedObject *obj);

    // store COVISE objects as .covise files in directory dir, create covise group file with name grpName.covgrp and fill it with labels
    bool storePortObj(string dir, string grpName, map<int, string> &labels);

    string getPortFileNm(const int &tok);

    virtual ~ReaderControl();

    // are the following methods useful
    string getFileVal(const int &tok);

    FileList getFileList()
    {
        return files_;
    };
    PortList getPortList()
    {
        return ports_;
    };
    FileItem *getFileItem(const int &tok);
    FileItem *getFileItem(const string &name);
    bool isCompVecPort(const int &tok);

private:
    ReaderControl();

    PortItem *getPortItem(const int &tok);

    void fillPortChoice(const int &tok, vector<string> &item, bool do_update);

    // !! helper: exit if not initialized
    void initCheck();

    static ReaderControl *instance_;

    FileList files_;
    PortList ports_;
    CompVecPortList cvec_ports_;
    map<string, int> fileNames_;
};
}
#endif
