/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// CLASS   ReadLasat
//
// Description:
//
//
// Initial version: 11.12.2002 (CS)
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// (C) 2002 by VirCinity IT Consulting
// All Rights Reserved.
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//
// $Id: ReadLasat.h,v 1.3 2002/12/17 13:36:05 ralf Exp $
//
#ifndef _READ_LASAT_H
#define _READ_LASAT_H

//#include <api/coModule.h>
using namespace covise;
//#include "HouseFile.h"
//#include "DmnaFiles.h"
//#include <api/coStepFile.h>
#ifdef __linux
#include <unistd.h>
#endif
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <fcntl.h>
class HouseFile;
class DmnaFiles;
class coFileBrowserParam;
class coOutputPort;
class ReadLasat : public coModule
{

private:
    // parameters

    // ports
    coOutputPort *p_house, *p_grid, *p_data;
    coFileBrowserParam *p_houseFile;
    coFileBrowserParam *p_dmnaFile;
    coFileBrowserParam *p_zFile;
    // private data
    HouseFile *houseFile;
    DmnaFiles *dmnaFile;
    int xSize;
    int ySize;
    int zSize;
    float *zValues;
    bool readZFile(const char *zFile, char *&errMsg);
    void init();

public:
    virtual void param(const char *paramName);
    virtual int compute();
    ReadLasat::ReadLasat(const char *houseFile, const char *dmnaFile, const char *zFile);
    ReadLasat();
    ~ReadLasat();
    float *getZValues()
    {
        return zValues;
    }
    int getZSize()
    {
        return zSize;
    }
};
#endif

//
// History:
//
// $Log: ReadLasat.h,v $
// Revision 1.3  2002/12/17 13:36:05  ralf
// adapted for windows
//
// Revision 1.2  2002/12/12 16:56:54  ralfm_te
// -
//
// Revision 1.1  2002/12/12 11:59:24  cs_te
// initial version
//
//
