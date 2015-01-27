/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READ2DM_H
#define READ2DM_H

#include <api/coModule.h>
#include <util/coFileUtil.h>
using namespace covise;

class Read2DM : public coModule
{
public:
    Read2DM(int argc, char *argv[]);
    virtual ~Read2DM();

    virtual int compute(const char *port);
    virtual void param(const char *paramname, bool inMapLoading);

private:
    bool read2DMFile();
    bool readSRHFile();

    coFileBrowserParam *twodmFilenameParameter;
    coFileBrowserParam *srhFilenameParameter;
    coChoiceParam **dataParameter;

    coOutputPort *polygonOutputPort;
    coOutputPort **dataOutputPort;

    void updateChoice(int number);

    int dataValuesExpected;

    std::vector<std::string> choiceValues;
};

#endif
