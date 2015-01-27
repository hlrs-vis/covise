/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of base class coReader                   ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 11.04.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#include "coReader.h"
#include "ReaderControl.h"
#include <api/coOutputPort.h>
#include <api/coChoiceParam.h>

//#include <iostream>

//using namespace std;
using namespace covise;

//
// Constructor
//
coReader::coReader(int argc, char *argv[], const string &desc)
    : coModule(argc, argv, desc.c_str())
{

    // create input parameters
    FileList::iterator it;

    FileList fl(READER_CONTROL->getFileList());
    const FileList::iterator beg = fl.begin();
    const FileList::iterator end = fl.end();

    for (it = beg; it != end; ++it)
    {
        FileItem *fIt((*it).second);
        string name(fIt->getName());
        string desc(fIt->getDesc());
        coFileBrowserParam *cTmp = addFileBrowserParam(name.c_str(), desc.c_str());
        if (!cTmp)
        {
            cerr << "coReader::coReader() ERROR creating browser " << name << endl;
        }

        string def(fIt->getValue());
        string mask(fIt->getMask());
        int fail = cTmp->setValue(def.c_str(), mask.c_str());
        if (fail == 0)
        {
            cerr << "coReader::coReader() ERROR setting file-browser default " << endl;
        }
        fIt->setBrowserPtr(cTmp);
        fileBrowsers_.push_back(cTmp);
    }

    // create output ports
    PortList::iterator itP;

    PortList pl(READER_CONTROL->getPortList());
    const PortList::iterator begP = pl.begin();
    const PortList::iterator endP = pl.end();

    for (itP = begP; itP != endP; ++itP)
    {
        PortItem *portIt((*itP).second);
        if (portIt != NULL)
        {
            string name(portIt->getName());
            string type(portIt->getType());
            string desc(portIt->getDesc());
            coOutputPort *cTmp = addOutputPort(name.c_str(), type.c_str(), desc.c_str());
            portIt->setPortPtr(cTmp);
            outPorts_.push_back(cTmp);
            //check if it is composed vector output
            if (READER_CONTROL->isCompVecPort((*itP).first))
            {
                CompVecPortItem *vecItem = (CompVecPortItem *)portIt;
                int i = 0;
                vector<coChoiceParam *> chc;

                for (i = 0; i < 3; i++)
                {
                    string name(vecItem->getCompName(i));
                    coChoiceParam *chTmp = addChoiceParam(name.c_str(), "desc");
                    chc.push_back(chTmp);
                }
                vecItem->setChoicePtrs(chc);
            }
            else if (portIt->hasChoice())
            {
                // create assciated choice-param if needed
                coChoiceParam *chTmp;
                string chNam(string("data_for_") + name);
                chTmp = addChoiceParam(chNam.c_str(), "desc");
                portIt->setChoicePtr(chTmp);
            }
        }
        else
        {
            cerr << "coReader::coReader() FATAL ERROR in line __LINE__ : got NULL port item" << endl;
        }
    }
}

//
// Destructor
//
coReader::~coReader()
{
}

#ifdef _TESTING

int main(int argc, char *argv[])

{
    // define tokens
    enum
    {
        FBROWSER,
        XBRAUSE,
        AAA,
        BBB
    };
    enum
    {
        GEOPORT,
        DPORT1,
        DPORT2,
        DPORT3,
        DPORT4
    };

    // define outline of reader
    READER_CONTROL->addFile(FBROWSER, "test file browser", "test f\374r FileBrowser", "/test.1dat");
    READER_CONTROL->addFile(XBRAUSE, "test file2", "test 2 f\374r FileBrowser", "/test.dat2");
    READER_CONTROL->addFile(AAA, "test file3", "test 3 f\374r FileBrowser", "/test.dat3");
    READER_CONTROL->addFile(BBB, "test file4", "test 4 f\374r FileBrowser", "/test.dat4");

    READER_CONTROL->addOutputPort(GEOPORT, "geoOut", "UnstructuredGrid", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1, "sdata1", "Float", "data1");
    READER_CONTROL->addOutputPort(DPORT2, "sdata2", "Unstructured_S3D_data", "data2");

    READER_CONTROL->addOutputPort(DPORT3, "vdata1", "Vec3", "data1");
    READER_CONTROL->addOutputPort(DPORT4, "vdata2", "Unstructured_V3D_data", "data2");

    // create the module
    coReader *application = new coReader;

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
#endif
