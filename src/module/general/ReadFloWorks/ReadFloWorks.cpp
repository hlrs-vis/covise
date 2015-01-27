/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of module ReadFloWorks                   ++
// ++                                                                     ++
// ++ Author:  Sven Kufer (sk@vircinity.com)                              ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 09.09.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <util/coviseCompat.h>

#include "ReadFloWorks.h"

// define tokens for ports
enum
{
    CASE_BROWSER,
    GEOPORT,
    DPORT1,
    DPORT2,
    DPORT3,
    DPORT4,
    VPORT
};
const int DataPorts[] = { DPORT1, DPORT2, DPORT3, DPORT4, VPORT };
const int NumDataPorts = 5;

const int ScalarDataPorts[] = { DPORT1, DPORT2, DPORT3, DPORT4 };
const int NumScalarDataPorts = 4;

const int VectorDataPorts[] = { VPORT };
const int NumVectorDataPorts = 1;
//
// Constructor
//
ReadFloWorks::ReadFloWorks(int argc, char *argv[])
    : coReader(argc, argv, string("Reader for FloWorks files"))
{
}

//
// Destructor
//
ReadFloWorks::~ReadFloWorks()
{
    delete flow_;
}

void
ReadFloWorks::param(const char *paramName)
{
    //    cerr << "ReadFloWorks::param(..) called : " << paramName << endl;
    int mapLoading = in_map_loading();

    FileItem *fii = READER_CONTROL->getFileItem(CASE_BROWSER);

    string caseBrowserName;
    if (fii)
    {
        caseBrowserName = fii->getName();
    }
    //    cerr << "ReadFloWorks::param(..)  case browser name <" << caseBrowserName << ">" << endl;

    /////////////////  CALLED BY FILE BROWSER  //////////////////

    if (caseBrowserName == string(paramName))
    {

        FileItem *fi(READER_CONTROL->getFileItem(string(paramName)));
        if (fi)
        {

            coFileBrowserParam *bP = fi->getBrowserPtr();

            if (bP)
            {
                string caseNm(bP->getValue());
                if (caseNm.empty())
                {
                    cerr << "ReadFloWorks::param(..) no case file found " << endl;
                }
                else
                {
                    flow_ = new FloWorks(caseNm.c_str());
                    dl_ = flow_->getDataIts();

                    if (!mapLoading)
                    {

                        DataList::iterator it;

                        vector<string> allchoices;

                        // fill in NONE to READ no data
                        string noneStr("----");

                        allchoices.push_back(noneStr);

                        for (it = dl_.begin(); it != dl_.end(); ++it)
                        {
                            // fill choice parameter of out-port for scalar data
                            switch ((*it).getType())
                            {
                            case DataItem::scalar:
                            {
                                allchoices.push_back((*it).getDesc());
                                break;
                            }
                            case DataItem::vector:
                                break;
                            case DataItem::tensor:
                                // fill in ports for tensor data
                                break;
                            }
                        }
                        int i;
                        for (i = 0; i < NumDataPorts; i++)
                        {
                            READER_CONTROL->updatePortChoice(DataPorts[i], allchoices);
                        }
                    }
                }
            }

            else
            {
                cerr << "ReadFloWorks::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

// read geometry of current time step
int
ReadFloWorks::readGeometry(const int &portTok)
{
    int numTs = flow_->getNumSteps();
    int cnt, n;
    float real_time;
    int *cellList, *typeList, *connList;
    int numCell, numConn, numVert;

    float *x, *y, *z;
    float *dest;

    if (numTs > 1)
    {
        // create set
        string objNameBase = READER_CONTROL->getAssocObjName(portTok);

        coDoUnstructuredGrid **objects = new coDoUnstructuredGrid *[numTs + 1];

        for (cnt = 0; cnt < numTs; cnt++)
        {
            flow_->activateTimeStep(cnt, real_time);
            // create DO
            string actObjNm(objNameBase);
            actObjNm += "_";
            actObjNm += cnt;

            flow_->getGridSize(numCell, numConn, numVert);

            objects[cnt] = new coDoUnstructuredGrid(actObjNm.c_str(),
                                                    numCell,
                                                    numConn,
                                                    numVert,
                                                    1);

            objects[cnt]->getAddresses(&cellList, &connList, &x, &y, &z);
            objects[cnt]->getTypeList(&typeList);

            flow_->getGridConn(cellList, typeList, connList);
            flow_->getGridCoord(x, y, z);
        }
        objects[cnt] = NULL;

        coDoSet *outSet = new coDoSet(objNameBase.c_str(), (coDistributedObject **)objects);

        // delete !!!!
        int i;
        for (i = 0; i < cnt; ++i)
            delete objects[i];
        delete[] objects;

        // set attribute
        char ch[64];
        sprintf(ch, "1 %d", cnt);
        string attr(ch);
        outSet->addAttribute("TIMESTEP", attr.c_str());

        READER_CONTROL->setAssocPortObj(portTok, outSet);
    }
    else
    {
        string objName(READER_CONTROL->getAssocObjName(portTok));
        flow_->activateTimeStep(0, real_time);
        flow_->getGridSize(numCell, numConn, numVert);

        coDoUnstructuredGrid *geoOut = new coDoUnstructuredGrid(objName.c_str(),
                                                                numCell,
                                                                numConn,
                                                                numVert,
                                                                1);

        geoOut->getAddresses(&cellList, &connList, &x, &y, &z);
        geoOut->getTypeList(&typeList);

        flow_->getGridCoord(x, y, z);
        flow_->getGridConn(cellList, typeList, connList);

        READER_CONTROL->setAssocPortObj(portTok, geoOut);
    }

    return Success;
}

// read Scalar data
int
ReadFloWorks::readScalarData(const int &portTok)
{
    int totNumTs = flow_->getNumSteps();
    int chc = READER_CONTROL->getPortChoice(portTok) - 2;
    cerr << chc << " " << endl;

    int cnt;
    float *data, real_time;

    if (chc < 0)
    {
        return Success;
    }

    string species = dl_[chc].getDesc();
    cerr << species << endl;

    // we want to have a set only if we have transient data
    if (totNumTs > 1)
    {
        // transient data
        // create set
        string objNameBase = READER_CONTROL->getAssocObjName(portTok);

        coDoFloat **objects = new coDoFloat *[totNumTs + 1];

        for (cnt = 0; cnt < totNumTs; cnt++)
        {
            flow_->activateTimeStep(cnt, real_time);

            char ch[64];
            sprintf(ch, "%d", cnt);
            string num(ch);
            string actObjNm(objNameBase + string("_") + num);

            objects[cnt] = new coDoFloat(actObjNm.c_str(), flow_->getNumCells());
            objects[cnt]->getAddress(&data);
            flow_->getData(chc, 0, data);
        }
        objects[cnt] = NULL;

        coDoSet *outSet = new coDoSet(objNameBase.c_str(), (coDistributedObject **)objects);

        // delete !!!!
        int i;
        for (i = 0; i < cnt; ++i)
            delete objects[i];
        delete[] objects;

        // set attribute
        char ch[64];
        sprintf(ch, "1 %d", cnt);
        string attr(ch);
        outSet->addAttribute("TIMESTEP", attr.c_str());
        outSet->addAttribute("SPECIES", species.c_str());

        READER_CONTROL->setAssocPortObj(portTok, outSet);
    }
    else
    {
        // stationary data - no set -
        string objName = READER_CONTROL->getAssocObjName(portTok);

        coDoFloat *sdata1 = new coDoFloat(objName.c_str(), flow_->getNumCells());
        sdata1->getAddress(&data);
        sdata1->addAttribute("SPECIES", species.c_str());

        flow_->getData(chc, 0, data);

        READER_CONTROL->setAssocPortObj(portTok, sdata1);
    }

    return Success;
}

// read Scalar data
int
ReadFloWorks::readVectorData(const int &portTok)
{
    int totNumTs = flow_->getNumSteps();
    // - 2;
    vector<int> chc = READER_CONTROL->getCompVecChoices(portTok);
    int cnt;
    for (cnt = 0; cnt < 3; cnt++)
    {
        chc[cnt] -= 2;
    }

    if (chc[0] < 0 || chc[1] < 0 || chc[2] < 0)
    {
        return Success;
    }

    string species = dl_[chc[0]].getDesc();
    float *xdata, *ydata, *zdata, real_time;

    // we want to have a set only if we have transient data
    if (totNumTs > 1)
    {
        // transient data
        // create set
        string objNameBase = READER_CONTROL->getAssocObjName(portTok);

        coDoVec3 **objects = new coDoVec3 *[totNumTs + 1];

        for (cnt = 0; cnt < totNumTs; cnt++)
        {
            flow_->activateTimeStep(cnt, real_time);

            char ch[64];
            sprintf(ch, "%d", cnt);
            string num(ch);
            string actObjNm(objNameBase + string("_") + num);

            objects[cnt] = new coDoVec3(actObjNm.c_str(), flow_->getNumCells());
            objects[cnt]->getAddresses(&xdata, &ydata, &zdata);
            flow_->getData(chc[0], 0, xdata);
            flow_->getData(chc[1], 0, ydata);
            flow_->getData(chc[2], 0, zdata);
        }
        objects[cnt] = NULL;

        coDoSet *outSet = new coDoSet(objNameBase.c_str(), (coDistributedObject **)objects);

        // delete !!!!
        int i;
        for (i = 0; i < cnt; ++i)
            delete objects[i];
        delete[] objects;

        // set attribute
        char ch[64];
        sprintf(ch, "1 %d", cnt);
        string attr(ch);
        outSet->addAttribute("TIMESTEP", attr.c_str());
        outSet->addAttribute("SPECIES", species.c_str());

        READER_CONTROL->setAssocPortObj(portTok, outSet);
    }
    else
    {
        // stationary data - no set -
        string objName = READER_CONTROL->getAssocObjName(portTok);

        coDoVec3 *vdata1 = new coDoVec3(objName.c_str(), flow_->getNumCells());
        vdata1->getAddresses(&xdata, &ydata, &zdata);
        flow_->getData(chc[0], 0, xdata);
        flow_->getData(chc[1], 0, ydata);
        flow_->getData(chc[2], 0, zdata);
        vdata1->addAttribute("SPECIES", species.c_str());
        READER_CONTROL->setAssocPortObj(portTok, vdata1);
    }

    return Success;
}

int
ReadFloWorks::compute(const char *)
{

    if (flow_->empty())
    {
        cerr << "ReadFloWorks::compute(..) case file not found  " << endl;
        Covise::sendError("case file not found");
        return 0;
    }

    // check this !!
    //    if ( readGeo_ ) {
    int state;
    state = readGeometry(GEOPORT);

    if (state == Failure)
    {
        cerr << "ReadFloWorks::compute(..) geometry file not found  " << endl;
        return 0;
    }
    //}

    // this flag is set to true only if the case file has changed
    //readGeo_ = false;

    // now read data

    int i;
    for (i = 0; i < NumScalarDataPorts; i++)
    {
        readScalarData(ScalarDataPorts[i]);
    }

    for (i = 0; i < NumVectorDataPorts; i++)
    {
        readVectorData(VectorDataPorts[i]);
    }

    return CONTINUE_PIPELINE;
}

int main(int argc, char *argv[])
{
    // define outline of reader
    READER_CONTROL->addFile(CASE_BROWSER, "result_path", "result path", "/", "*");

    READER_CONTROL->addOutputPort(GEOPORT, "geoOut", "UnstructuredGrid", "Geometry", false);

    int i;
    char portname[128];
    for (i = 0; i < NumDataPorts; i++)
    {
        sprintf(portname, "sdata%d", i);
        READER_CONTROL->addOutputPort(DataPorts[i], portname, "Float", portname + 1);
    }

    READER_CONTROL->addCompVecOutPort(VPORT, "vectorOut", "Vec3", "vector data");

    // create the module
    coReader *application = new ReadFloWorks(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
