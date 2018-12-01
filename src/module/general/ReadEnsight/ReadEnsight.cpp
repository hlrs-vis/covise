/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

//-*-Mode: C++;-*-
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// ++                                                  (C)2002 VirCinity  ++
// ++ Description:                                                        ++
// ++             Implementation of module ReadEnsight                    ++
// ++                                                                     ++
// ++ Author:  Ralf Mikulla (rm@vircinity.com)                            ++
// ++                                                                     ++
// ++               VirCinity GmbH                                        ++
// ++               Nobelstrasse 15                                       ++
// ++               70569 Stuttgart                                       ++
// ++                                                                     ++
// ++ Date: 15.04.2002                                                    ++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include <util/coviseCompat.h>
#include <do/coDoData.h>
#include <do/coDoSet.h>
#include <do/coDoUnstructuredGrid.h>
#include <util/coRestraint.h>
#include <util/covise_regexp.h>
#include <alg/coCellToVert.h>
#include <reader/ReaderControl.h>

#include "ReadEnsight.h"
#include "CaseParser.hpp"
#include "GeoFileAsc.h"
#include "EnFile.h"
#include "EnPart.h"
#include "DataFileAsc.h"
#include "GeoFileBin.h"
#include "DataFileBin.h"
#include "DataFileGold.h"
#include "DataFileGoldBin.h"
#include "EnFile.h"
#include "Reducer.h"
#include "AutoColors.h"
#include <alg/coCellToVert.h>

// #define DEBUG

// define tokens for ports
enum
{
    CASE_BROWSER,
    GEOPORT3D,
    DPORT1_3D,
    DPORT2_3D,
    DPORT3_3D,
    DPORT4_3D,
    DPORT5_3D,
    GEOPORT2D,
    DPORT1_2D,
    DPORT2_2D,
    DPORT3_2D,
    DPORT4_2D,
    DPORT5_2D,
    GEOPORT1D,
    DPORT1_1D,
    DPORT2_1D,
    DPORT3_1D,
    DPORT4_1D,
    DPORT5_1D,
};


//
// Constructor
//
ReadEnsight::ReadEnsight(int argc, char *argv[])
    : coReader(argc, argv, string("Reader for ENSIGHT 6 - type formats - Version 1.0"))
    , readGeo_(true)
    , binType_(EnFile::NOBIN)
    , dataByteSwap_(true)
    , geoObj_(NULL)
{
    geoObjs_[0] = NULL;
    geoObjs_[1] = NULL;
    geoObjs_[2] = NULL;

    // parameter to decide if cell based data should be automatically transformed to
    // vertex based data. The default is true (=transform)

    //     transformToVert_ = addBooleanParam("transfer cell data to vertex data", "transfrer cell based data to vertex based data");

    //     transformToVert_->setValue(1);

    dataByteSwapParam_ = addBooleanParam("data_byte_swap", "set if data is byte swapped");
    dataByteSwapParam_->setValue(1);

    partStringParam_ = addStringParam("choose_parts", "string to extract parts");
    partStringParam_->setValue("all");

    // parameter to use usg->comprssConnectivity
    repairConnectivity_ = addBooleanParam("repair_connectivity", "remove degenerated cells from the geometry");
    repairConnectivity_->setValue(0);

    autoColoring_ = addBooleanParam("enable_autocoloring", "add automatic coloring to 2D parts");
    autoColoring_->setValue(1);

    storeGrpParam_ = addBooleanParam("store_covgrp", "store result in covise group file");
    storeGrpParam_->setValue(0);

    includePolyederParam_ = addBooleanParam("include_polyhedra", "include 3D polyhedral cells in grid output");
    // includePolyederParam_->setValue(0);
    includePolyederParam_->setValue(1);
}

//
// Destructor
//
ReadEnsight::~ReadEnsight()
{
}

void
ReadEnsight::param(const char *paramName, bool inMapLoading)
{
    //cerr << "ReadEnsight::param(..) called : " << paramName << endl;

    if (string(paramName) == "data_byte_swap")
    {
        dataByteSwap_ = dataByteSwapParam_->getValue() == 1;
        return;
    }

    FileItem *fii = READER_CONTROL->getFileItem(CASE_BROWSER);

    string caseBrowserName;
    if (fii)
    {
        caseBrowserName = fii->getName();
    }
    // cerr << "ReadEnsight::param(..)  case browser name <" << caseBrowserName << ">" << endl;

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
                    cerr << "ReadEnsight::param(..) no case file found " << endl;
                }
                else
                {
                    // cerr << "ReadEnsight::param(..) filename " << caseNm << endl;
                    // we parse the case file
                    CaseParser *parser;
                    parser = new CaseParser(caseNm);
                    // uncomment this line to see more debug output
                    //parser->yydebug=1;
                    if (!parser->isOpen())
                    {
                        Covise::sendError("case file not found");
                        return;
                    }

                    parser->yyparse();

                    case_ = parser->getCaseObj();

                    case_.setFullFilename(caseNm);

                    delete parser;

                    // feed choice parameters

                    if (case_.empty())
                    {
                        cerr << "ReadEnsight::param(..) case obj empty !!!" << endl;
                    }
                    else
                    {
                        // we should read the geometry
                        readGeo_ = true;
                        // this is dirty TBD do it better
                        ////// print Number of Timesteps
                        const TimeSets &ts = case_.getAllTimeSets();
                        TimeSets::const_iterator tsIter;
                        int numTs = 0;
                        for (tsIter = ts.begin(); tsIter != ts.end(); ++tsIter)
                        {
                            numTs += (*tsIter)->getNumTs();
                        }
                        coModule::sendInfo("Found Dataset with %d timesteps", numTs);

                        // Get data fields
                        DataList dl = case_.getDataIts();
                        DataList::iterator it;

                        // lists for Choice Labels
                        vector<string> vectChoices;
                        vector<string> scalChoices;

                        // fill in NONE to READ no data
                        string noneStr("NONE");
                        scalChoices.push_back(noneStr);
                        vectChoices.push_back(noneStr);

                        // fill in all species for the appropriate Ports/Choices
                        for (it = dl.begin(); it != dl.end(); ++it)
                        {
                            // fill choice parameter of out-port for scalar data
                            // that is token  DPORT1 DPORT2
                            switch ((*it).getType())
                            {
                            case DataItem::scalar:
                                scalChoices.push_back((*it).getDesc());
                                break;
                            case DataItem::vector:
                                vectChoices.push_back((*it).getDesc());
                                break;
                            case DataItem::tensor:
                                // fill in ports for tensor data
                                break;
                            }
                        }
                        if (inMapLoading)
                            return;
                        READER_CONTROL->updatePortChoice(DPORT1_3D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_3D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_3D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_3D, vectChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_3D, vectChoices);
                        READER_CONTROL->updatePortChoice(DPORT1_2D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_2D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_2D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_2D, vectChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_2D, vectChoices);
                        READER_CONTROL->updatePortChoice(DPORT1_1D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT2_1D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT3_1D, scalChoices);
                        READER_CONTROL->updatePortChoice(DPORT4_1D, vectChoices);
                        READER_CONTROL->updatePortChoice(DPORT5_1D, vectChoices);
                    }
                }
                if (!inMapLoading)
                    createMasterPL();
                return;
            }

            else
            {
                cerr << "ReadEnsight::param(..) BrowserPointer NULL " << endl;
            }
        }
    }
}

vector<string>
ReadEnsight::mkFileNames(const string &baseName, int &realNumTs)
{
    // we may have transient data
    TimeSets ts(case_.getAllTimeSets());
    TimeSets::iterator tsIt, beg(ts.begin());
    int rNumTs = 0;
    // this is the timeset index of the geometry file
    int geoTimesetIdx(case_.getGeoTsIdx());
    int numTs(0);
    // name of geometry file
    vector<string> allGeoFiles;
    for (tsIt = beg; tsIt != ts.end(); tsIt++)
    {
        int idx = (*tsIt)->getIdx();
        if ((idx == geoTimesetIdx) || (geoTimesetIdx_ == -1) || (geoTimesetIdx == -1))
        {
            numTs = (*tsIt)->getNumTs();
            rNumTs += numTs;
            vector<string> ff = (*tsIt)->getFileNames(baseName);
            vector<string>::iterator ii;

            int fcnt(0);
            for (ii = ff.begin(); ii != ff.end(); ii++)
            {
                if (fcnt < numTs)
                    allGeoFiles.push_back(*ii);
                fcnt++;
            }
        }
    }
    realNumTs = rNumTs;
    return allGeoFiles;
}

// Create a list of all Ensight parts ( of the first timestep in time dependent data )
// This is the master part
// Write a table of all parts in the master part list  to the info channel
void
ReadEnsight::createMasterPL()
{
    // cerr << "ReadEnsight::createMasterPL() called" << endl;
    // we read only the first geometry file and assume that for moving geometries
    // the split-up in parts does not change
    string geoFileName = case_.getGeoFileNm();
    string fName;
    int rNumTs;
    vector<string> allGeoFiles(mkFileNames(geoFileName, rNumTs));
    //int numTs( allGeoFiles.size() );
    if (allGeoFiles.empty())
        fName = geoFileName;
    else
        fName = allGeoFiles[0];

    // create file object
    EnFile *enf = EnFile::createGeometryFile(this, case_, fName);
    masterPL_.clear();
    if (enf != NULL)
    {
        enf->setDataByteSwap(dataByteSwap_);
        enf->setPartList(&masterPL_);
        // this creates the table and extracts the part infos
        enf->parseForParts();
        if (enf->fileMayBeCorrupt_)
        {
            coModule::sendInfo("GeoFile %s may be corrupt - try data byteSwap!!", fName.c_str());
            bool bs = !dataByteSwap_;
            masterPL_.clear();
            delete enf;
            enf = EnFile::createGeometryFile(this, case_, fName);
            enf->setDataByteSwap(bs);
            enf->setPartList(&masterPL_);
            enf->parseForParts();
            if (enf->fileMayBeCorrupt_)
            {
                coModule::sendInfo("GeoFile %s seems to be corrupt!! Check case file or geo file", fName.c_str());
            }
            else
            {
                if (dataByteSwapParam_->getValue() == 1)
                    dataByteSwapParam_->setValue(0);
                else
                    dataByteSwapParam_->setValue(1);
            }
        }
    }
    delete enf;
}

// read geometry file or geometry files
// to a given port with ReaderControl token portTok
int
ReadEnsight::readGeometry(const int &portTok2d, const int &portTok3d)
{
    // set filenames
    string geoFileName = case_.getGeoFileNm();
    if (geoFileName.length() < 5)
    {
        return Failure;
    }
    int realNumTs = 0;
    vector<string> allGeoFiles(mkFileNames(geoFileName, realNumTs));
    size_t numTs(allGeoFiles.size());
    //const int *idxMap=NULL;

    // set object names and create object arrays for timesteps
    string objNameBase2d = READER_CONTROL->getAssocObjName(portTok2d);
    string objNameBase3d = READER_CONTROL->getAssocObjName(portTok3d);
    coDistributedObject **objects2d = new coDistributedObject *[numTs + 1];
    coDistributedObject **objects3d = new coDistributedObject *[numTs + 1];

    vector<string>::iterator ii;

    if (numTs == 0)
        allGeoFiles.push_back(case_.getGeoFileNm());

    // read all files
    int cnt(0);
    for (ii = allGeoFiles.begin(); ii != allGeoFiles.end(); ii++)
    {
        char ch[64];
        sprintf(ch, "%d", cnt);
        string num(ch);
        string actObjNm2d(objNameBase2d);
        string actObjNm3d(objNameBase3d);
        if (realNumTs > 1)
        {
            actObjNm2d = objNameBase2d + "_" + num;
            actObjNm3d = objNameBase3d + "_" + num;
        }
        // create representation of geometry file..
        EnFile *enf = EnFile::createGeometryFile(this, case_, *ii);
        if (enf == NULL)
        {
            cerr << "ReadEnsight::readGeometry(..) enf NULL " << endl;
            return Failure;
        }

        binType_ = enf->binType();
        enf->setDataByteSwap(dataByteSwapParam_->getValue() == 1);
        enf->setIncludePolyeder(includePolyederParam_->getValue() == 1);

        PartList *pl = new PartList;
        enf->setPartList(pl);
        enf->setMasterPL(masterPL_);
        enf->setActiveAlloc((cnt != 0));

        if (!enf->isOpen())
        {
            cerr << "ReadEnsight::readGeometry(..) enf NOT open " << endl;
            return Failure;
        }

        coModule::sendInfo(" start reading geometry ( %s ) - please be patient..", (*ii).c_str());
        enf->read(this, EnFile::GEOMETRY, objects2d,objects3d, actObjNm2d,actObjNm3d,cnt);

#ifdef DEBUG
        cerr << " elem: " << pl->at(0).subParts_numElem.size() << " conn: " << pl->at(0).subParts_numConn.size() << endl;
#endif


        delete enf;

    }

    // we have no timesteps - feed objectsXY[0] to outports
    if (realNumTs <= 1)
    {
        if (objects2d[0] != NULL)
            READER_CONTROL->setAssocPortObj(portTok2d, objects2d[0]);
        if (objects3d[0] != NULL)
            READER_CONTROL->setAssocPortObj(portTok3d, objects3d[0]);
        geoObjs_[0] = objects2d[0];
        geoObjs_[1] = objects3d[0];
    }
    else // handle timesteps
    {
        TimeSets ts(case_.getAllTimeSets());
        // create warning if more than one timeset is present
        // actually covise can handle only one timeset
        if (ts.size() > 1)
        {
            coModule::sendInfo("number of time-sets greater than one - COVISE can handle only one time-set");
        }

        vector<float> rTimes(ts[0]->getRealTimes());
        if (rTimes.empty())
            rTimes.resize(allGeoFiles.size(), 0.0);

        char ch[64];
        bool staGeoMultTs = false;
        // recognize static geo - timedep. data
        // in this case we have to multiply the geometry
        TimeSets::const_iterator tsIter;
        int trueNumTs = 0;
        for (tsIter = ts.begin(); tsIter != ts.end(); ++tsIter)
            trueNumTs += (*tsIter)->getNumTs();
        if ((numTs == 1) && (trueNumTs > 0))
        {
            cerr << "   static geo - timedep. data - " << trueNumTs << endl;
            staGeoMultTs = true;
            coDistributedObject **tmp = new coDistributedObject *[1 + trueNumTs];
            int jj = 0;
            for (jj = 0; jj < trueNumTs; ++jj)
            {
                incrRefCnt(objects2d[0]);
                tmp[jj] = objects2d[0];
            }
            tmp[jj] = NULL;
            objects2d = tmp;
            tmp = new coDistributedObject *[1 + trueNumTs];
            for (jj = 0; jj < trueNumTs; ++jj)
            {
                // increment refcounter recursive
                tmp[jj] = objects3d[0];
                tmp[jj]->incRefCount();
            }
            tmp[jj] = NULL;
            objects3d = tmp;
            sprintf(ch, "1 %d", trueNumTs);
        }
        else
        {
            objects2d[cnt] = NULL;
            objects3d[cnt] = NULL;

            // set attribute - realtime
            for (int i = 0; (i < cnt) && (i < rTimes.size()); ++i)
            {
                sprintf(ch, "%f", rTimes[i]);
                objects2d[i]->addAttribute("REALTIME", ch);
                objects3d[i]->addAttribute("REALTIME", ch);
            }
            sprintf(ch, "1 %d", cnt);
        }

        coDoSet *outSet2d = new coDoSet(objNameBase2d.c_str(), (coDistributedObject **)objects2d);
        coDoSet *outSet3d = new coDoSet(objNameBase3d.c_str(), (coDistributedObject **)objects3d);

        // set attribute - timesteps
        string attr(ch);
        outSet2d->addAttribute("TIMESTEP", attr.c_str());
        outSet3d->addAttribute("TIMESTEP", attr.c_str());

        // delete !!!!
        int i;
        if (staGeoMultTs)
        {
            delete objects2d[0];
            delete objects3d[0];
        }
        else
        {
            for (i = 0; i < cnt; ++i)
                delete objects2d[i];
            for (i = 0; i < cnt; ++i)
                delete objects3d[i];
        }
        delete[] objects2d;
        delete[] objects3d;

        // single geometry / multiple timesteps
        if (cnt == 1)
            geoTimesetIdx_ = -1;

        geoObjs_[0] = outSet2d;
        geoObjs_[1] = outSet3d;

        READER_CONTROL->setAssocPortObj(portTok2d, outSet2d);
        READER_CONTROL->setAssocPortObj(portTok3d, outSet3d);
    }
    coModule::sendInfo(" reading geometry finished");

#ifdef DEBUG
    cerr << "readGeometry() DONE" << endl;
#endif

    return Success;
}

// read Measured geometry file or geometry files
// to a given port with ReaderControl token portTok
int
ReadEnsight::readMGeometry(const int &portTok1d)
{
    // set filenames
    string mgeoFileName = case_.getMGeoFileNm();
    if (mgeoFileName.length() < 6) // don't do anything if there is no MGeoFile
        return Failure;
    int realNumTs = 0;
    vector<string> allMGeoFiles(mkFileNames(mgeoFileName, realNumTs));
    size_t numMTs(allMGeoFiles.size());
    //const int *idxMap=NULL;

    // set object names and create object arrays for timesteps
    string objNameBase1d = READER_CONTROL->getAssocObjName(portTok1d);
    coDistributedObject **objects1d = new coDistributedObject *[numMTs + 1];

    vector<string>::iterator ii;
    if (numMTs == 0)
        allMGeoFiles.push_back(case_.getMGeoFileNm());

    // read all files
    int cnt = 0;
    for (ii = allMGeoFiles.begin(); ii != allMGeoFiles.end(); ii++)
    {
        char ch[64];
        sprintf(ch, "%d", cnt);
        string num(ch);
        string actObjNm1d(objNameBase1d);
        if (realNumTs > 1)
            actObjNm1d = objNameBase1d + "_" + num;

        // create representation of geometry file..
        EnFile *enf = EnFile::createGeometryFile(this, case_, *ii);
        if (enf == NULL)
        {
            cerr << "ReadEnsight::readGeometry(..) enf NULL " << endl;
            delete[] objects1d;
            return Failure;
        }

        MbinType_ = enf->binType();
        enf->setActiveAlloc((cnt != 0));

        if (!enf->isOpen())
        {
            cerr << "ReadEnsight::readGeometry(..) enf NOT open " << endl;
            delete[] objects1d;
            return Failure;
        }

        coModule::sendInfo(" start reading geometry ( %s ) - please be patient..", (*ii).c_str());
        enf->read(this, EnFile::DIM1D, objects1d, actObjNm1d, cnt);

        numCoordsM_.push_back(enf->getDataCont().getNumCoord());
        objects1d[cnt] = enf->getDataObject(actObjNm1d);
        if (objects1d[cnt] != NULL)
            cnt++;
        delete enf;
    }

    // we have no timesteps - feed objectsXY[0] to outports
    if (realNumTs <= 1)
    {
        if (objects1d[0] != NULL)
            READER_CONTROL->setAssocPortObj(portTok1d, objects1d[0]);
        geoObjs_[2] = objects1d[0];
    }
    else // handle timesteps
    {
        TimeSets ts(case_.getAllTimeSets());
        // create warning if more than one timeset is present
        // actually covise can handle only one timeset
        if (ts.size() > 1)
        {
            coModule::sendInfo("number of time-sets greater than one - COVISE can handle only one time-set");
        }

        vector<float> rTimes(ts[0]->getRealTimes());
        if (rTimes.empty())
            rTimes.resize(allMGeoFiles.size(), 0.0);

        char ch[64];
        bool staGeoMultTs = false;
        // recognize static geo - timedep. data
        // in this case we have to multiply the geometry
        TimeSets::const_iterator tsIter;
        int trueNumTs = 0;
        for (tsIter = ts.begin(); tsIter != ts.end(); ++tsIter)
            trueNumTs += (*tsIter)->getNumTs();
        if ((numMTs == 1) && (trueNumTs > 0))
        {
            cerr << "   static geo - timedep. data - " << trueNumTs << endl;
            staGeoMultTs = true;
            coDistributedObject **tmp = new coDistributedObject *[1 + trueNumTs];
            int jj = 0;
            for (jj = 0; jj < trueNumTs; ++jj)
            {
                incrRefCnt(objects1d[0]);
                tmp[jj] = objects1d[0];
            }
            tmp[jj] = NULL;
            objects1d = tmp;
            sprintf(ch, "1 %d", trueNumTs);
        }
        else
        {
            objects1d[cnt] = NULL;
            // set attribute - realtime
            for (int i = 0; i < cnt; ++i)
            {
                sprintf(ch, "%f", rTimes[i]);
                objects1d[i]->addAttribute("REALTIME", ch);
            }
            sprintf(ch, "1 %d", cnt);
        }
        coDoSet *outSet1d = new coDoSet(objNameBase1d.c_str(), (coDistributedObject **)objects1d);
        // set attribute - timesteps
        string attr(ch);
        outSet1d->addAttribute("TIMESTEP", attr.c_str());

        // delete !!!!
        int i;
        if (staGeoMultTs)
        {
            delete objects1d[0];
        }
        else
        {
            for (i = 0; i < cnt; ++i)
                delete objects1d[i];
        }
        // single geometry / multiple timesteps
        if (cnt == 1)
            geoTimesetIdx_ = -1;
        geoObjs_[2] = outSet1d;
        READER_CONTROL->setAssocPortObj(portTok1d, outSet1d);
    }
    delete[] objects1d;
    coModule::sendInfo(" reading geometry finished");
    return Success;
}

void
ReadEnsight::incrRefCnt(const coDistributedObject *obj)
{
    if (obj)
    {
        if (obj->isType("SETELE"))
        {
            int nSubSets = ((coDoSet *)obj)->getNumElements();
            int i;
            for (i = 0; i < nSubSets; ++i)
            {
                const coDistributedObject *subSetEle = ((coDoSet *)obj)->getElement(i);
                incrRefCnt(subSetEle);
            }
        }
        obj->incRefCount();
    }
    //cerr << "    INCR-REFCNT for obj " << obj->getName() << endl;
}

EnFile *
ReadEnsight::createDataFilePtr(const string &filename,
                               const int &d,
                               const int &numCoord)
{
    // create representation of data file..
    // and check file type
    EnFile *enf = NULL;
    switch (binType_)
    {
    case EnFile::CBIN:
    case EnFile::FBIN:
        if (case_.getVersion() == CaseFile::gold)
        {
            //cerr << "ReadEnsight::createDataFilePtr(..) DataFileGoldBin to be created" << endl;
            enf = new DataFileGoldBin(this, filename, d, numCoord, binType_);
        }
        else
        {
            //cerr << "ReadEnsight::createDataFilePtr(..) DataFileBin CBIN to be created" << endl;
            enf = new DataFileBin(this, filename, d, numCoord, binType_);
        }
        break;

    case EnFile::NOBIN:
        if (case_.getVersion() == CaseFile::gold)
        {
            //cerr << "ReadEnsight::createDataFilePtr(..) DataFileGold to be created" << endl;
            enf = new DataFileGold(this, filename, d, numCoord);
        }
        else
        {
            //cerr << "ReadEnsight::createDataFilePtr(..) DataFileAsc to be created" << endl;
            enf = new DataFileAsc(this, filename, d, numCoord);
        }
        break;
    case EnFile::UNKNOWN:
        break;
    }
    if (enf)
    {
        enf->setDataByteSwap(dataByteSwapParam_->getValue() == 1);
        enf->setIncludePolyeder(includePolyederParam_->getValue() == 1);
    }
    return enf;
}

// helper for static geometry / transient data
// extends all arrays
void
ReadEnsight::extendArrays(const int &numTimesteps)
{
    //cerr << "ReadEnsight::extendArrays(..) geoTimesetIdx " << geoTimesetIdx_  << endl;
    if (geoTimesetIdx_ == -1)
    {
        int nc(numCoords_[0]);
        //       int ne( numElem_[0] );
        //const int *idxMap = idxMaps_[0];
        PartList pl(globalParts_[0]);

        //       numCoords_.clear();
        //       numElem_.clear();
        //idxMaps_.clear();
        globalParts_.clear();

        numCoords_.insert(numCoords_.begin(), numTimesteps, nc);
        //       numElem_.insert( numElem_.begin(), numTimesteps, ne );
        //idxMaps_.insert( idxMaps_.begin(), numTimesteps, (int *) idxMap );
        globalParts_.insert(globalParts_.begin(), numTimesteps, pl);
    }
}

int
ReadEnsight::compute(const char *)
{
    numCoords_.clear();
    numCoordsM_.clear();
    AutoColors::instance()->reset();
    time_t anfT = time(NULL);

    if (case_.empty())
    {
        cerr << "ReadEnsight::compute(..) case file not found  " << endl;
        Covise::sendError("case file not found");
        return 0;
    }

    if (masterPL_.empty())
        createMasterPL();

    string partStr(partStringParam_->getValue());
    if (evalPartString(partStr) == false)
        return STOP_PIPELINE;

    // lists with desc labels to store covise files
    map<int, string> portVals;

    int state, state2;

    // read geometry
    state = readGeometry(GEOPORT2D, GEOPORT3D);
    state2 = readMGeometry(GEOPORT1D);
    portVals[GEOPORT1D] = string("Grid1D");
    portVals[GEOPORT2D] = string("Grid2D");
    portVals[GEOPORT3D] = string("Grid3D");

    if (state == Failure && state2 == Failure)
    {
        cerr << "ReadEnsight::compute(..) neither geometry nor measured points could be read  " << endl;
        return 0;
    }

    // this flag is set to true only if the case file has changed
    readGeo_ = false;

    string desc;

#ifdef DEBUG
    cerr << " elem: " << globalParts_.at(0).at(0).subParts_numElem.size() << " conn: " << globalParts_.at(0).at(0).subParts_numConn.size() << endl;
#endif

    // now read data
    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 1 (1D)
    ////////////////////////////////////////////////////////////////////////////////
    int pos = READER_CONTROL->getPortChoice(DPORT1_1D);

    bool vertexData;
    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData1d(DPORT1_1D, dataFileName, vertexData, 1, desc);
        portVals[DPORT1_1D] = desc + "_1D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 1 (2D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT1_2D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData2d(DPORT1_2D, dataFileName, vertexData, 1, desc);
        portVals[DPORT1_2D] = desc + "_2D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 1 (3D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT1_3D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData3d(DPORT1_3D, dataFileName, vertexData, 1, desc);
        portVals[DPORT1_3D] = desc + "_3D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 2 (1D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT2_1D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData1d(DPORT2_1D, dataFileName, vertexData, 1, desc);
        portVals[DPORT2_1D] = desc + "_1D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 2 (2D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT2_2D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        readData2d(DPORT2_2D, dataFileName, vertexData, 1, desc);
        portVals[DPORT2_2D] = desc + "_2D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 2 (3D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT2_3D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        readData3d(DPORT2_3D, dataFileName, vertexData, 1, desc);
        portVals[DPORT2_3D] = desc + "_3D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 3 (1D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT3_1D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData1d(DPORT3_1D, dataFileName, vertexData, 1, desc);
        portVals[DPORT3_1D] = desc + "_1D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 3 (2D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT3_2D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        readData2d(DPORT3_2D, dataFileName, vertexData, 1, desc);
        portVals[DPORT3_2D] = desc + "_2D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 3 (3D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT3_3D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        readData3d(DPORT3_3D, dataFileName, vertexData, 1, desc);
        portVals[DPORT3_3D] = desc + "_3D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 4 (1D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT4_1D);

    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData1d(DPORT4_1D, dataFileName, vertexData, 1, desc);
        portVals[DPORT4_1D] = desc + "_1D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- vector data ---    D P O R T 4 (2D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT4_2D);

    // obtain filename tied to choice value
    // obtain if vertex or cell based data
    // pos 1 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                break;
            case DataItem::vector:
                cnt++;
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        // now read it
        readData2d(DPORT4_2D, dataFileName, vertexData, 3, desc);
        portVals[DPORT4_2D] = desc + "_2D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- vector data ---    D P O R T 4 (3D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT4_3D);

    // obtain filename tied to choice value
    // obtain if vertex or cell based data
    // pos 1 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                break;
            case DataItem::vector:
                cnt++;
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        // now read it
        readData3d(DPORT4_3D, dataFileName, vertexData, 3, desc);
        portVals[DPORT4_3D] = desc + "_3D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- scalar data ---    D P O R T 5 (1D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT5_1D);
    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                cnt++;
                break;
            case DataItem::vector:
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }

        readData1d(DPORT5_1D, dataFileName, vertexData, 1, desc);
        portVals[DPORT5_1D] = desc + "_1D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- vector data ---    D P O R T 5 (2D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT5_2D);

    // obtain filename tied to choice value
    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                break;
            case DataItem::vector:
                cnt++;
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        readData2d(DPORT5_2D, dataFileName, vertexData, 3, desc);
        portVals[DPORT5_2D] = desc + "_2D";
    }

    ////////////////////////////////////////////////////////////////////////////////
    // -- vector data ---    D P O R T 5 (3D)
    ////////////////////////////////////////////////////////////////////////////////
    pos = READER_CONTROL->getPortChoice(DPORT5_3D);

    // obtain filename tied to choice value
    // pos 0 is by definition NONE
    if (pos > 0)
    {
        DataList dl = case_.getDataIts();
        DataList::iterator it;
        string dataFileName;

        int cnt(0);
        for (it = dl.begin(); it != dl.end(); ++it)
        {
            switch ((*it).getType())
            {
            case DataItem::scalar:
                break;
            case DataItem::vector:
                cnt++;
                break;
            case DataItem::tensor:
                // fill in ports for tensor data
                break;
            }
            if (cnt == pos)
            {
                dataFileName = case_.getDir() + string("/") + trim((*it).getFileName());
                vertexData = (*it).perVertex();
                desc = (*it).getDesc();
                break;
            }
        }
        readData3d(DPORT5_3D, dataFileName, vertexData, 3, desc);
        portVals[DPORT5_3D] = desc + "_3D";
    }

    // clean up part list
    unsigned int i, j;
    // check if we have static geo with transient data
    size_t indexSteps = (geoTimesetIdx_ == -1) ? 1 : globalParts_.size();
    for (i = 0; i < indexSteps; ++i)
    {
        for (j = 0; j < globalParts_[i].size(); ++j)
        {
            globalParts_[i][j].clearFields();
            // 	      if (globalParts_[i][j].indexMap2d_!=NULL) {
            // 	          delete [] globalParts_[i][j].indexMap2d_;
            // 	      }
            // 	      if (globalParts_[i][j].indexMap3d_!=NULL) {
            // 		  delete [] globalParts_[i][j].indexMap3d_;
            // 	      }
        }
        globalParts_[i].clear();
    }
    globalParts_.clear();

    // clean up idxMaps_
    // in case stat Geo - moving data only one entry exists
    //    if ( geoTimesetIdx_ == -1 ) {
    //        delete [] idxMaps_[0];
    //    }
    //    else {
    //        for (i=0; i<idxMaps_.size(); ++i) {
    // 	   delete [] idxMaps_[i];
    //        }
    //    }
    //    idxMaps_.clear();

    if (storeGrpParam_->getValue())
    {
        if (!READER_CONTROL->storePortObj(case_.getDir(), case_.getProjectName() + ".covgrp", portVals))
        {
            coModule::sendError("Failed to store COVISE files in the Ensight data directory. Please check your permissions.");
        }
    }

    char buf[256];
    sprintf(buf, "ReadEnsight finished - run took %ld seconds", (long)(time(NULL) - anfT));
    coModule::sendInfo("%s", buf);

    // clean up geoObjs
    //    if ( geoObjs_[0] != NULL ) delete [] geoObjs_[0];
    //    if ( geoObjs_[1] != NULL ) delete [] geoObjs_[1];

    // return 0;
    return Success;
}

bool
ReadEnsight::evalPartString(const string &partStr)
{
    if (!masterPL_.empty())
    {
        size_t numParts(masterPL_.size());
        int i;
        // all means all parts will be used - nothing to do
        if (partStr.find("all") != string::npos)
        {
            for (i = 0; i < numParts; i++)
            {
                masterPL_[i].activate(true);
            }

            return true;
        }
        coRestraint res;
        res.add(partStr.c_str());
        for (i = 0; i < numParts; i++)
        {
            if (res(masterPL_[i].getPartNum()))
                masterPL_[i].activate(true);
            else
                masterPL_[i].activate(false);
        }

        /*
      bool *blacklist = new bool[ numParts ];

      for ( i=0; i<numParts; i++ ) blacklist[i]=false;

      size_t pos, beg, raPos;
      string atom;

      pos = 0;
      beg = 0;
      // check if the string contains bad characters
      CoviseRegexp  regexp("[A-Za-z+]");
      if ( regexp.isMatching( partStr.c_str() ) ) {
          cerr << "ReadEnsight::evalPartString( ..) wrong part STRING exiting.." << endl;
      #ifdef WIN32
            DebugBreak();
      #endif
          sendError("WRONG PART STRING %s",  partStr.c_str());
          stopPipeline();
          return false;
      }

      // separate string in atoms and parse it
      // fill blacklist
      while ( pos != string::npos ) {
          pos =  partStr.find(",", pos+1);
          atom = partStr.substr(beg, pos-beg);
          //cerr << "ReadEnsight::evalPartString(..) ATOM <" << atom << ">" << endl;
          raPos = atom.find("-");
          if ( raPos != string::npos ) {
         int rBeg = atoi( atom.substr(0,raPos).c_str() );
         int rEnd = atoi( atom.substr(raPos+1).c_str() );

         for ( i=rBeg; i<=rEnd; i++ ) {
             if ( i < numParts ) blacklist[i]=true;
         }
          }
          else {
         int rPos( atoi(atom.c_str()) );
         if ( rPos < numParts ) blacklist[rPos]=true;
          }
          beg = pos+1;
      }
      for ( i=0; i<numParts; i++ ) masterPL_[i].activate( blacklist[i] );
      */
    }
    else
    {
        cerr << "ReadEnsight::evalPartString(..) master part list empty" << endl;
    }
    return true;
}

coDistributedObject **
ReadEnsight::createGeoOutObj(const string &baseName2d,
                             const string &baseName3d,
                             const int &step)
{
#ifdef DEBUG
    cerr << "createGeoOutObj()" << endl;
#endif


    if (step >= (int)globalParts_.size())
        return NULL;

    coDistributedObject **retArr = new coDistributedObject *[3];

    PartList thePl = globalParts_[step];
    int numActiveSetEle = 0;
    PartList::iterator it = thePl.begin();
    while (it != thePl.end())
    {
        if (it->isActive())
            numActiveSetEle++;
        it++;
    }

    coDistributedObject **objects = new coDistributedObject *[numActiveSetEle + 1];

    it = thePl.begin();

    float *x = NULL, *y = NULL, *z = NULL;
    int numCoordsGlob = 0, numCoords = 0;
    // if we have Ensight 5,6 the first part contains the coordinates
    // rem: we have a global coordinate list in this case
    if (case_.getVersion() != CaseFile::gold)
    {
        if (it != thePl.end())
        {
            x = it->x3d_;
            y = it->y3d_;
            z = it->z3d_;
            numCoordsGlob = it->numCoords();
        }
    }
    int cnt = 0;
    while (it != thePl.end())
    {
        if (it->numEleRead3d() > 0 && it->isActive())
        {
#ifdef DEBUG
            cerr << " 3D part with " << it->numEleRead3d() << " elements";
#endif

            coDistributedObject *tmp;

            char cn[16];
            sprintf(cn, "%d", it->getPartNum());
            string oNm = baseName3d + "_el_" + string(cn);

            // if we have Ensight gold coordinate lists belong to parts
            DataCont dc;
            if (case_.getVersion() == CaseFile::gold)
            {
                x = it->x3d_;
                y = it->y3d_;
                z = it->z3d_;
                numCoords = it->numCoords();
                numCoordsGlob += numCoords;
                dc.setNumCoord(numCoords);
            }
            else
            {
                dc.setNumCoord(numCoordsGlob);
            }

            it->subParts_numElem.clear();
            it->subParts_numConn.clear();

            if (it->numConnRead3d() <= MAX_LIST_SIZE)
            {
#ifdef DEBUG
                cerr << "  no split" << endl;
#endif

                // reduce coordinates - remove unused coordinates and
                // adopt the corner-list
                dc.setNumElem(it->numEleRead3d());
                dc.setNumConn(it->numConnRead3d());
                dc.x = x;
                dc.y = y;
                dc.z = z;
                dc.el = it->el3d_;
                dc.cl = it->cl3d_;
                dc.tl = it->tl3d_;

                float *xn = NULL, *yn = NULL, *zn = NULL;
                Reducer r(dc);
                r.removeUnused(&xn, &yn, &zn);

                // store the index-map: it maps the global coordinate array to
                // the coordinate array of each part
                (*it).indexMap3d_ = (int *)r.getIdxMap();
                //cerr << "       IDX-MAP SET "  << (*it).indexMap3d_ << "  " << numCoords << endl;

                // coDoUnstructuredGrid *tmp = new coDoUnstructuredGrid( oNm.c_str(),
                // dc.getNumElem(),
                // dc.getNumConn(),
                // dc.getNumCoord(),
                // dc.el, dc.cl,
                // xn, yn, zn, dc.tl );

                tmp = new coDoUnstructuredGrid(oNm.c_str(),
                                               dc.getNumElem(),
                                               dc.getNumConn(),
                                               dc.getNumCoord(),
                                               dc.el, dc.cl,
                                               xn, yn, zn, dc.tl);

                delete[] xn;
                delete[] yn;
                delete[] zn;
            }

            /**************************************************************/
            /* Split into multiple unstructured grids for very large datasets  */
            /**************************************************************/

            else
            {
#ifdef DEBUG
                cerr << "  split" << endl;
#endif

                // determine size of subParts
                uint64_t elemOffset(0);
                uint64_t connOffset(0);
                for (int i = 0; i < it->numEleRead3d(); ++i)
                {
                    uint64_t nextElementsConn = (i < it->numEleRead3d() - 1) ? (it->el3d_[i + 1]) : (it->numConnRead3d());
                    if (nextElementsConn > connOffset + MAX_LIST_SIZE)
                    {
                        // found a reason to split
                        // i is the first element after the split
                        unsigned int numElem = i - elemOffset;
                        unsigned int numConn = it->el3d_[i] - connOffset;
                        it->subParts_numElem.push_back(numElem);
                        it->subParts_numConn.push_back(numConn);
                        elemOffset += numElem;
                        connOffset += numConn;
                    }
                }
                it->subParts_numElem.push_back(it->numEleRead3d() - elemOffset);
                it->subParts_numConn.push_back(it->numConnRead3d() - connOffset);
                // prepare
                int numberOfSubParts = it->subParts_numElem.size();
                coDistributedObject **subObjects = new coDistributedObject *[numberOfSubParts + 1];
                subObjects[numberOfSubParts] = NULL;
                // work on the sub parts
                elemOffset = 0;
                connOffset = 0;
                for (int subPart = 0; subPart < numberOfSubParts; ++subPart)
                {
                    // split (create temp lists)
                    int currentNumElem = it->subParts_numElem.at(subPart);
                    int currentNumConn = it->subParts_numConn.at(subPart);

#ifdef DEBUG
                    cerr << "   " << currentNumElem << " elements, " << currentNumConn << " connections" << endl;
                    cerr << "    copy lists" << endl;
#endif

                    int *tmp_el = new int[currentNumElem];
                    int *tmp_tl = new int[currentNumElem];
                    for (int i = 0; i < currentNumElem; ++i)
                    {
                        tmp_el[i] = it->el3d_[elemOffset + i] - connOffset;
                        tmp_tl[i] = it->tl3d_[elemOffset + i];
                    }
                    int *tmp_cl = new int[currentNumConn];
                    for (int i = 0; i < currentNumConn; ++i)
                        tmp_cl[i] = it->cl3d_[connOffset + i];
// standard reduce
#ifdef DEBUG
                    cerr << "    reduce" << endl;
#endif

                    DataCont dc;
                    dc.setNumCoord(numCoords);
                    dc.setNumElem(currentNumElem);
                    dc.setNumConn(currentNumConn);
                    dc.x = x;
                    dc.y = y;
                    dc.z = z;
                    dc.el = tmp_el;
                    dc.cl = tmp_cl;
                    dc.tl = tmp_tl;
                    float *xn = NULL, *yn = NULL, *zn = NULL;
                    Reducer r(dc);
                    r.removeUnused(&xn, &yn, &zn);
                    (*it).indexMap3d_ = (int *)r.getIdxMap(); // TODO
// create grid
#ifdef DEBUG
                    cerr << "    create grid" << endl;
#endif

                    char sp[16];
                    sprintf(sp, "%d", subPart);
                    string subName = oNm + "_sub_" + string(sp);
                    subObjects[subPart] = new coDoUnstructuredGrid(subName.c_str(),
                                                                   dc.getNumElem(),
                                                                   dc.getNumConn(),
                                                                   dc.getNumCoord(),
                                                                   dc.el, dc.cl,
                                                                   xn, yn, zn, dc.tl);
                    //             // clean tmp lists
                    delete[] xn;
                    delete[] yn;
                    delete[] zn;
                    delete[] tmp_el;
                    delete[] tmp_cl;
                    delete[] tmp_tl;
#ifdef DEBUG
                    cerr << "    done" << endl;
#endif

                    // prepare next
                    elemOffset += currentNumElem;
                    connOffset += currentNumConn;
                }
                // create set and clean up
                tmp = new coDoSet(oNm.c_str(), subObjects);
                for (int i = 0; i < numberOfSubParts; ++i)
                    delete subObjects[i];
                delete[] subObjects;
            }

            // remove trailing blanks
            string partname = it->comment();
            int idx = partname.length() - 1;
            while (idx >= 0)
            {
                idx--;
            }
            idx = partname.length() - 1;
            while (idx >= 0 && partname.at(idx) <= 32)
            {
                idx--;
            }
            if (idx >= 0)
            {
                partname = partname.substr(0, idx + 1);
            }
            else
            {
                partname = "";
            }
            tmp->addAttribute("PART", (partname).c_str());
            objects[cnt] = tmp;
            (*it).distGeo3d_ = tmp;

            // clean el, cl, typeLst
            // will be deleted in EnPart clearFieldsdelete [] dc.el;
            // will be deleted in EnPart clearFieldsdelete [] dc.cl;
            // will be deleted in EnPart clearFieldsdelete [] it->tl2d_;
            // delete [] xn;    delete [] yn;     delete [] zn;

            cnt++;
        }
        it++;
    }
    numCoords_.push_back(numCoordsGlob);
    objects[cnt] = NULL;

    retArr[0] = new coDoSet(baseName3d.c_str(), (coDistributedObject **)objects);

    // delete !!!!
    int i;
    for (i = 0; i < cnt; ++i)
        delete objects[i];
    delete[] objects;

    //////////////////////////////////////////
    // now we're working on the 2d elements
    //////////////////////////////////////////
    numActiveSetEle = 0;
    it = thePl.begin();
    while (it != thePl.end())
    {
        if (it->isActive())
            numActiveSetEle++;
        it++;
    }

    if (numActiveSetEle == 0)
        return NULL;

    it = thePl.begin();
    coDistributedObject **objects2d = new coDistributedObject *[numActiveSetEle + 1];
    cnt = 0;

    while (it != thePl.end())
    {
        char cn[16];
        sprintf(cn, "%d", it->getPartNum());
        string oNm = baseName2d + "_el_" + string(cn);

        // remove trailing blanks
        string partname = it->comment();
        int idx = partname.length() - 1;
        while (idx >= 0)
        {
            idx--;
        }
        idx = partname.length() - 1;
        while (idx >= 0 && partname.at(idx) <= 32)
        {
            idx--;
        }
        if (idx >= 0)
        {
            partname = partname.substr(0, idx + 1);
        }
        else
        {
            partname = "";
        }

        if (it->numCoords() > 0 && it->numEleRead3d() == 0 && it->numEleRead2d() == 0 && it->isActive()) // HACK: if no elements read, output coordinates as points at the 2D port
        {
            coDoPoints *tmp = new coDoPoints(oNm.c_str(), it->numCoords(), it->x3d_, it->y3d_, it->z3d_);
            tmp->addAttribute("PART", (partname).c_str());
            objects2d[cnt++] = tmp;
            (*it).distGeo2d_ = tmp;
        }
        else if ((it->numEleRead2d() > 0) && (it->isActive()))
        {
            DataCont dc;
            if (case_.getVersion() == CaseFile::gold)
            {
                x = it->x3d_;
                y = it->y3d_;
                z = it->z3d_;
                numCoords = it->numCoords();
                numCoordsGlob += numCoords;
                dc.setNumCoord(numCoords);
            }
            else
            {
                dc.setNumCoord(numCoordsGlob);
            }
            // as the reducer is build upon a DataCont we gonna use it here
            // we may create a better interface later to save up some more
            // lines of code
            dc.setNumElem(it->numEleRead2d());
            dc.setNumConn(it->numConnRead2d());
            dc.x = x;
            dc.y = y;
            dc.z = z;
            dc.el = it->el2d_;
            dc.cl = it->cl2d_;
            dc.tl = it->tl2d_;

            float *xn = NULL, *yn = NULL, *zn = NULL;

            Reducer r(dc);
            r.removeUnused(&xn, &yn, &zn);
            // store the index-map: it maps the global coordinate array to
            // the coordinate array of each part
            it->indexMap2d_ = (int *)r.getIdxMap();

            coDoPolygons *tmp = new coDoPolygons(oNm.c_str(),
                                                 dc.getNumCoord(), xn, yn, zn,
                                                 dc.getNumConn(), dc.cl,
                                                 dc.getNumElem(), dc.el);

            tmp->addAttribute("PART", partname.c_str());
            if (autoColoring_->getValue())
            {
                tmp->addAttribute("COLOR", (AutoColors::instance()->next()).c_str());
            }

            it->distGeo2d_ = tmp;

            objects2d[cnt] = tmp;

            // clean el, cl, typeLst
            // will be deleted in EnPart clearFieldsdelete [] dc.el;
            // will be deleted in EnPart clearFieldsdelete [] dc.cl;
            // will be deleted in EnPart clearFieldsdelete [] it->tl2d_;
            delete[] xn;
            delete[] yn;
            delete[] zn;
            cnt++;
        }
        it++;
    }
    objects2d[cnt] = NULL;

    retArr[1] = new coDoSet(baseName2d.c_str(), (coDistributedObject **)objects2d);

    // clean up
    for (i = 0; i < cnt; ++i)
        delete objects2d[i];
    delete[] objects2d;

    // will be deleted in EnPart clearFields delete [] x; delete [] y; delete [] z;

    // we have modified the local part list -> restore
    globalParts_[step] = thePl;
    return retArr;
}

coDistributedObject **
ReadEnsight::createDataOutObj(EnFile::dimType dim, const string &baseName,
                                DataCont &dcIn,
                                const int &step, const bool &perVertex)
{
#ifdef DEBUG
    cerr << "createDataOutObj3d()" << endl;
#endif
    int numGeometries = (int)globalParts_.size();
    if (step >= numGeometries)
        return NULL;


    coDistributedObject **retArr = new coDistributedObject *[3];
    PartList &thePl=globalParts_[step];
    int numActiveSetEle = 0;
    PartList::iterator it = thePl.begin();
    while (it != thePl.end())
    {
        if (it->isActive())
            numActiveSetEle++;
        it++;
    }

    coDistributedObject **objects = new coDistributedObject *[numActiveSetEle + 1];
    it = thePl.begin();

    bool scalarData = ((dcIn.y == NULL) && (dcIn.z == NULL));
    // //////////////////////////////////////////////
    //     3d - parts
    // /////////////////////////////////////////////
    int cnt = 0;
    while (it != thePl.end())
    {
        int numElem;
        if (dim == EnFile::EnFile::DIM2D)
            numElem = it->numEleRead2d();
        if (dim == EnFile::EnFile::DIM3D)
            numElem = it->numEleRead3d();
        if ((numElem > 0) && (it->isActive()) || (dim == EnFile::EnFile::DIM2D && (it->numCoords() > 0 && it->numEleRead3d() == 0 && it->numEleRead2d() == 0))) // HACK: if no elements read, output coordinates as points at the 2D port)
        {
            char cn[16];
            sprintf(cn, "%d", it->getPartNum());
            string oNm = baseName + "t"+std::to_string(step)+"_el_" + string(cn);
            string oNmCTV = baseName + "t" + std::to_string(step) + "_elv_" + string(cn);
            coDistributedObject *tmp = NULL;
            if (perVertex)
            {
#ifdef DEBUG
                cerr << " vertex data" << endl;
#endif
                int *index = NULL;
                if (dim == EnFile::EnFile::DIM2D)
                    index = it->indexMap2d_;
                if (dim == EnFile::EnFile::DIM3D)
                    index = it->indexMap3d_;
                Reducer red(dcIn, index);
                //cerr << "       IDX-MAP GET "  << (*it).indexMap3d_ << "   ";
                //cerr << dcIn.getNumCoord() << endl;

                DataCont dcOut;
                if (case_.getVersion() == CaseFile::gold)
                {

                    dcIn.setNumCoord(it->numCoords());
                    dcIn.x = it->arr1_;
                    dcIn.y = it->arr2_;
                    dcIn.z = it->arr3_;

                    // set data to 0 if a part without data arrives
                    // if (dcOut.x == NULL)
                    if (dcIn.x == NULL)
                    {
                        cerr << " DATA  NULL " << it->comment() << endl;
                        if (scalarData)
                        {
                            dcIn.x = new float[it->numCoords()];
                            fill(dcIn.x, dcIn.x + it->numCoords(), 0.0);
                        }
                        else
                        {
                            dcIn.x = new float[it->numCoords()];
                            fill(dcIn.x, dcIn.x + it->numCoords(), 0.0);
                            dcIn.y = new float[it->numCoords()];
                            fill(dcIn.y, dcIn.y + it->numCoords(), 0.0);
                            dcIn.z = new float[it->numCoords()];
                            fill(dcIn.z, dcIn.z + it->numCoords(), 0.0);
                        }
                    }
                    it->arr1_ = NULL;
                    it->arr2_ = NULL;
                    it->arr3_ = NULL;
                }
                dcOut = red.reduceAndCopyData();


                if (scalarData)
                {
                    tmp = new coDoFloat(oNm.c_str(),
                                        dcOut.getNumCoord(),
                                        dcOut.x);
                }
                else
                {
                    tmp = new coDoVec3(oNm.c_str(),
                                       dcOut.getNumCoord(),
                                       dcOut.x, dcOut.y, dcOut.z);
                }
                dcOut.cleanAll();
            }

            /************/
            /* Cell Data */
            /************/

            else
            {
#ifdef DEBUG
                cerr << " cell data" << endl;
                cerr << " elem: " << it->subParts_numElem.size() << " conn: " << it->subParts_numConn.size() << endl;
#endif
                float *dx_, *dy_, *dz_;
                int numElem;
                if (dim == EnFile::DIM2D)
                {
                    dx_ = it->d2dx_; dy_ = it->d2dy_; dz_ = it->d2dz_; 
                    numElem = it->numEleRead2d();
                }
                if (dim == EnFile::DIM3D)
                {
                    dx_ = it->d3dx_; dy_ = it->d3dy_; dz_ = it->d3dz_;
                    numElem = it->numEleRead3d();
                }
                scalarData = (dy_ == NULL) && (dz_ == NULL);
                if (it->subParts_numElem.empty())
                {
#ifdef DEBUG
                    cerr << "  default" << endl;
#endif

                    if (dx_ != NULL)
                    {
                        if (scalarData)
                        {
                            //cerr << "ReadEnsight::createDataOutObj(..) Obj Name ";
                            //cerr << oNm.c_str() << "  DATAPTR " << it->d3dx_;
                            //cerr << " numEleRead3d " << it->numEleRead3d() << endl;
                            tmp = new coDoFloat(oNm.c_str(),
                                                numElem,
                                                dx_);
                            delete[] dx_;
                            if (dim == EnFile::DIM2D)
                            {
                                it->d2dx_ = NULL;
                            }
                            if (dim == EnFile::DIM3D)
                            {
                                it->d3dx_ = NULL;
                            }
                        }
                        else
                        {
                            tmp = new coDoVec3((oNm + "t").c_str(),
                                               numElem,
                                               dx_, dy_, dz_);
                            delete[] dx_; delete[] dy_; delete[] dz_;
                            if (dim == EnFile::DIM2D)
                            {
                                it->d2dx_ = NULL; it->d2dy_ = NULL; it->d2dz_ = NULL;
                            }
                            if (dim == EnFile::DIM3D)
                            {
                                it->d3dx_ = NULL; it->d3dy_ = NULL; it->d3dz_ = NULL;
                            }
                        }
                        // integrate CellToVert
                        const coDistributedObject *geoObj = getGeoObject(step, cnt, dim);
                        coCellToVert c;
                        tmp = c.interpolate(geoObj, tmp, oNmCTV.c_str());
                    }
                    else
                    {
                        cerr << " WARNING EMPTY DATA for " << oNm << endl;
                        float dummy = 0;
                        tmp = new coDoFloat(oNm.c_str(), 1, &dummy);
                    }
                }
                else
                {
// splitted geometry !!!
#ifdef DEBUG
                    cerr << "  split" << endl;
#endif

                    // prepare
                    int numberOfSubParts = it->subParts_numElem.size();
                    coDistributedObject **subObjects = new coDistributedObject *[numberOfSubParts + 1];
                    subObjects[numberOfSubParts] = NULL;
                    const coDistributedObject *geoObj = getGeoObject(step, cnt, dim);
                    // work on the sub parts
                    int offset(0);
                    for (int subPart = 0; subPart < numberOfSubParts; ++subPart)
                    {
#ifdef DEBUG
                        cerr << "   subPart: " << subPart << endl;
#endif

                        char c[16];
                        sprintf(c, "%d", subPart);
                        string oNmsub = oNm + "_sp_" + string(c);
                        string oNmCTVsub = oNmCTV + "_sp_" + string(c);

                        if (dx_ != NULL)
                        {
// split (create temp lists)
#ifdef DEBUG
                            cerr << "    create lists" << endl;
#endif

                            int currentNum = it->subParts_numElem.at(subPart);
                            float *tmp_x;
                            float *tmp_y;
                            float *tmp_z;
                            if (scalarData)
                            {
                                tmp_x = new float[currentNum];
                            }
                            else
                            {
                                tmp_x = new float[currentNum];
                                tmp_y = new float[currentNum];
                                tmp_z = new float[currentNum];
                            }
                            for (int i = 0; i < currentNum; ++i)
                            {
                                if (scalarData)
                                {
                                    tmp_x[i] = dx_[offset + i];
                                }
                                else
                                {
                                    tmp_x[i] = dx_[offset + i];
                                    tmp_y[i] = dy_[offset + i];
                                    tmp_z[i] = dz_[offset + i];
                                }
                            }
// standard create
#ifdef DEBUG
                            cerr << "    create grid" << endl;
#endif

                            if (scalarData)
                            {
                                subObjects[subPart] = new coDoFloat(oNmsub.c_str(), // TODO: name
                                                                    currentNum,
                                                                    tmp_x);
                            }
                            else
                            {
                                subObjects[subPart] = new coDoVec3((oNmsub + "t").c_str(), // TODO: name
                                                                   currentNum,
                                                                   tmp_x, tmp_y, tmp_z);
                            }
// integrate CellToVert
#ifdef DEBUG
                            cerr << "    CellToVert" << endl;
#endif

                            coCellToVert c;
                            subObjects[subPart] = c.interpolate(((coDoSet *)geoObj)->getElement(subPart), subObjects[subPart], oNmCTVsub.c_str()); // TODO: name
                            // prepare next
                            offset += currentNum;
#ifdef DEBUG
                            cerr << "    done" << endl;
#endif
                        }
                        else
                        {
                            float dummy = 0;
                            subObjects[subPart] = new coDoFloat(oNmsub.c_str(), 1, &dummy);
                        }
                    }
#ifdef DEBUG
                    cerr << "   loop finished" << endl;
#endif

                    // create set and clean up
                    tmp = new coDoSet(oNmCTV.c_str(), subObjects);
                    for (int i = 0; i < numberOfSubParts; ++i)
                        delete subObjects[i];
                    delete[] subObjects;
#ifdef DEBUG
                    cerr << "   done" << endl;
#endif
                }
            }
            //tmp->addAttribute("PART",(it->comment()).c_str());

            objects[cnt] = tmp;
            cnt++;
        }
        it++;
    }
    objects[cnt] = NULL;

    retArr[0] = new coDoSet((baseName + "t" + std::to_string(step)).c_str(), (coDistributedObject **)objects);

    // delete !!!!
    int i;
    for (i = 0; i < cnt; ++i)
        delete objects[i];
    delete[] objects;

    return retArr;
}
int
ReadEnsight::readData1d(const int &portTok1d,
                        const string &fileNameBase,
                        const bool &pV, const int &dim, const string &desc)
{
    int rNumTs;
    vector<string> allFiles(mkFileNames(fileNameBase, rNumTs));
    int totNumTs(allFiles.size());

    if (totNumTs == 0)
        allFiles.push_back(fileNameBase);

    // we want to have a set only if we have transient data
    if (rNumTs > 0)
        extendArrays(rNumTs);

    // set object names and create object arrays for timesteps
    string objNameBase1d = READER_CONTROL->getAssocObjName(portTok1d);
    coDistributedObject **objects1d = new coDistributedObject *[rNumTs + 1];

    vector<string>::iterator ii;

    EnFile::BinType oldBinType_ = binType_;
    binType_ = MbinType_;

    int cnt(0);
    for (ii = allFiles.begin(); ii != allFiles.end(); ii++)
    {
        char ch[64];
        sprintf(ch, "%d", cnt);
        string num(ch);
        string actObjNm1d(objNameBase1d);
        if (rNumTs > 1)
            actObjNm1d = objNameBase1d + "_" + num;

        char buf[256];
        switch (dim)
        {
        case 1:
            sprintf(buf, "ReadEnsight start reading scalar data out of file %s", (*ii).c_str());
            break;
        case 3:
            sprintf(buf, "ReadEnsight start reading vector data out of file %s", (*ii).c_str());
            break;
        }
        coModule::sendInfo("%s", buf);

        // really read the data
        DataCont ddc;
        EnFile *dFile(NULL);
        if (pV) // per vertex
        {
            dFile = createDataFilePtr(*ii, dim, numCoordsM_[cnt]);
            if (!dFile->isOpen())
            {
                sendError(" could not open file %s", (*ii).c_str());
                return 0;
            }
            dFile->read(this, EnFile::EnFile::DIM1D, objects1d, actObjNm1d, cnt);
         /*   objects1d[cnt] = dFile->getDataObject(actObjNm1d);
            if (objects1d[cnt] != NULL)
            {
                ++cnt;
                objects1d[cnt] = NULL;
            }
            
            ddc = dFile->getDataCont();
            ddc.cleanAll();*/

            delete dFile;
        }
    }

    // we have no timesteps - feed objectsXY[0] to outports
    if (rNumTs <= 1)
    {
        if (!desc.empty())
        {
            objects1d[0]->addAttribute("SPECIES", desc.c_str());
        }
        if (objects1d[0] != NULL)
            READER_CONTROL->setAssocPortObj(portTok1d, objects1d[0]);
        delete[] objects1d;
    }
    else
    {
        TimeSets ts(case_.getAllTimeSets());
        // create warning if more than one timeset is present
        // actually covise can handle only one timeset
        if (ts.size() > 1)
        {
            coModule::sendInfo("number of time-sets greater than one - COVISE can handle only one time-set");
        }

        vector<float> rTimes(ts[0]->getRealTimes());

        objects1d[cnt] = NULL;

        // set attribute - realtime
        char ch[64];
        for (int i = 0; i < cnt; ++i)
        {
            sprintf(ch, "%f", rTimes[i]);
            objects1d[i]->addAttribute("REALTIME", ch);
        }

        coDoSet *outSet1d = new coDoSet(objNameBase1d.c_str(), (coDistributedObject **)objects1d);

        // set attribute - timesteps
        sprintf(ch, "1 %d", cnt);
        string attr(ch);
        outSet1d->addAttribute("TIMESTEP", attr.c_str());

        // delete !!!!
        int i;
        for (i = 0; i < cnt; ++i)
            delete objects1d[i];
        delete[] objects1d;

        // single geometry / multiple timesteps
        if (cnt == 1)
            geoTimesetIdx_ = -1;

        if (!desc.empty())
        {
            outSet1d->addAttribute("SPECIES", desc.c_str());
        }
        READER_CONTROL->setAssocPortObj(portTok1d, outSet1d);
    }

    binType_ = oldBinType_;
    sendInfo(" reading data finished");
    return Success;
}

int
ReadEnsight::readData2d(const int &portTok2d,
                        const string &fileNameBase,
                        const bool &pV, const int &dim, const string &desc)
{
    int rNumTs;
    vector<string> allFiles(mkFileNames(fileNameBase, rNumTs));
    int totNumTs(allFiles.size());

    if (totNumTs == 0)
        allFiles.push_back(fileNameBase);

    // we want to have a set only if we have transient data
    if (rNumTs > 0)
        extendArrays(rNumTs);

    // set object names and create object arrays for timesteps
    string objNameBase2d = READER_CONTROL->getAssocObjName(portTok2d);
    coDistributedObject **objects2d = new coDistributedObject *[rNumTs + 1];

    vector<string>::iterator ii;

    int cnt(0);
    for (ii = allFiles.begin(); ii != allFiles.end(); ii++)
    {
        char ch[64];
        sprintf(ch, "%d", cnt);
        string num(ch);
        string actObjNm2d(objNameBase2d);
        if (rNumTs > 1)
            actObjNm2d = objNameBase2d + "_" + num;

        char buf[256];
        switch (dim)
        {
        case 1:
            sprintf(buf, "ReadEnsight start reading scalar data out of file %s", (*ii).c_str());
            break;
        case 3:
            sprintf(buf, "ReadEnsight start reading vector data out of file %s", (*ii).c_str());
            break;
        }
        coModule::sendInfo("%s", buf);

        // really read the data
        DataCont ddc;
        EnFile *dFile(NULL);
        if (pV) // per vertex
        {
            dFile = createDataFilePtr(*ii, dim, numCoords_[cnt]);
            if (!dFile->isOpen())
            {
                sendError(" could not open file %s", (*ii).c_str());
                return 0;
            }
            dFile->setPartList(&globalParts_[cnt]);
            dFile->setMasterPL(masterPL_);
            dFile->read(this, EnFile::EnFile::DIM2D, objects2d, actObjNm2d, cnt);
            ddc = dFile->getDataCont();
            delete dFile;
           /* // create DO's
            coDistributedObject **oOut = createDataOutObj(EnFile::DIM2D,actObjNm2d, ddc, cnt);

            ddc.cleanAll();

            if (oOut[0] != NULL)
                objects2d[cnt] = oOut[0];

            ++cnt;*/
        }
        else // per cell
        {
            // memory is allocated now in build parts due to the information collected
            // in the part list
            dFile = createDataFilePtr(*ii, dim, 0);
            //	    dFile = createDataFilePtr( *ii, fileNameBase, dim, 0 );
            if (!dFile->isOpen())
            {
                sendError(" could not open file %s", fileNameBase.c_str());
                return 0;
            }
            dFile->setPartList(&globalParts_[cnt]);
            dFile->setMasterPL(masterPL_);
            dFile->readCells(this, EnFile::EnFile::DIM2D, objects2d, actObjNm2d, cnt);
           /* // create DO's
            DataCont ddc;
            coDistributedObject **oOut = createDataOutObj(EnFile::DIM2D,actObjNm2d, ddc, cnt, false);
            delete dFile;

            if (oOut[0] != NULL)
                objects2d[cnt] = oOut[0];

            cnt++;*/
        }
    }

    // we have no timesteps - feed objectsXY[0] to outports
    if (rNumTs <= 1)
    {
        if (!desc.empty())
        {
            objects2d[0]->addAttribute("SPECIES", desc.c_str());
        }
        if (objects2d[0] != NULL)
            READER_CONTROL->setAssocPortObj(portTok2d, objects2d[0]);
        delete[] objects2d;
    }
    else
    {
        TimeSets ts(case_.getAllTimeSets());
        // create warning if more than one timeset is present
        // actually covise can handle only one timeset
        if (ts.size() > 1)
        {
            coModule::sendInfo("number of time-sets greater than one - COVISE can handle only one time-set");
        }
        
        objects2d[cnt] = NULL;
        vector<float> rTimes(ts[0]->getRealTimes());
        if(ts[0]->getRealTimes().size() >= cnt)
        {
            // set attribute - realtime
            char ch[64];
            for (int i = 0; i < cnt; ++i)
            {
                sprintf(ch, "%f", rTimes[i]);
                objects2d[i]->addAttribute("REALTIME", ch);
            }
        }

        coDoSet *outSet2d = new coDoSet(objNameBase2d.c_str(), (coDistributedObject **)objects2d);
        
        char ch[64];
        // set attribute - timesteps
        sprintf(ch, "1 %d", cnt);
        string attr(ch);
        outSet2d->addAttribute("TIMESTEP", attr.c_str());

        // delete !!!!
        int i;
        for (i = 0; i < cnt; ++i)
            delete objects2d[i];
        delete[] objects2d;

        // single geometry / multiple timesteps
        if (cnt == 1)
            geoTimesetIdx_ = -1;

        if (!desc.empty())
        {
            outSet2d->addAttribute("SPECIES", desc.c_str());
        }
        READER_CONTROL->setAssocPortObj(portTok2d, outSet2d);
    }

    sendInfo(" reading data finished");
    return Success;
}

int
ReadEnsight::readData3d(const int &portTok3d,
                        const string &fileNameBase,
                        const bool &pV, const int &dim, const string &desc)
{
    int rNumTs;
    vector<string> allFiles(mkFileNames(fileNameBase, rNumTs));
    int totNumTs(allFiles.size());

    if (totNumTs == 0)
        allFiles.push_back(fileNameBase);

    // we want to have a set only if we have transient data
    if (rNumTs > 0)
        extendArrays(rNumTs);

    // set object names and create object arrays for timesteps
    string objNameBase3d = READER_CONTROL->getAssocObjName(portTok3d);
    coDistributedObject **objects3d = new coDistributedObject *[rNumTs + 1];

    vector<string>::iterator ii;

    int cnt(0);
    for (ii = allFiles.begin(); ii != allFiles.end(); ii++)
    {
        char ch[64];
        sprintf(ch, "%d", cnt);
        string num(ch);
        string actObjNm3d(objNameBase3d);
        if (rNumTs > 1)
            actObjNm3d = objNameBase3d + "_" + num;

        char buf[256];
        switch (dim)
        {
        case 1:
            sprintf(buf, "ReadEnsight start reading scalar data out of file %s", (*ii).c_str());
            break;
        case 3:
            sprintf(buf, "ReadEnsight start reading vector data out of file %s", (*ii).c_str());
            break;
        }
        coModule::sendInfo("%s", buf);

        // really read the data
        DataCont ddc;
        EnFile *dFile(NULL);
        if (pV) // per vertex
        {
            dFile = createDataFilePtr(*ii, dim, numCoords_[cnt]);
            if (!dFile->isOpen())
            {
                sendError(" could not open file %s", (*ii).c_str());
                return 0;
            }
            dFile->setPartList(&globalParts_[cnt]);
            dFile->setMasterPL(masterPL_);
            dFile->read(this, EnFile::EnFile::DIM3D, objects3d, actObjNm3d, cnt);
            ddc = dFile->getDataCont();
            delete dFile;
         /*   // create DO's
            coDistributedObject **oOut = createDataOutObj(EnFile::DIM3D, actObjNm3d, ddc, cnt);

            ddc.cleanAll();

            if (oOut[0] != NULL)
                objects3d[cnt] = oOut[0];

            ++cnt;*/
        }
        else // per cell
        {
            // memory is allocated now in build parts due to the information collected
            // in the part list
            dFile = createDataFilePtr(*ii, dim, 0);
            //	    dFile = createDataFilePtr( *ii, fileNameBase, dim, 0 );
            if (!dFile->isOpen())
            {
                sendError(" could not open file %s", fileNameBase.c_str());
                return 0;
            }
            dFile->setPartList(&globalParts_[cnt]);
            dFile->setMasterPL(masterPL_);
            dFile->readCells(this, EnFile::EnFile::DIM3D, objects3d, actObjNm3d, cnt);
            // create DO's
         /*   DataCont ddc;
            coDistributedObject **oOut = createDataOutObj(EnFile::DIM3D, actObjNm3d, ddc, cnt, false);
            delete dFile;

            if (oOut[0] != NULL)
                objects3d[cnt] = oOut[0];

            cnt++;*/
        }
    }

    // we have no timesteps - feed objectsXY[0] to outports
    if (rNumTs <= 1)
    {
        if (!desc.empty())
        {
            objects3d[0]->addAttribute("SPECIES", desc.c_str());
        }
        if (objects3d[0] != NULL)
            READER_CONTROL->setAssocPortObj(portTok3d, objects3d[0]);
        delete[] objects3d;
    }
    else
    {
        TimeSets ts(case_.getAllTimeSets());
        // create warning if more than one timeset is present
        // actually covise can handle only one timeset
        if (ts.size() > 1)
        {
            coModule::sendInfo("number of time-sets greater than one - COVISE can handle only one time-set");
        }

        vector<float> rTimes(ts[0]->getRealTimes());

        objects3d[cnt] = NULL;

        // set attribute - realtime
        char ch[64];
        for (int i = 0; i < cnt; ++i)
        {
            sprintf(ch, "%f", rTimes[i]);
            objects3d[i]->addAttribute("REALTIME", ch);
        }

        coDoSet *outSet3d = new coDoSet(objNameBase3d.c_str(), (coDistributedObject **)objects3d);

        // set attribute - timesteps
        sprintf(ch, "1 %d", cnt);
        string attr(ch);
        outSet3d->addAttribute("TIMESTEP", attr.c_str());

        // delete !!!!
        int i;
        for (i = 0; i < cnt; ++i)
            delete objects3d[i];
        delete[] objects3d;

        // single geometry / multiple timesteps
        if (cnt == 1)
            geoTimesetIdx_ = -1;

        if (!desc.empty())
        {
            outSet3d->addAttribute("SPECIES", desc.c_str());
        }
        READER_CONTROL->setAssocPortObj(portTok3d, outSet3d);
    }

    sendInfo(" reading data finished");
    return Success;
}

const coDistributedObject *
ReadEnsight::getGeoObject(const int &step, const int &iPart, const int &dimFlag)
{
    // 3d data
    coDistributedObject *geoIn = NULL;
    if (dimFlag == EnFile::EnFile::DIM2D)
        geoIn = geoObjs_[0];
    else if (dimFlag == EnFile::EnFile::DIM3D)
        geoIn = geoObjs_[1];
    coDoSet *geo3d = NULL;
    if (!geoIn->isType("SETELE"))
    {
        cerr << "ReadEnsight::getGeoObject(..) ERROR expect SETELE at toplevel of 3d geo" << endl;
        return NULL;
    }
    geo3d = (coDoSet *)geoIn;
    int nSubSetsGeo = geo3d->getNumElements();

    if (step >= nSubSetsGeo)
    {
        cerr << "ReadEnsight::getGeoObject(..) ERROR timestep not valid" << endl;
        return NULL;
    }

    const coDistributedObject *geoCand = NULL;
    const coDistributedObject *subSetEleGeo = geo3d->getElement(step);
    //    if ( subSetEleGeo->isType("SETELE") )
    //    {
    //       if ( iPart >= ((coDoSet *) subSetEleGeo)->getNumElements() )
    //       {
    //          cerr << "ReadEnsight::getGeoObject(..) ERROR part number not valid" << endl;
    //       }
    //       geoCand = ((coDoSet *)subSetEleGeo)->getElement( iPart );
    //    }

    ///////////////////////////////
    if (subSetEleGeo->isType("SETELE") && (subSetEleGeo->getAttribute("PART") == NULL))
    {
        if (iPart >= ((const coDoSet *)subSetEleGeo)->getNumElements())
        {
            cerr << "ReadEnsight::getGeoObject(..) ERROR part number not valid" << endl;
        }
        geoCand = ((const coDoSet *)subSetEleGeo)->getElement(iPart);
    }

    else if (subSetEleGeo->getAttribute("PART") != NULL)
    { //( subSetEleGeo->isType("POLYGN") ) || ( subSetEleGeo->isType("UNSGRD") ) ) {
        return geo3d->getElement(iPart);
    }

    if (geoCand->getAttribute("PART") != NULL)
    { //( geoCand->isType("POLYGN") ) || ( geoCand->isType("UNSGRD") ) ) {
        return geoCand;
    }
    ///////////////////////////////

    //    else
    //    {
    //       if ( iPart >= ((coDoSet *) geo3d)->getNumElements() )
    //       {
    //          cerr << "ReadEnsight::getGeoObject(..) ERROR part number not valid" << endl;
    //       }
    //       geoCand = ((coDoSet *)geo3d)->getElement( iPart );
    //    }
    //
    //    if ( ( geoCand->isType("POLYGN") ) || ( geoCand->isType("UNSGRD") ) )
    //    {
    //       return geoCand;
    //    }

    else
    {
        cerr << "ReadEnsight::getGeoObject(..) ERROR NO Geometry found" << endl;
    }

    return NULL;
}

int main(int argc, char *argv[])
{
    // define outline of reader
    READER_CONTROL->addFile(CASE_BROWSER, "case_file", "case file", ".", "*.case;*.CASE;*.encas");

    READER_CONTROL->addOutputPort(GEOPORT3D, "geoOut_3D", "UnstructuredGrid", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_3D, "sdata1_3D", "Float", "data1-3d");
    READER_CONTROL->addOutputPort(DPORT2_3D, "sdata2_3D", "Float", "data2-3d");
    READER_CONTROL->addOutputPort(DPORT3_3D, "sdata3_3D", "Float", "data3-3d");
    READER_CONTROL->addOutputPort(DPORT4_3D, "vdata1_3D", "Vec3", "data2-3d");
    READER_CONTROL->addOutputPort(DPORT5_3D, "vdata2_3D", "Vec3", "data2-3d");

    READER_CONTROL->addOutputPort(GEOPORT2D, "geoOut_2D", "Polygons", "Geometry", false);

    READER_CONTROL->addOutputPort(DPORT1_2D, "sdata1_2D", "Float", "data1-2d");
    READER_CONTROL->addOutputPort(DPORT2_2D, "sdata2_2D", "Float", "data2-2d");
    READER_CONTROL->addOutputPort(DPORT3_2D, "sdata3_2D", "Float", "data3-2d");
    READER_CONTROL->addOutputPort(DPORT4_2D, "vdata1_2D", "Vec3", "data1-2d");
    READER_CONTROL->addOutputPort(DPORT5_2D, "vdata2_2D", "Vec3", "data2-2d");

    READER_CONTROL->addOutputPort(GEOPORT1D, "geoOut_1D", "Points", "Measured points", false);

    READER_CONTROL->addOutputPort(DPORT1_1D, "sdata1_1D", "Float", "data1-1d");
    READER_CONTROL->addOutputPort(DPORT2_1D, "sdata2_1D", "Float", "data2-1d");
    READER_CONTROL->addOutputPort(DPORT3_1D, "sdata3_1D", "Float", "data3-1d");
    READER_CONTROL->addOutputPort(DPORT4_1D, "vdata1_1D", "Vec3", "data1-1d");
    READER_CONTROL->addOutputPort(DPORT5_1D, "vdata2_1D", "Vec3", "data2-1d");

    // create the module
    coReader *application = new ReadEnsight(argc, argv);

    // this call leaves with exit(), so we ...
    application->start(argc, argv);

    // ... never reach this point
    return 0;
}
