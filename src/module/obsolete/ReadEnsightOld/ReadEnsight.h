/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ENSIGHT_DATA_H
#define _ENSIGHT_DATA_H
/*****************************************************
Description:	  Data reader for Ensight format
Author:	    	  B.Teplitski
Created:    	  19.09.2000
*****************************************************/

#include <util/coviseCompat.h>

#include <api/coModule.h>
using namespace covise;
//#include "CaseFile.h"

const int FLOAT_LENG = 12; // one float has 12 figures
const int INT_LENG = 8; // one int has 8 figures

class ReadEnsight : public coModule
{

public:
    ReadEnsight(int argc, char *argv[]);
    virtual ~ReadEnsight();
    int isPolygon(char *strPolygonType);
    int isGrid(char *strGridType);
    int isLine(char *strLineType);
    int getNumberOfCorner(char *strType);
    int getType(char *strType);
    int readScalarValuesSequent(char *strFilename, float *fltKindOfData, int intBodiesNumber);
    int readScalarValuesParallel(char *strFilename, float *fltKindOfData, int intBodiesNumber);
    int readVectorValuesParallel(char *strFilename, float *fltXData, float *fltYData, float *fltZData, int intNumberCoords);
    int readVectorValuesSequent(char *strFilename, float *fltXData, float *fltYData, float *fltZData, int intNumberCoords);
    int getCoordNumber(char *strHeader);
    int read3DCoordsParallel(ifstream &iFile, int intCoordsNumber, float *fltXCoords, float *fltYCoords, float *fltZCoords);
    int read3DCoordsSequent(ifstream &iFile, int intCoordsNumber, float *fltXCoords, float *fltYCoords, float *fltZCoords);
    int lookForHeader(ifstream &ifFile, char *strHeader);
    int getNumberOfElements(ifstream &iFile, int &intPolygonNumber, int &intGridNumber, int &intPolygonCornerNumber, int &intGridCornerNumber);
    int fillCornerArrays(ifstream &iFile, int *intGridList, int *intGridCornerList, int *intTypeList, int *intPolygonList, int *intPolygonCornerList);
    void sendInfoSuccess(const char *strObject);
    int readResultFile(const char *strResultFileName);
    int readVarFiles(ifstream &iFile, int intNumberOfValues, int intNumberOfTimeSteps, int isScalar);
    char *getEnsightVersion(ifstream &iFile);
    int isThereNodeIDSection(ifstream &iFile);
    int isThereElementIDSection(ifstream &iFile);
    int getNumberOfTimeSteps(ifstream &iFile);
    void initFileMap(void);
    void fillFileMap(int intLineNumber, int intCornersNumber, int intNumberOfLines, int intType);
    void showFileMap(void);

    void initGridMap(void);
    void fillGridMap(int intNumberOfElements);
    void showGridMap(void);

    void initPolygonMap(void);
    void fillPolygonMap(int intNumberOfElements);
    void showPolygonMap(void);
    char *buildTimestepFileName(char *strFileName, int intNumberOfTimeSteps, int intStep);
    int getPerNodeOrPerElement(ifstream &iFile);

private:
    int readScalarValuesV6(char *strFilename, float *fltKindOfData, int intNumberOfElements);
    int readVectorValuesV6(char *strFilename, float *fltXData, float *fltYData, float *fltZData, int intNumberElements);
    //    CaseFile *caseF_;

    virtual int compute();
    int computeGold();
    int compute5();
    virtual void postInst();

    int LINE_ELEMENT_LENGTH;
    int CORNER_LINE_ELEMENT_LENGTH;
    int NUMBER_VALUES_IN_LINE;
    char ENSIGHT_VERSION[20];

    int nodeIdFlg_;
    int elementIdFlg_;
    int caseFileFlg_;

    static const int PER_NODE = 1;
    static const int PER_ELEMENT = 2;

    coOutputPort *polygonOutPort;
    coOutputPort *gridOutPort;
    coOutputPort *polygonScalarOutPort;
    coOutputPort *polygonVectorOutPort;
    coOutputPort *gridScalarOutPort;
    coOutputPort *gridVectorOutPort;

    coDoPolygons *polygonObj;
    coDoUnstructuredGrid *gridObj;
    coDoFloat *polygonScalarObj;
    coDoVec3 *polygonVectorObj;
    coDoFloat *gridScalarObj;
    coDoVec3 *gridVectorObj;
    coDistributedObject **setObject;

    coFileBrowserParam *geoFileParam;
    coFileBrowserParam *caseFileParam;
    coFileBrowserParam *paramPolygonScalarFile;
    coFileBrowserParam *paramPolygonVectorFile;
    coFileBrowserParam *paramGridScalarFile;
    coFileBrowserParam *paramGridVectorFile;
    coFileBrowserParam *resFileParam;
    coIntScalarParam *paramLenghOfLineElement;
    coIntScalarParam *paramLenghOfCornerLineElement;
    coIntScalarParam *paramNumberValuesInLine;

    int intLineCounter; //line number in the file
    int intAncor;

    int intPolygonNumber; //number of all polygons
    int intGridNumber;

    int intPolygonCoordsNumber;
    int intGridCoordsNumber;

    int intPolygonCornerNumber;
    int intGridCornerNumber;

    int intElementCornerNumber;

    int intNumberOfTimeSteps;

    int *intPolygonList;
    int *intGridList;
    int *intTypeList;

    int *intElementList;
    int *intElementTypeList;
    int *intElementCornerList;

    int *intPolygonCornerList;
    int *intGridCornerList;

    float *fltGridXCoords;
    float *fltGridYCoords;
    float *fltGridZCoords;

    int maxIndex_;
    int *index_;
    float *fltXCoords;
    float *fltYCoords;
    float *fltZCoords;

    float *fltXData;
    float *fltYData;
    float *fltZData;
    float *fltKindOfGridScalarData;
    float *fltData;

    char strEnsightVersion[50];
    int intNodeIdSectionExist;
    int intElementIDSectionExist;

    int intCoordsNumber;
    int intElementNumber;

    int intElementCount;
    int intElementCornerCount;

    int intFileMap[100][4];
    int intPolygonMap[100];
    int intGridMap[100];

    int intFileMapIndex;
    int intPolygonMapIndex;
    int intGridMapIndex;
    ifstream ifstrFile;
    int intKindOfMeasuring;

    int *translList_;
    int maxTranslL_;
    //some element are duplicates because of merging different programs
};
#endif
