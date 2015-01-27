/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/** @file MeshFileTransParserHmAscii.h
 * a mesh file parser for ASCII files.
 */

// #include "MeshFileTransParserHmAscii.h"  // a mesh file parser for changing (transient) meshes in HyperMesh ASCII format.

#ifndef __MeshFileTransParserHmAscii_h__
#define __MeshFileTransParserHmAscii_h__

#include "OutputHandler.h" // an output handler for displaying information on the screen.
#include "MeshFileTransParser.h" // a mesh file parser.
#include "errorinfo.h" // a container for error data.
#include <api/coModule.h>
using namespace covise;
#include <vector>

#define LINE_SIZE 1024

/**
 * a mesh file parser for changing (transient) meshes in HyperMesh ASCII format.
 */
class MeshFileTransParserHmAscii : public MeshFileTransParser, public MeshDataTrans, public MeshDataStat
{
public:
    MeshFileTransParserHmAscii(OutputHandler *outputHandler);
    virtual ~MeshFileTransParserHmAscii(){};

    virtual void parseMeshFile(std::string filename,
                               coRestraint &sel,
                               bool subdivide,
                               const int &noOfTimeStepToSkip,
                               const int &noOfTimeStepsToParse);
    virtual int getNoOfMeshes(void) const
    {
        return 1;
    };
    virtual MeshDataStat *getMeshDataStat(int /*timeStepNo*/)
    {
        return this;
    }

    virtual int getNoOfElements(void) const
    {
        return _noOfElements;
    };
    virtual int getNoOfPoints(void) const
    {
        return _noOfPoints;
    };
    virtual int getMaxNoOfPoints(void) const
    {
        return _noOfPoints;
    };

    virtual MeshDataTrans *getMeshDataTrans(void)
    {
        return this;
    };
    virtual void addMesh(MeshDataStat *)
    {
        ERROR0("illegal function call", _outputHandler);
    };

    virtual void getMeshData(int *noOfElements, int *noOfVertices, int *noOfPoints,
                             int *elementsArr[], int *verticesArr[],
                             float *xPointsArr[], float *yPointsArr[], float *zPointsArr[],
                             int *typesArr[]);

    virtual int getMaxNodeNo(void) const;
    virtual int getMaxNodeNo(int) const
    {
        return getMaxNodeNo();
    };
    virtual int getInternalNodeNo(int) const;

private:
    OutputHandler *_outputHandler;

    int _noOfPoints;
    std::vector<float> _xPoints;
    std::vector<float> _yPoints;
    std::vector<float> _zPoints;

    // elements and vertices (=node numbers of elements)
    int _noOfVertices;
    std::vector<int> _vertices;

    int _noOfElements;
    std::vector<int> _elements;

    std::vector<int> _types;

    // -------- stuff needed during parsing ---------
    FILE *_fileHandle;
    void getLine(void);
    void pushBack(void); // push back this _line (don´t read a new on next time)

    bool _pushedBack;
    char _line[LINE_SIZE];
    char *_c;

    void debug(void) const;
};

#endif
