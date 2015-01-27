/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef READ_VTF_DATA_H
#define READ_VTF_DATA_H

/**************************************************************************\ 
 **                                                           (C)2001    **
 ** Author:                                                              **
 **                            Karin Mller                              **
 **                                        Vircinity                     **
 **                            Technologiezentrum                        **
 **                            70550 Stuttgart                           **
 ** Date:  01.10.01                                                      **
\**************************************************************************/

#include <api/coModule.h>
using namespace covise;

#define NUMPORTS 4

// structs for NODES
// 3D coordinates

struct NodeHeader
{
    int iHeaderSize;
    int iDataSize;
    int iWithID;
    int iNumNodes;
};

struct Point
{
    int ID;
    float x;
    float y;
    float z;
    int iPosInCornerList;
    int iPosInPointList;
};

struct Node
{
    int ID;
    NodeHeader header;
    Point *point;
};

// structs for INDEXEDFACESET
// block with an indexed face set

struct IFSHeader
{
    int iHeaderSize;
    int iDataSize;
    int iNodeBlockID;
    //char szDescription[80]; we don't need this
    float fRColor;
    float fGColor;
    float fBColor;
    int iWithID;
    int iNumPolygons; // number of polygons
    int iNumConnects; // number of edges
};

typedef struct
{
    int ID;
    int num_edges;
    int *edgeIndex; // last edge in polygon must be negative
} MyPolygon;

struct IndexedFaceSet
{
    int ID;
    IFSHeader header;
    MyPolygon *poly;
};

// structs for ELEMENT
// block with element specification

struct ElementHeader
{
    int iHeaderSize;
    int iDataSize;
    int iNodeBlockID;
    //char szDescription[80]; we don't need this
    float fRColor;
    float fGColor;
    float fBColor;
    int iWithID;
    int iNumElementTypes; // number of elements types in block
};

struct ElementDataHeader
{
    int eType; // new type of Covise (eg TYPE_POINT...)
    int iNumElements; // number of elements of the same type
};

struct ElementElement
{
    int UserID;
    int iNumNodes;
    int *iNodeID;
    int iPosInPolyList;
};

struct ElementData
{
    ElementDataHeader dataHeader;
    ElementElement *tElement;
};

struct Element
{
    int ID;
    ElementHeader header;
    ElementData *data;
};

// structs for GLVIEWGEOMETRY
// block connecting element blocks into geometries

struct GeometryHeader
{
    int iHeaderSize;
    int iDataSize;
    char szDescription[80]; // we don't need this
    int iNumSteps; // number of time steps defined
};

struct GeometryDataHeader
{
    int iStepNumber;
    char szStepName[80]; // we don't need this
    float fStepTime;
    int iNumElementBlocks;
    int iNumIFSBlocks;
    int iControl1; // must be -1
    int iControl2; // must be -1
};

struct GeometryData
{
    int *iElemBlockID;
    int *iIFSBlock;
};

struct GLViewGeometry
{
    int ID;
    GeometryHeader header;
    GeometryDataHeader *dataHeader;
    GeometryData *data;
};

// structs for RESULTS
// block witz 1D or 3D results

struct ResultHeader
{
    int iHeaderSize;
    int iDataSize;
    int iDimension;
    int iMapToBlockID;
    int iMappingType; // 0: results per node, 1: per element, 2: per face (IFS)
    int iWithID;
    int iNumResults;
};

struct ResultData
{
    int iUserID;
    float fScalarValue;
    float fVectorValue[3]; // x, y, z
};

struct Results
{
    int ID;
    ResultHeader header;
    ResultData *data;
};

// structs for GLVIEWSCALAR
// a definition of a GLview scalar result

struct GLViewScalarHeader
{
    int iHeaderSize;
    int iDataSize;
    // char szDescription[80]; we don't need this
    int iNumSteps;
    int iResultID;
    int iSectionID;
    int iWithStateID;
};

struct GLViewScalarDataHeader
{
    int iStepNumber;
    // char szStepName[80]; we don't need this
    float fStepTime;
    int iNumResultsBlocks;
};

struct GLViewScalarData
{
    int *iResultBlockID;
};

struct GLViewScalar
{
    int ID;
    GLViewScalarHeader header;
    GLViewScalarDataHeader *dataHeader;
    GLViewScalarData *data;
};

// structs for GLVIEWVECTOR
// a definition of a GLview vector result

struct GLViewVectorHeader
{
    int iHeaderSize;
    int iDataSize;
    // char szDescription[80]; we don't need this
    int iNumSteps;
};

struct GLViewVectorDataHeader
{
    int iStepNumber;
    // char szStepName[80]; we don't need this
    float fStepTime;
    int iNumResultsBlocks;
};

struct GLViewVectorData
{
    int *iResultBlockID;
};

struct GLViewVector
{
    int ID;
    GLViewVectorHeader header;
    GLViewVectorDataHeader *dataHeader;
    GLViewVectorData *data;
};

// structs for TRANSFORMATIONS
// a set of transformations matrices

struct TransformationHeader
{
    int iHeaderSize;
    int iDataSize;
    // char szDescription[80]; we don't need this
    int iWithID;
    int iNumSteps;
};

struct TransformationDataHeader
{
    int iStepNumber;
    // char szStepName[80]; we don't need this
    float fStepTime;
    int iNumElementBlocks;
    int iNumIFSBlocks;
};

struct TransformationData
{
    int *iElementBlockID;
    float **iElementTransMatrix;
    int *iIFSID;
    float **iIFSTransMatrix;
};

struct Transformation
{
    int ID;
    TransformationHeader header;
    TransformationDataHeader *dataHeader;
    TransformationData *data;
};

// structs for VIEWPOINTS
// a set of viewpoints
// not used!

// structs for 2DPLOTSERIES
// a 2D plot series
// not used!

// structs for USER
// a user defined block, not used by GLview
// not used!

// structs for POSITIONRESULTS
// a block of results with position space

struct PositionResultsHeader
{
    int iHeaderSize;
    int iDataSize;
    int iDimension;
    int iMapToBlockType; // 1: Element block
    int iMapToBlockID;
    int iMappingType; // 0: no mapping info, 1: map to item IDs,2: map to item indices
    int iGlobalPositions;
    int iGlobalResults;
    int iNumResults;
};

struct PositionResultsData
{
    int iMap;
    float fPosition[3]; // Position x(y,Z) component (lokal or global)
    float fScalarValue;
    float fVectorValue[3]; // x, y, z
};

struct PositionResults
{
    int ID;
    PositionResultsHeader header;
    PositionResultsData data[100];
};

// structs for GLVIEWPOSITIONSCALAR
// a definition of a sequence of scalars with pos

struct GLViewPositionScalarHeader
{
    int iHeaderSize;
    int iDataSize;
    // char szDescription[80]; we don't need this
    int iNumSteps;
};

struct GLViewPositionScalarDataHeader
{
    int iStepNumber;
    // char szStepName[80]; we don't need this
    float fStepTime;
    int iNumResultsBlocks;
};

struct GLViewPositionScalarData
{
    int iPosResultBlockID[100];
};

struct GLViewPositionScalar
{
    int ID;
    GLViewPositionScalarHeader header;
    GLViewPositionScalarDataHeader dataHeader[100];
    GLViewPositionScalarData data[100];
};

// structs for GLVIEWPOSITIONVECTOR
// a definition of a GLview separate vector

struct GLViewPositionVectorHeader
{
    int iHeaderSize;
    int iDataSize;
    // char szDescription[80]; we don't need this
    int iNumSteps;
};

struct GLViewPositionVectorDataHeader
{
    int iStepNumber;
    // char szStepName[80]; we don't need this
    float fStepTime;
    int iNumResultsBlocks;
};

struct GLViewPositionVectorData
{
    int iPosResultBlockID[100];
};

struct GLViewPositionVector
{
    int ID;
    GLViewPositionVectorHeader header;
    GLViewPositionVectorDataHeader dataHeader[100];
    GLViewPositionVectorData data[100];
};

// structs for TRANSFORMATIONRESULT
// a transformation matrix

struct TransformationResultHeader
{
    int iHeaderSize;
    int iDataSize;
    int iIFSBlockID;
    int iElementBlockID;
};

struct TransformationResult
{
    int ID;
    TransformationResultHeader header;
    float pfTransformationMatrix[12];
};

// structs for GLVIEWTRANSFORMATION
// a definition of a GLview transformation result

struct GLViewTransformationHeader
{
    int iHeaderSize;
    int iDataSize;
    // char szDescription[80]; we don't need this
    int iNumSteps;
    int iWithStateID;
};

struct GLViewTransformationDataHeader
{
    int iStepNumber;
    // char szStepName[80]; we don't need this
    float fStepTime;
    int iNumTransBlocks;
    int iStateID;
};

struct GLViewTransformationData
{
    int iTransResultsBlockID[100];
};

struct GLViewTransformation
{
    int ID;
    GLViewTransformationHeader header;
    GLViewTransformationDataHeader dataHeader[100];
    GLViewTransformationData data[100];
};

//==============================================================================

class ReadVTFData
{
public:
    ReadVTFData(FILE *d_file);
    void setValues(); // initialize all values
    int getBlockType(FILE *file);
    bool readHeader(FILE *file); // TRUE, if Header=*VTF - 1.00 (binary: 231272 -160871 251271)
    void readGLViewScalarHeader(FILE *pt_file);
    void readGLViewVectorHeader(FILE *pt_file);
    void reset()
    {
        fpos = 0;
    };

    bool eof()
    {
        return (fpos >= file_size);
    };

    void seek(int num)
    {

        fpos += num;
        if (eof())
        {
            if (fpos > file_size)
                cout << "Error" << endl;
            fpos = file_size;
        }
    };

    // **** funktions for reading types and data
    void readNextIntElement(FILE *file); // read the next integer and fill m_iElem
    void readNextFloatElement(FILE *file); // read the next float and fill m_fElem
    void readNextCharElement(FILE *file); // read char, we don't need this
    int getBlockID(FILE *file);
    int readNodes(FILE *file);
    void readIndexedFaceSet(FILE *file);
    int readElement(FILE *file, bool is_poly = false);
    void readGLViewGeometry(FILE *file);
    int readResult(FILE *pt_file);
    void readGLViewScalar(FILE *pt_file);
    void readGLViewVector(FILE *pt_file);
    void readTransformations(FILE *pt_file);
    void readViewPoints(FILE *pt_file);
    void readPloteries(FILE *pt_file);
    void readUser(FILE *pt_file);
    void readPositionResults(FILE *pt_file);
    void readPositionScalar(FILE *pt_file);
    void readGLViewPositionVector(FILE *pt_file);
    void readTransformationResult(FILE *pt_file);
    void readGLViewTransformation(FILE *pt_file);

    // **** helping funktions for reading types and data
    int getNumNodesInElem(int eType); // for readElement();
    int getNumNodesInElem(int iDataSize, int iNumElements, int iWith_ID);
    int getCoviseType(int eType); // switch VTFType into CoviseType
    int getNumCoviseNodesInElem(int eType); // eg: TYPE_BAR = 1 node, ...
    void setPosInCornerList(int iNodeID, int iPointID, int iNumCorners);
    // of a point in a NODE
    int getPosInCornerList(int iNodeID, int iPointID);
    void setPosInPointList(int iNodeID, int iPointID, int iNumPoints);
    // of a point in a NODE
    int getPosInPointList(int iNodeID, long int iPointID);
    // of an element in ELEMENTS
    int getPosInElemList(int iElemID, int iUserID);
    // of an element in polygon_list
    int getPosInPolyList(int iElemBlockID, int iUserID);

    // functions to get counters
    int getElemCounter();
    int getNodesCounter();
    int getNumPoints();
    int getResCounter();
    int getS3DCounter();
    int getV3DCounter();
    int getVSCounter();
    int getVVCounter();
    int getGeomCounter();

    void deleteAll();

    // **** member variables
    // NODES
    Node *m_tNodes; // all NODES blocks are in this array
    int m_iNumNodes;
    // IFS
    IndexedFaceSet m_tIFS[100]; // all IFS blocks are in this array
    // ELEMENT
    Element *m_tElem;
    int m_iNumElem;
    // GLVIEWGEOMETRY
    GLViewGeometry m_tGeom[500];
    // RESULTS
    Results *m_tRes;
    int m_iNumRes;
    // GLVIEWSCALAR
    GLViewScalar m_tViewS[500];
    int m_iVSCounter; // GLVIEWSCALAR

    // GLVIEWVECTOR
    GLViewVector m_tViewV[50];
    int m_iVVCounter; // GLVIEWVECTOR
    //TRANSFORMATION
    Transformation m_tTrans[100];
    /*
      //POSITIONRESULTS
      PositionResults m_tPosRes[100];
      // GLVIEWPOSITIONSCALAR
      GLViewPositionScalar m_tPosS[100];
      //GLVIEWPOSITIONVECTOR
      GLViewPositionVector m_tPosV[100];
      */
    //TRANSFORMATIONRESULT
    TransformationResult m_tTransRes[100];
    //GLVIEWTRANSFORMATION
    GLViewTransformation m_tViewTrans[100];

    // member variables for description
    char *m_cDescription[20]; // description of scalar an vector data
    char m_cCopy[20][80]; // copy of m_cDescription
    int m_iNumDes; // number of descriptions

    int m_iPort[NUMPORTS]; // where are scalar=0 or vector=1 data?
    bool m_bResults; // TRUE if there are GLViewScalar or GLViewVector results
    enum
    {
        IFS = -1,
        MAX_POLY_CONN = 100
    };
    int m_iTransCounter; // TRANSFORMATION
    int m_iTransResCounter; // TRANSFORMATIONRESULT
    int m_iViewTransCounter; // GLVIEWTRANSFORMATION

private:
    char *file_buf;
    bool buf_read;
    long fpos, file_size;

    int m_iElem; // one integer of the file
    float m_fElem; // one float of the file
    char m_cElem[80]; // one char of the file
    int m_iNumScalar; // number of blocks of scalar data
    int m_iNumVector; // number of blocks of vector data
    bool m_bByteswap; // TRUE if file is byteswaped, else FALSE
    // counter for all types
    int m_iNodesCounter; // how many NODES blocks are in the array
    int m_iElemCounter; // ELEMENT
    int m_iGeomCounter; // GLVIEWGEOMETRY
    int m_iResCounter; // RESULTS
    int m_iS3DCounter;
    int m_iV3DCounter;

    int m_iIFSCounter; // how many IFS blocks are in the array
    int m_iPosResCounter; // POSITIONRESULTS
    int m_iPosSCounter; // GLVIEWPOSITIONSCALAR
    int m_iPosVCounter; // GLVIEWPOSITIONVECTOR
};
#endif // READ_VTF_DATA_H
