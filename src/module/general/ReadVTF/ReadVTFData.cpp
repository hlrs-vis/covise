/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/**************************************************************************\ 
 **                                                           (C)2001    **
 ** Author:                                                              **
 **                            Karin Mueller                             **
 **                                        Vircinity                     **
 **                            Technologiezentrum                        **
 **                            70550 Stuttgart                           **
 ** Date:  01.10.01                                                      **
\**************************************************************************/

#include "ReadVTFData.h"
#include "Binary.h"

#include <sys/types.h>
#include <sys/stat.h>

#include <util/coviseCompat.h>
#include <do/coDoUnstructuredGrid.h>

bool firstError;
bool ignoreIDs;

Binary tBin;
/* // only for testing
   ofstream DataInput ("/vobs/covise/src/application/general/READ_VTF/input.data");
*/
ReadVTFData::ReadVTFData(FILE *d_file)
{
    buf_read = false;
    file_buf = NULL;
    fpos = 0;

    firstError = true;
    ignoreIDs = false;

    struct stat statRec;

    if (d_file == NULL || 0 != fstat(fileno(d_file), &statRec))
    {
        return;
    }

    file_buf = new char[statRec.st_size];

    if (!file_buf)
    {
        cerr << "Not enough memory" << endl;
        return;
    }

    file_size = statRec.st_size;

    if (fread(file_buf, statRec.st_size, 1, d_file) != 1)
    {
        cerr << "ReadVTFData::ReadVTFData: fread failed" << endl;
    }

    fclose(d_file);

    buf_read = true;
    ;
}

//==============================================================================

void ReadVTFData::setValues()
{
    // initialize all member variables
    m_bByteswap = false;
    m_iElem = 0;
    m_fElem = 0.0;
    m_iNumScalar = 0;
    m_iNumVector = 0;
    m_iNumDes = 0;
    m_iNodesCounter = 0;
    m_iIFSCounter = 0;
    m_iElemCounter = 0;
    m_iGeomCounter = 0;
    m_iResCounter = 0;
    m_iS3DCounter = 0;
    m_iV3DCounter = 0;
    m_iVSCounter = 0;
    m_iVVCounter = 0;
    m_iTransCounter = 0;
    m_iTransResCounter = 0;
    m_iPosResCounter = 0;
    m_iPosSCounter = 0;
    m_iPosVCounter = 0;
    m_iViewTransCounter = 0;
}

//==============================================================================

int ReadVTFData::getElemCounter()
{
    return m_iElemCounter;
}

int ReadVTFData::getNodesCounter()
{
    return m_iNodesCounter;
}

int ReadVTFData::getResCounter()
{
    return m_iResCounter;
}

int ReadVTFData::getS3DCounter()
{
    return m_iS3DCounter;
}

int ReadVTFData::getV3DCounter()
{
    return m_iV3DCounter;
}

int ReadVTFData::getVSCounter()
{
    return m_iVSCounter;
}

int ReadVTFData::getVVCounter()
{
    return m_iVVCounter;
}

int ReadVTFData::getGeomCounter()
{
    return m_iGeomCounter;
}

int ReadVTFData::getNumPoints()
{
    int i;
    int sum = 0;

    for (i = 0; i < m_iNodesCounter; i++)
    {
        sum += m_tNodes[i].header.iNumNodes;
    }

    return sum;
}

//==============================================================================

int ReadVTFData::getBlockType(FILE *pt_file)
{
    readNextIntElement(pt_file);
    return m_iElem;
}

//==============================================================================

void ReadVTFData::readNextIntElement(FILE *pt_file)
{
    (void)pt_file;
    //fread( &m_iElem, sizeof(int), 1, pt_file); // 4 bytes

    int *i = (int *)(file_buf + fpos);

    m_iElem = *i;

    this->seek(sizeof(int));

    if (m_bByteswap)
        tBin.byteswap(m_iElem);
}

//==============================================================================

void ReadVTFData::readNextFloatElement(FILE *pt_file)
{
    (void)pt_file;
    //fread( &m_fElem, sizeof(float), 1, pt_file); // 4 bytes

    float *f = (float *)(file_buf + fpos);
    m_fElem = *f;

    this->seek(sizeof(float));

    if (m_bByteswap)
        tBin.byteswap(m_fElem);
}

//==============================================================================

void ReadVTFData::readNextCharElement(FILE *pt_file)
{
    (void)pt_file;
    //fread( m_cElem, sizeof(char), 80, pt_file); // 80 bytes
    strncpy(m_cElem, file_buf + fpos, 80);
    this->seek(80);
}

//==============================================================================

bool ReadVTFData::readHeader(FILE *pt_file)
{
    bool bReturn = false;
    int iElem[4];

#ifdef BYTESWAP
    m_bByteswap = 1;
#else
    m_bByteswap = 0;
#endif
    // initialize iElem

    for (int i = 0; i < 4; i++)
        iElem[i] = 0;

    if (tBin.isBinary(pt_file))
    {
        // read the first 4 bytes

        if (fread(iElem, sizeof(int), 4, pt_file) != 4)
        {
            cerr << "ReadVTFData::readHeader: fread failed" << endl;
        }

        this->seek(4 * sizeof(int));

        /*
        File Header
        Byte start      Length  Format  Description
        0                       4               I4              Magic number 1: 231272
        4                       4               I4              Magic number 2: -160871
        8                       4               I4              Magic number 3: 251271
        12                      4               I4              File Version. Current version: 1
      */

        // compare with Magic numbers, which define the header

        if (iElem[0] == 231272 && iElem[1] == -160871 && iElem[2] == -251271)
        {
            bReturn = true;
            m_bByteswap = false;
        }
        else
        {
            tBin.byteswap(iElem, 4);

            if (iElem[0] == 231272 && iElem[1] == -160871 && iElem[2] == 251271)
            {
                m_bByteswap = !m_bByteswap;
                bReturn = true;
            }
            else
                bReturn = false;
        }
    }
    else
        Covise::sendInfo("file is not binary ");

    return bReturn;
}

//==============================================================================

void ReadVTFData::readGLViewScalarHeader(FILE *pt_file)
{
    Covise::sendInfo("read header of GLViewScalar");
    m_iNumScalar++;

    readNextIntElement(pt_file); // Block ID
    int iHeaderSize = getBlockType(pt_file); // HeaderSize
    int iDataSize = getBlockType(pt_file); // DataSize
    readNextCharElement(pt_file); // info!! must be saved
    sprintf(m_cCopy[m_iNumDes], "%s ", m_cElem);
    m_cDescription[m_iNumDes] = m_cCopy[m_iNumDes];
    Covise::sendInfo("%s", m_cDescription[m_iNumDes]);
    // saving type of block, scalar = 0
    m_iPort[m_iNumDes] = 0;
    m_iNumDes++;
    //fseek(pt_file, iHeaderSize+iDataSize-84, SEEK_CUR);
    fpos += iHeaderSize + iDataSize - 84;
    m_bResults = true;
}

//==============================================================================

void ReadVTFData::readGLViewVectorHeader(FILE *pt_file)
{
    Covise::sendInfo("read header of GLViewVector");
    m_iNumVector++;

    readNextIntElement(pt_file); // Block ID
    int iHeaderSize = getBlockType(pt_file); // HeaderSize
    int iDataSize = getBlockType(pt_file); // DataSize
    readNextCharElement(pt_file); // info! must be saved
    sprintf(m_cCopy[m_iNumDes], "%s ", m_cElem);
    m_cDescription[m_iNumDes] = m_cCopy[m_iNumDes];
    Covise::sendInfo("%s", m_cDescription[m_iNumDes]);
    // saving type of block, vector = 1
    m_iPort[m_iNumDes] = 1;
    m_iNumDes++;
    //fseek(pt_file, iHeaderSize+iDataSize-84, SEEK_CUR);
    fpos += iHeaderSize + iDataSize - 84;
    m_bResults = true;
}

//==============================================================================

int ReadVTFData::readNodes(FILE *pt_file)
{
    // fill struct Node
    readNextIntElement(pt_file);
    m_tNodes[m_iNodesCounter].ID = m_iElem;
    readNextIntElement(pt_file); // iHeaderSize, always 16
    m_tNodes[m_iNodesCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file); // size of following data section
    m_tNodes[m_iNodesCounter].header.iDataSize = m_iElem;
    readNextIntElement(pt_file); // 1 if nodes have user defined IDs, else 0
    m_tNodes[m_iNodesCounter].header.iWithID = m_iElem;
    readNextIntElement(pt_file); // number of nodes in block
    m_tNodes[m_iNodesCounter].header.iNumNodes = m_iElem;

    m_tNodes[m_iNodesCounter].point = new Point[m_tNodes[m_iNodesCounter].header.iNumNodes + 1];

    if (m_tNodes[m_iNodesCounter].header.iWithID)
    {
        for (int j = 0; j < m_tNodes[m_iNodesCounter].header.iNumNodes; j++)
        {
            readNextIntElement(pt_file); // user ID
            m_tNodes[m_iNodesCounter].point[j].ID = m_iElem;
            readNextFloatElement(pt_file); // x Coord
            m_tNodes[m_iNodesCounter].point[j].x = m_fElem;
            readNextFloatElement(pt_file); // Y Coord
            m_tNodes[m_iNodesCounter].point[j].y = m_fElem;
            readNextFloatElement(pt_file); // Z Coord
            m_tNodes[m_iNodesCounter].point[j].z = m_fElem;
            m_tNodes[m_iNodesCounter].point[j].iPosInCornerList = -1;
            m_tNodes[m_iNodesCounter].point[j].iPosInPointList = -1;
        }
    }
    else
    {
        for (int j = 0; j < m_tNodes[m_iNodesCounter].header.iNumNodes; j++)
        {
            readNextFloatElement(pt_file); // x Coord
            m_tNodes[m_iNodesCounter].point[j].x = m_fElem;
            readNextFloatElement(pt_file); // Y Coord
            m_tNodes[m_iNodesCounter].point[j].y = m_fElem;
            readNextFloatElement(pt_file); // Z Coord
            m_tNodes[m_iNodesCounter].point[j].z = m_fElem;
            m_tNodes[m_iNodesCounter].point[j].iPosInCornerList = -1;
            m_tNodes[m_iNodesCounter].point[j].iPosInPointList = -1;
        }
    }

    m_iNodesCounter++; // now there is one more NODES block in the m_tNodes array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
    {
        Covise::sendError("Error reading NODES");
        return 1;
    }

    return 0;
}

//==============================================================================

void ReadVTFData::readIndexedFaceSet(FILE *pt_file)
{
    // Block ID
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].ID = m_iElem;

    // fill struct IFSHeader
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].header.iDataSize = m_iElem;
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].header.iNodeBlockID = m_iElem;
    readNextCharElement(pt_file); // we don't need this
    readNextFloatElement(pt_file);
    m_tIFS[m_iIFSCounter].header.fRColor = m_fElem;
    readNextFloatElement(pt_file);
    m_tIFS[m_iIFSCounter].header.fGColor = m_fElem;
    readNextFloatElement(pt_file);
    m_tIFS[m_iIFSCounter].header.fBColor = m_fElem;
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].header.iWithID = m_iElem;
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].header.iNumPolygons = m_iElem;
    readNextIntElement(pt_file);
    m_tIFS[m_iIFSCounter].header.iNumConnects = m_iElem;

    // fill struct Polygon

    for (int i = 0; i < m_tIFS[m_iIFSCounter].header.iNumPolygons; i++)
        for (int j = 0; j < m_tIFS[m_iIFSCounter].header.iNumConnects; j++)
            if (m_tIFS[m_iIFSCounter].header.iWithID)
            {
                readNextIntElement(pt_file);
                m_tIFS[m_iIFSCounter].poly[i].ID = m_iElem;
                readNextIntElement(pt_file);
                m_tIFS[m_iIFSCounter].poly[i].edgeIndex[j] = m_iElem;
            }
            else
            {
                readNextIntElement(pt_file);
                m_tIFS[m_iIFSCounter].poly[i].edgeIndex[j] = m_iElem;
            }

    m_iIFSCounter += 1; // now there is one more IFS in the array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading IFS");

    // not used
    /* das hier war das berlesen des IFS, als oben noch auskommentiert war. das darf dann nicht mehr
      gemacht werden.
      readNextIntElement(pt_file);
      while (m_iElem != -999 && m_iElem != -47474747)
      readNextIntElement(pt_file);*/
    Covise::sendInfo("done reading IFS");
}

//==============================================================================

int ReadVTFData::getNumNodesInElem(int eType)
{
    int iRet = 0;

    switch (eType)
    {
    case IFS:
    {
        iRet = 200;
        break;
    }

    case 1:
    case 17:
        iRet = 2; // TYPE_BAR;
        break;
    case 2:
        iRet = 3; // TYPE_BAR;
        break;
    case 3:
        iRet = 3; // TYPE_TRIANGLE;
        break;
    case 4:
        iRet = 6; // TYPE_TRIANGLE;
        break;
    case 5:
    case 15:
    case 16:
        iRet = 4; // TYPE_QUAD;
        break;
    case 6:
        iRet = 8; // TYPE_QUAD;
        break;
    case 7:
        iRet = 4; // TYPE_TETRAHEDER;
        break;
    case 8:
        iRet = 10; // TYPE_TETRAHEDER;
        break;
    case 9:
    case 13:
    case 14:
    case 24:
        iRet = 8; // TYPE_HEXAEDER;
        break;
    case 10:
        iRet = 20; // TYPE_HEXAEDER;
        break;
    case 11:
        iRet = 6; // TYPE_PRISM;
        break;
    case 12:
        iRet = 15; // TYPE_PRISM;
        break;
    case 18:
        iRet = 1; // TYPE_POINT;
        break;
    case 20:
        iRet = 4; // TYPE_TETRAHEDER;
        break;
    default:
        iRet = 0;
        break;
    }

    return iRet;
}

//==============================================================================

int ReadVTFData::getNumNodesInElem(int iDataSize, int iNumElements, int iWith_ID)
{
    int iRet;

    iRet = iDataSize - 8; // DataHeader ElementsType+NumElements=8 bytes

    if (iWith_ID)
    {
        iRet = iRet - (iNumElements * 4); // UserID
    }

    iRet = iRet / 4; // number of integers
    iRet = iRet / iNumElements;

    return iRet;
}

//==============================================================================

int ReadVTFData::getCoviseType(int eType)
{
    int iRet = 0;

    switch (eType)
    {
    case IFS:
        iRet = IFS;
        break;
    case 1:
    case 2:
    case 17:
        iRet = TYPE_BAR;
        break;
    case 3:
    case 4:
        iRet = TYPE_TRIANGLE;
        break;
    case 5:
    case 6:
    case 15:
    case 16:
        iRet = TYPE_QUAD;
        break;
    case 7:
    case 8:
    case 20:
        iRet = TYPE_TETRAHEDER;
        break;
    case 9:
    case 10:
    case 13:
    case 14:
    case 24:
        iRet = TYPE_HEXAEDER;
        break;
    case 11:
    case 12:
        iRet = TYPE_PRISM;
        break;
    case 18:
        iRet = TYPE_POINT;
        break;
    default:
        iRet = 0;
        break;
    }

    return iRet;
}

//==============================================================================

int ReadVTFData::getNumCoviseNodesInElem(int eType)
{
    int iRet = 0;

    switch (eType)
    {
    case TYPE_BAR:
        iRet = 2;
        break;
    case TYPE_TRIANGLE:
        iRet = 3;
        break;
    case TYPE_QUAD:
        iRet = 4;
        break;
    case TYPE_TETRAHEDER:
        iRet = 4;
        break;
    case TYPE_HEXAEDER:
        iRet = 8;
        break;
    case TYPE_PRISM:
        iRet = 6;
        break;
    case TYPE_POINT:
        iRet = 1;
        break;
    default:
        iRet = 0;
        break;
    }

    return iRet;
}

//==============================================================================

int ReadVTFData::readElement(FILE *pt_file, bool is_poly)
{
    if (is_poly)
        Covise::sendInfo("IFS");

    // Block ID
    readNextIntElement(pt_file);

    m_tElem[m_iElemCounter].ID = m_iElem;

    // fill struct ElementHeader
    readNextIntElement(pt_file);

    m_tElem[m_iElemCounter].header.iHeaderSize = m_iElem;

    readNextIntElement(pt_file);

    m_tElem[m_iElemCounter].header.iDataSize = m_iElem;

    readNextIntElement(pt_file);

    m_tElem[m_iElemCounter].header.iNodeBlockID = m_iElem;

    readNextCharElement(pt_file);

    //Covise::sendInfo(m_cElem);
    readNextFloatElement(pt_file);

    m_tElem[m_iElemCounter].header.fRColor = m_fElem;

    readNextFloatElement(pt_file);

    m_tElem[m_iElemCounter].header.fGColor = m_fElem;

    readNextFloatElement(pt_file);

    m_tElem[m_iElemCounter].header.fBColor = m_fElem;

    readNextIntElement(pt_file);

    m_tElem[m_iElemCounter].header.iWithID = m_iElem;

    readNextIntElement(pt_file);

    m_tElem[m_iElemCounter].header.iNumElementTypes = (is_poly) ? 1 : m_iElem;

    int num_polygons = m_iElem;

    bool unknownVer7FileFormatJustAGuess = false;

    // 112 bytes of header, if the header is longer, we have to skip the rest we dont know anything about
    if (m_tElem[m_iElemCounter].header.iHeaderSize > 112)
    {
        unknownVer7FileFormatJustAGuess = true;
        ignoreIDs = true;
        seek(m_tElem[m_iElemCounter].header.iHeaderSize - 112);
    }

    if (is_poly)
    {
        readNextIntElement(pt_file); //number of connections
    }

    m_tElem[m_iElemCounter].data = new ElementData[m_tElem[m_iElemCounter].header.iNumElementTypes];
    int iNodesInElem = 0;
    // fill struct ElementData

    for (int i = 0; i < m_tElem[m_iElemCounter].header.iNumElementTypes; i++)
    {
        if (unknownVer7FileFormatJustAGuess)
        {
            readNextIntElement(pt_file);
        }

        // fill struct ElementDataHeader
        if (!is_poly)
        {
            readNextIntElement(pt_file);
        }
        else
        {
            m_iElem = IFS;
        }

        m_tElem[m_iElemCounter].data[i].dataHeader.eType = m_iElem;

        if (!is_poly)
        {
            readNextIntElement(pt_file);
        }
        else
        {
            m_iElem = num_polygons;
        }

        m_tElem[m_iElemCounter].data[i].dataHeader.iNumElements = m_iElem;

        m_tElem[m_iElemCounter].data[i].tElement = new ElementElement[m_tElem[m_iElemCounter].data[i].dataHeader.iNumElements];

        //int iNodesInElemOne= getNumNodesInElem
        //                                                (m_tElem[m_iElemCounter].header.iDataSize,
        //                                                m_tElem[m_iElemCounter].data[i].dataHeader.iNumElements,
        //                                                m_tElem[m_iElemCounter].header.iWithID);

        iNodesInElem = (is_poly) ? MAX_POLY_CONN : getNumNodesInElem(m_tElem[m_iElemCounter].data[i].dataHeader.eType);

        if (!iNodesInElem)
        {
            Covise::sendError("nodes in element = 0");
            return 1;
        }

        m_tElem[m_iElemCounter].data[i].dataHeader.eType = getCoviseType(m_tElem[m_iElemCounter].data[i].dataHeader.eType);

        if (unknownVer7FileFormatJustAGuess)
        {
            readNextIntElement(pt_file);
            readNextIntElement(pt_file);
        }

        for (int j = 0; j < m_tElem[m_iElemCounter].data[i].dataHeader.iNumElements; j++)
        {
            m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID = new int[iNodesInElem];

            if (m_tElem[m_iElemCounter].header.iWithID)
            {
                readNextIntElement(pt_file);
                m_tElem[m_iElemCounter].data[i].tElement[j].UserID = m_iElem;
            }

            int n = 0;

            if (is_poly)
            {
                do
                {
                    readNextIntElement(pt_file);
                    m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID[n++] = m_iElem;
                } while (m_iElem >= 0);

                m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID[n - 1] *= -1;

                m_tElem[m_iElemCounter].data[i].tElement[j].iNumNodes = n;
            }
            else
            {
                m_tElem[m_iElemCounter].data[i].tElement[j].iNumNodes = getNumCoviseNodesInElem(m_tElem[m_iElemCounter].data[i].dataHeader.eType);

                switch (iNodesInElem)
                {
                case 10: // we only need the first 4 points

                    for (n = 0; n < 4; n++)
                    {
                        readNextIntElement(pt_file);
                        m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID[n] = m_iElem;
                    }

                    for (n = 0; n < 6; n++) // we don't need these points, so let's kill them
                    {
                        readNextIntElement(pt_file);
                    }
                    //m_tElem[m_iElemCounter].data[i].dataHeader.eType = 4;
                    break;

                case 15: // we only need the first 6 points
                    for (n = 0; n < 6; n++)
                    {
                        readNextIntElement(pt_file);
                        m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID[n] = m_iElem;
                    }

                    for (n = 0; n < 9; n++) // we don't need these points, so let's kill them
                    {
                        readNextIntElement(pt_file);
                    }
                    //m_tElem[m_iElemCounter].data[i].dataHeader.eType = 6;
                    break;

                case 20: // we need only the first 8 points
                    for (n = 0; n < 8; n++)
                    {
                        readNextIntElement(pt_file);
                        m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID[n] = m_iElem;
                    }

                    for (n = 0; n < 12; n++) // we don't need these points, so let's kill them
                    {
                        readNextIntElement(pt_file);
                    }
                    //m_tElem[m_iElemCounter].data[i].dataHeader.eType = 8;
                    break;

                case 2:
                case 3:
                case 4:
                case 6:
                case 8:
                    for (n = 0; n < iNodesInElem; n++)
                    {
                        readNextIntElement(pt_file);
                        m_tElem[m_iElemCounter].data[i].tElement[j].iNodeID[n] = m_iElem;
                    }

                    break;
                default:
                    Covise::sendError("can't read this element type!");
                    return 1;
                    break;
                }
            }

            m_tElem[m_iElemCounter].data[i].tElement[j].iPosInPolyList = -1;
        }
    }

    m_iElemCounter++; // now there is one more Element in the array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
    {
        Covise::sendError("Error reading ELEMENT");
        return 1;

        while (m_iElem != -999 && m_iElem != -47474747)
            readNextIntElement(pt_file);
    }

    return 0;
}

//==============================================================================

void ReadVTFData::readGLViewGeometry(FILE *pt_file)
{
    Covise::sendInfo("ANI - reading GLVIEWGEOMETRY");
    // Block ID
    readNextIntElement(pt_file);
    m_tGeom[m_iGeomCounter].ID = m_iElem;

    // fill struct GeometryHeader
    readNextIntElement(pt_file);
    m_tGeom[m_iGeomCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tGeom[m_iGeomCounter].header.iDataSize = m_iElem;
    readNextCharElement(pt_file);
    Covise::sendInfo("%s", m_cElem);
    readNextIntElement(pt_file);
    m_tGeom[m_iGeomCounter].header.iNumSteps = m_iElem;

    if (m_tGeom[m_iGeomCounter].header.iHeaderSize == 100)
    {
        // longer header than documented
        readNextIntElement(pt_file);

        if (m_iElem != 1)
        {
            Covise::sendInfo("Undocumented Header, last time we got 1 now it is: %d", m_iElem);
        }

        readNextIntElement(pt_file);

        if (m_iElem != 1)
        {
            Covise::sendInfo("Undocumented Header, last time we got 1 now it is: %d", m_iElem);
        }
    }

    m_tGeom[m_iGeomCounter].dataHeader = new GeometryDataHeader[m_tGeom[m_iGeomCounter].header.iNumSteps];

    m_tGeom[m_iGeomCounter].data = new GeometryData[m_tGeom[m_iGeomCounter].header.iNumSteps];

    for (int i = 0; i < m_tGeom[m_iGeomCounter].header.iNumSteps; i++)
    {
        // fill struct GeometryDataHeader
        readNextIntElement(pt_file);
        m_tGeom[m_iGeomCounter].dataHeader[i].iStepNumber = m_iElem;
        readNextCharElement(pt_file); // we don't need this
        readNextFloatElement(pt_file);
        m_tGeom[m_iGeomCounter].dataHeader[i].fStepTime = m_fElem;
        readNextIntElement(pt_file);
        m_tGeom[m_iGeomCounter].dataHeader[i].iNumElementBlocks = m_iElem;
        readNextIntElement(pt_file);
        m_tGeom[m_iGeomCounter].dataHeader[i].iNumIFSBlocks = m_iElem;
        readNextIntElement(pt_file);
        m_tGeom[m_iGeomCounter].dataHeader[i].iControl1 = m_iElem;
        readNextIntElement(pt_file);
        m_tGeom[m_iGeomCounter].dataHeader[i].iControl2 = m_iElem;
        //

        if (m_tGeom[m_iGeomCounter].header.iHeaderSize == 100)
        {
            // longer header than documented and also longer content
            readNextIntElement(pt_file);

            if (m_iElem != i)
            {
                Covise::sendInfo("Undocumented Header, last time we got %d now it is: %d", i, m_iElem);
            }

            readNextIntElement(pt_file);

            if (m_iElem != i)
            {
                Covise::sendInfo("Undocumented Header, last time we got %d now it is: %d", i, m_iElem);
            }
        }

        // only use these infos for testing files
        Covise::sendInfo("step number: %d", m_tGeom[m_iGeomCounter].dataHeader[i].iStepNumber);

        Covise::sendInfo("step time: %f", m_tGeom[m_iGeomCounter].dataHeader[i].fStepTime);

        Covise::sendInfo("number of element blocks: %d", m_tGeom[m_iGeomCounter].dataHeader[i].iNumElementBlocks);

        Covise::sendInfo("number of IFS blocks: %d", m_tGeom[m_iGeomCounter].dataHeader[i].iNumIFSBlocks);

        m_tGeom[m_iGeomCounter].data[i].iElemBlockID = new int[m_tGeom[m_iGeomCounter].dataHeader[i].iNumElementBlocks];

        //        if (m_tGeom[m_iGeomCounter].dataHeader[i].iNumIFSBlocks > 1000) m_tGeom[m_iGeomCounter].dataHeader[i].iNumIFSBlocks = 0;
        m_tGeom[m_iGeomCounter].data[i].iIFSBlock = new int[m_tGeom[m_iGeomCounter].dataHeader[i].iNumIFSBlocks];

        // fill struct GeometryData
        for (int j = 0; j < m_tGeom[m_iGeomCounter].dataHeader[i].iNumElementBlocks; j++)
        {
            readNextIntElement(pt_file);
            m_tGeom[m_iGeomCounter].data[i].iElemBlockID[j] = m_iElem;
            Covise::sendInfo("ElemBlockID: %d", m_iElem);
        }

        for (int k = 0; k < m_tGeom[m_iGeomCounter].dataHeader[i].iNumIFSBlocks; k++)
        {
            readNextIntElement(pt_file);
            m_tGeom[m_iGeomCounter].data[i].iIFSBlock[k] = m_iElem;
        }
    }

    m_iGeomCounter += 1; // now there is one more Geometry in the array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading GLVIEWGEOMETRY");

    Covise::sendInfo("End reading GLVIEWGEOMETRY");
}

//==============================================================================

int ReadVTFData::readResult(FILE *pt_file)
{

    // Block ID
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].ID = m_iElem;

    // fill struct ResultHeader
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iDataSize = m_iElem;
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iDimension = m_iElem;
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iMapToBlockID = m_iElem;
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iMappingType = m_iElem;
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iWithID = m_iElem;
    readNextIntElement(pt_file);
    m_tRes[m_iResCounter].header.iNumResults = m_iElem;

    // fill struct ResultData

    m_tRes[m_iResCounter].data = new ResultData[m_tRes[m_iResCounter].header.iNumResults];

    for (int i = 0; i < m_tRes[m_iResCounter].header.iNumResults; i++)
    {
        if (m_tRes[m_iResCounter].header.iWithID)
        {
            readNextIntElement(pt_file);
            m_tRes[m_iResCounter].data[i].iUserID = m_iElem;
        }

        if (m_tRes[m_iResCounter].header.iDimension == 1)
        {
            readNextFloatElement(pt_file);
            m_tRes[m_iResCounter].data[i].fScalarValue = m_fElem;
            m_iS3DCounter++;
            /* // only for testing
            if (m_tRes[m_iResCounter].data[i].fScalarValue)
            DataInput<<m_tRes[m_iResCounter].data[i].fScalarValue<<endl;
         */
        }
        else
        {
            for (int j = 0; j < 3; j++)
            {
                readNextFloatElement(pt_file);
                m_tRes[m_iResCounter].data[i].fVectorValue[j] = m_fElem;
                m_iV3DCounter++;
            }

            /* // only for testing
            if (m_tRes[m_iResCounter].data[i].fVectorValue[0] ||
            m_tRes[m_iResCounter].data[i].fVectorValue[1] ||
            m_tRes[m_iResCounter].data[i].fVectorValue[2])
            {
            DataInput<<m_tRes[m_iResCounter].data[i].fVectorValue[0]<<" "
            <<m_tRes[m_iResCounter].data[i].fVectorValue[1]<<" "
            <<m_tRes[m_iResCounter].data[i].fVectorValue[2]<<endl;
            }
         */
        }
    }

    m_iResCounter++; // now there is one more Result in the array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
    {
        Covise::sendError("Error reading RESULTS");
        return 1;
    }

    return 0;
}

//==============================================================================

void ReadVTFData::readGLViewScalar(FILE *pt_file)
{
    Covise::sendInfo("GLVIEWSCALAR");
    // Block ID
    readNextIntElement(pt_file);
    m_tViewS[m_iVSCounter].ID = m_iElem;

    // fill struct GLViewScalarHeader
    readNextIntElement(pt_file);
    m_tViewS[m_iVSCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tViewS[m_iVSCounter].header.iDataSize = m_iElem;
    readNextCharElement(pt_file);
    //Covise::sendInfo(m_cElem);
    readNextIntElement(pt_file);
    m_tViewS[m_iVSCounter].header.iNumSteps = m_iElem;
    m_tViewS[m_iVSCounter].header.iResultID = 0;
    m_tViewS[m_iVSCounter].header.iSectionID = 0;
    m_tViewS[m_iVSCounter].header.iWithStateID = 0;

    if (m_tViewS[m_iVSCounter].header.iHeaderSize == 104)
    {
        readNextIntElement(pt_file);
        m_tViewS[m_iVSCounter].header.iResultID = m_iElem;
        readNextIntElement(pt_file);
        m_tViewS[m_iVSCounter].header.iSectionID = m_iElem;
        readNextIntElement(pt_file);
        m_tViewS[m_iVSCounter].header.iWithStateID = m_iElem;
    }

    Covise::sendInfo("num of time steps: %d", m_tViewS[m_iVSCounter].header.iNumSteps);

    m_tViewS[m_iVSCounter].dataHeader = new GLViewScalarDataHeader[m_tViewS[m_iVSCounter].header.iNumSteps];
    m_tViewS[m_iVSCounter].data = new GLViewScalarData[m_tViewS[m_iVSCounter].header.iNumSteps];

    for (int j = 0; j < m_tViewS[m_iVSCounter].header.iNumSteps; j++)
    {
        // fill struct GLViewScalarDataHeader
        readNextIntElement(pt_file);
        m_tViewS[m_iVSCounter].dataHeader[j].iStepNumber = m_iElem;
        readNextCharElement(pt_file); // we don't need this
        readNextFloatElement(pt_file);
        m_tViewS[m_iVSCounter].dataHeader[j].fStepTime = m_fElem;
        readNextIntElement(pt_file);
        m_tViewS[m_iVSCounter].dataHeader[j].iNumResultsBlocks = m_iElem;
        m_tViewS[m_iVSCounter].data[j].iResultBlockID = new int[m_tViewS[m_iVSCounter].dataHeader[j].iNumResultsBlocks];

        if (m_tViewS[m_iVSCounter].header.iWithStateID)
            readNextIntElement(pt_file);

        // fill struct GLViewScalarData
        for (int k = 0; k < m_tViewS[m_iVSCounter].dataHeader[j].iNumResultsBlocks; k++)
        {
            readNextIntElement(pt_file);
            m_tViewS[m_iVSCounter].data[j].iResultBlockID[k] = m_iElem;
        }
    }

    //sprintf(infobuf, "num of result blocks: %d", m_tViewS[m_iVSCounter].dataHeader[0].iNumResultsBlocks);
    //Covise::sendInfo(infobuf);

    m_iVSCounter += 1; // now there is one more GLViewScalar in the array

    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading GLVIEWSCALAR");
}

//==============================================================================

void ReadVTFData::readGLViewVector(FILE *pt_file)
{
    Covise::sendInfo("GLVIEWVECTOR");
    // Block ID
    readNextIntElement(pt_file);
    m_tViewV[m_iVVCounter].ID = m_iElem;

    // fill struct GLViewScalarHeader
    readNextIntElement(pt_file);
    m_tViewV[m_iVVCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tViewV[m_iVVCounter].header.iDataSize = m_iElem;
    readNextCharElement(pt_file);
    //Covise::sendInfo(m_cElem);
    readNextIntElement(pt_file);
    m_tViewV[m_iVVCounter].header.iNumSteps = m_iElem;
    Covise::sendInfo("num of time steps: %d", m_tViewV[m_iVVCounter].header.iNumSteps);

    m_tViewV[m_iVVCounter].dataHeader = new GLViewVectorDataHeader[m_tViewV[m_iVVCounter].header.iNumSteps];
    m_tViewV[m_iVVCounter].data = new GLViewVectorData[m_tViewV[m_iVVCounter].header.iNumSteps];

    for (int j = 0; j < m_tViewV[m_iVVCounter].header.iNumSteps; j++)
    {
        // fill struct GLViewScalarDataHeader
        readNextIntElement(pt_file);
        m_tViewV[m_iVVCounter].dataHeader[j].iStepNumber = m_iElem;
        readNextCharElement(pt_file); // we don't need this
        readNextFloatElement(pt_file);
        m_tViewV[m_iVVCounter].dataHeader[j].fStepTime = m_fElem;
        readNextIntElement(pt_file);
        m_tViewV[m_iVVCounter].dataHeader[j].iNumResultsBlocks = m_iElem;

        m_tViewV[m_iVVCounter].data[j].iResultBlockID = new int[m_tViewV[m_iVVCounter].dataHeader[j].iNumResultsBlocks];

        // fill struct GLViewScalarData

        for (int k = 0; k < m_tViewV[m_iVVCounter].dataHeader[j].iNumResultsBlocks; k++)
        {
            readNextIntElement(pt_file);
            m_tViewV[m_iVVCounter].data[j].iResultBlockID[k] = m_iElem;
        }
    }

    //sprintf(infobuf, "num of result blocks: %d",       m_tViewV[m_iVVCounter].dataHeader[0].iNumResultsBlocks);
    //Covise::sendInfo(infobuf);

    m_iVVCounter += 1; // now there is one more GLViewScalar in the array

    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading GLVIEWVECTOR");
}

//==============================================================================

void ReadVTFData::readTransformations(FILE *pt_file)
{
    Covise::sendInfo("TRANSFORMATIONS");
    int i, j;

    // Block ID
    readNextIntElement(pt_file);
    m_tTrans[m_iTransCounter].ID = m_iElem;

    // fill struct TransformationHeader
    readNextIntElement(pt_file);
    m_tTrans[m_iTransCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tTrans[m_iTransCounter].header.iDataSize = m_iElem;
    readNextCharElement(pt_file); // we don't need this
    readNextIntElement(pt_file);
    m_tTrans[m_iTransCounter].header.iWithID = m_iElem;
    readNextIntElement(pt_file);
    m_tTrans[m_iTransCounter].header.iNumSteps = m_iElem;
    m_tTrans[m_iTransCounter].dataHeader = new TransformationDataHeader[m_tTrans[m_iTransCounter].header.iNumSteps];
    m_tTrans[m_iTransCounter].data = new TransformationData[m_tTrans[m_iTransCounter].header.iNumSteps];

    //cerr << "trans " << m_iTransCounter << " with " << m_tTrans[m_iTransCounter].header.iNumSteps << endl;

    for (i = 0; i < m_tTrans[m_iTransCounter].header.iNumSteps; i++)
    {
        // fill struct TransformationDataHeader
        readNextIntElement(pt_file);
        m_tTrans[m_iTransCounter].dataHeader[i].iStepNumber = m_iElem;
        readNextCharElement(pt_file); // we don't need this
        readNextFloatElement(pt_file);
        m_tTrans[m_iTransCounter].dataHeader[i].fStepTime = m_fElem;
        readNextIntElement(pt_file);
        m_tTrans[m_iTransCounter].dataHeader[i].iNumElementBlocks = m_iElem;
        readNextIntElement(pt_file);
        m_tTrans[m_iTransCounter].dataHeader[i].iNumIFSBlocks = m_iElem;

        if (m_tTrans[m_iTransCounter].dataHeader[i].iNumElementBlocks > 0)
        {
            m_tTrans[m_iTransCounter].data[i].iElementBlockID = new int[m_tTrans[m_iTransCounter].dataHeader[i].iNumElementBlocks];

            m_tTrans[m_iTransCounter].data[i].iElementTransMatrix = new float *[m_tTrans[m_iTransCounter].dataHeader[i].iNumElementBlocks];
        }
        else
        {
            m_tTrans[m_iTransCounter].data[i].iElementBlockID = NULL;
            m_tTrans[m_iTransCounter].data[i].iElementTransMatrix = NULL;
        }

        //        }
        //        for ( i=0; i<m_tTrans[m_iTransCounter].header.iNumSteps; i++)
        //        {
        // fill struct TransformationData
        for (j = 0; j < m_tTrans[m_iTransCounter].dataHeader[i].iNumElementBlocks; j++)
        {

            m_tTrans[m_iTransCounter].data[i].iElementTransMatrix[j] = new float[12];

            if (m_tTrans[m_iTransCounter].header.iWithID)
            {
                readNextIntElement(pt_file);
                m_tTrans[m_iTransCounter].data[i].iElementBlockID[j] = m_iElem;
            }

            for (int k = 0; k < 12; k++)
            {
                readNextFloatElement(pt_file);
                m_tTrans[m_iTransCounter].data[i].iElementTransMatrix[j][k] = m_fElem;
            }
        }

        if (m_tTrans[m_iTransCounter].dataHeader[i].iNumIFSBlocks > 0)
        {
            m_tTrans[m_iTransCounter].data[i].iIFSID = new int[m_tTrans[m_iTransCounter].dataHeader[i].iNumIFSBlocks];

            m_tTrans[m_iTransCounter].data[i].iIFSTransMatrix = new float *[m_tTrans[m_iTransCounter].dataHeader[i].iNumIFSBlocks];
        }
        else
        {
            m_tTrans[m_iTransCounter].data[i].iIFSID = NULL;
            m_tTrans[m_iTransCounter].data[i].iIFSTransMatrix = NULL;
        }

        for (j = 0; j < m_tTrans[m_iTransCounter].dataHeader[i].iNumIFSBlocks; j++)
        {
            m_tTrans[m_iTransCounter].data[i].iIFSTransMatrix[j] = new float[12];

            if (m_tTrans[m_iTransCounter].header.iWithID)
            {
                readNextIntElement(pt_file);
                m_tTrans[m_iTransCounter].data[i].iIFSID[j] = m_iElem;
            }

            for (int k = 0; k < 12; k++)
            {
                readNextFloatElement(pt_file);
                m_tTrans[m_iTransCounter].data[i].iIFSTransMatrix[j][k] = m_fElem;
            }
        }
    }

    m_iTransCounter += 1; // now there is one more Transformation in the array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading TRANSFORMATION");

    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);

    Covise::sendInfo("reading TRANSFORMATION");
}

//==============================================================================

void ReadVTFData::readViewPoints(FILE *pt_file)
{
    Covise::sendInfo("VIEWPOINTS");
    // not used
    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);
}

//==============================================================================

void ReadVTFData::readPloteries(FILE *pt_file)
{
    Covise::sendInfo("PLOTERIES");
    // not used
    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);
}

//==============================================================================

void ReadVTFData::readUser(FILE *pt_file)
{
    Covise::sendInfo("USER");
    // not used
    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);
}

//==============================================================================

void ReadVTFData::readPositionResults(FILE *pt_file)
{
    /*
   // Block ID
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].ID = m_iElem;

   // fill struct PositionResultsHeader
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iHeaderSize = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iDataSize = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iDimension = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iMapToBlockType = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iMapToBlockID = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iMappingType = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iGlobalPositions = m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iGlobalResults= m_iElem;
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].header.iNumResults = m_iElem;

   // fill struct PositionResultsData

   for (int i=0; i<m_tPosRes[m_iPosResCounter].header.iNumResults; i++)
   {
   if (m_tPosRes[m_iPosResCounter].header.iMappingType)
   {
   readNextIntElement(pt_file);
   m_tPosRes[m_iPosResCounter].data[i].iMap = m_iElem;
   }
   for (int k=0; k<3; k++)
   {
   readNextFloatElement(pt_file);
   m_tPosRes[m_iPosResCounter].data[i].fPosition[k] = m_fElem;
   }
   if (m_tPosRes[m_iPosResCounter].header.iDimension == 1)
   {
   readNextFloatElement(pt_file);
   m_tPosRes[m_iPosResCounter].data[i].fScalarValue = m_fElem;
   }
   else
   {
   for (int j=0; j<3; j++)
   {
   readNextFloatElement(pt_file);
   m_tPosRes[m_iPosResCounter].data[i].fVectorValue[j] = m_fElem;
   }
   }
   }
   m_iPosResCounter += 1; // now there is one more Result in the array
   readNextIntElement(pt_file); // end marker: -999 or -47474747
   if (m_iElem != -999 && m_iElem != -47474747)
   Covise::sendError("Error reading POSITIONRESULTS");*/
    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);

    Covise::sendInfo("reading POSITIONRESULTS");
}

//==============================================================================

void ReadVTFData::readPositionScalar(FILE *pt_file)
{
    /*
   // Block ID
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].ID = m_iElem;

   // fill struct GLViewPositionScalarHeader
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].header.iHeaderSize = m_iElem;
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].header.iDataSize = m_iElem;
   readNextCharElement(pt_file); // we don't need this
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].header.iNumSteps = m_iElem;

   for (int i=0; i<m_tPosS[m_iPosSCounter].header.iNumSteps; i++)
   {
   // fill struct GLViewPositionScalarDataHeader
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].dataHeader[i].iStepNumber = m_iElem;
   readNextCharElement(pt_file); // we don't need this
   readNextFloatElement(pt_file);
   m_tPosS[m_iPosSCounter].dataHeader[i].fStepTime = m_fElem;
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].dataHeader[i].iNumResultsBlocks = m_iElem;

   // fill struct GLViewPositionScalarData
   for (int j=0; j<m_tPosS[m_iPosSCounter].dataHeader[i].iNumResultsBlocks; j++)
   {
   readNextIntElement(pt_file);
   m_tPosS[m_iPosSCounter].data[i].iPosResultBlockID[j] = m_iElem;
   }
   }
   m_iPosSCounter += 1; // now there is one more GLViewPositionScalar in array
   readNextIntElement(pt_file); // end marker: -999 or -47474747
   if (m_iElem != -999 && m_iElem != -47474747)
   Covise::sendError("Error reading GLVIEWPOSITIONSCALAR");     */
    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);

    Covise::sendInfo("reading GLVIEWPOSITIONSCALAR");
}

//==============================================================================

void ReadVTFData::readGLViewPositionVector(FILE *pt_file)
{
    /*
   // Block ID
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].ID = m_iElem;

   // fill struct GLViewPositionVectorHeader
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].header.iHeaderSize = m_iElem;
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].header.iDataSize = m_iElem;
   readNextCharElement(pt_file); // we don't need this
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].header.iNumSteps = m_iElem;

   for (int i=0; i<m_tPosV[m_iPosVCounter].header.iNumSteps; i++)
   {
   // fill struct GLViewPositionVectorDataHeader
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].dataHeader[i].iStepNumber = m_iElem;
   readNextCharElement(pt_file); // we don't need this
   readNextFloatElement(pt_file);
   m_tPosV[m_iPosVCounter].dataHeader[i].fStepTime = m_fElem;
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].dataHeader[i].iNumResultsBlocks = m_iElem;

   // fill struct GLViewPositionVectorData
   for (int j=0; j<m_tPosV[m_iPosVCounter].dataHeader[i].iNumResultsBlocks; j++)
   {
   readNextIntElement(pt_file);
   m_tPosV[m_iPosVCounter].data[i].iPosResultBlockID[j] = m_iElem;
   }
   }
   m_iPosVCounter += 1; // now there is one more GLViewPositionVector in array
   readNextIntElement(pt_file); // end marker: -999 or -47474747
   if (m_iElem != -999 && m_iElem != -47474747)
   Covise::sendError("Error reading GLVIEWPOSITIONVECTOR");*/
    readNextIntElement(pt_file);

    while (m_iElem != -999 && m_iElem != -47474747)
        readNextIntElement(pt_file);

    Covise::sendInfo("reading GLVIEWPOSITIONVECTOR");
}

//==============================================================================

void ReadVTFData::readTransformationResult(FILE *pt_file)
{

    readNextIntElement(pt_file);
    m_tTransRes[m_iTransResCounter].ID = m_iElem;

    // fill struct TransformationResultHeader
    readNextIntElement(pt_file);
    m_tTransRes[m_iTransResCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tTransRes[m_iTransResCounter].header.iDataSize = m_iElem;
    readNextIntElement(pt_file);
    m_tTransRes[m_iTransResCounter].header.iIFSBlockID = m_iElem;
    readNextIntElement(pt_file);
    m_tTransRes[m_iTransResCounter].header.iElementBlockID = m_iElem;

    for (int i = 0; i < 12; i++)
    {
        readNextFloatElement(pt_file);
        m_tTransRes[m_iTransResCounter].pfTransformationMatrix[i] = m_fElem;
    }

    m_iTransResCounter += 1; // there is one more TransformationResult in array
    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading TRANSFORMATIONRESULT");

    /*
     readNextIntElement(pt_file);
     while (m_iElem != -999 && m_iElem != -47474747)
     readNextIntElement(pt_file);
   */
    Covise::sendInfo("reading TRANSFORMATIONRESULT");
}

//==============================================================================
void ReadVTFData::readGLViewTransformation(FILE *pt_file)
{
    // Block ID
    readNextIntElement(pt_file);
    m_tViewTrans[m_iViewTransCounter].ID = m_iElem;

    // fill struct ViewTransformationHeader
    readNextIntElement(pt_file);
    m_tViewTrans[m_iViewTransCounter].header.iHeaderSize = m_iElem;
    readNextIntElement(pt_file);
    m_tViewTrans[m_iViewTransCounter].header.iDataSize = m_iElem;
    readNextCharElement(pt_file); // we don't need this
    readNextIntElement(pt_file);
    m_tViewTrans[m_iViewTransCounter].header.iNumSteps = m_iElem;

    readNextIntElement(pt_file); //iWithStateID
    readNextIntElement(pt_file); //iWithStateID
    m_tViewTrans[m_iViewTransCounter].header.iWithStateID = m_iElem;

    for (int i = 0; i < m_tViewTrans[m_iViewTransCounter].header.iNumSteps; i++)
    {
        // fill struct ViewTransformationDataHeader
        readNextIntElement(pt_file);
        m_tViewTrans[m_iViewTransCounter].dataHeader[i].iStepNumber = m_iElem;
        readNextCharElement(pt_file); // we don't need this
        readNextFloatElement(pt_file);
        m_tViewTrans[m_iViewTransCounter].dataHeader[i].fStepTime = m_fElem;
        readNextIntElement(pt_file);
        m_tViewTrans[m_iViewTransCounter].dataHeader[i].iNumTransBlocks = m_iElem;

        if (m_tViewTrans[m_iViewTransCounter].header.iWithStateID == 1)
        {
            readNextIntElement(pt_file);
            m_tViewTrans[m_iViewTransCounter].dataHeader[i].iStateID = m_iElem;
        }

        // fill struct ViewTransformationData
        for (int j = 0; j < m_tViewTrans[m_iViewTransCounter].dataHeader[i].iNumTransBlocks; j++)
        {
            //      for (int k=0; k<m_tViewTrans[m_iViewTransCounter].dataHeader[i].iNumTransBlocks; k++)
            //      {
            readNextIntElement(pt_file);
            m_tViewTrans[m_iViewTransCounter].data[i].iTransResultsBlockID[j] = m_iElem;
            //      }
        }
    }

    m_iViewTransCounter += 1; // now there is one more Transformation in the array

    readNextIntElement(pt_file); // end marker: -999 or -47474747

    if (m_iElem != -999 && m_iElem != -47474747)
        Covise::sendError("Error reading VIEWTRANSFORMATION");

    /*      readNextIntElement(pt_file);
           while (m_iElem != -999 && m_iElem != -47474747)
           readNextIntElement(pt_file);
   */
    Covise::sendInfo("end reading GLVIEWTRANSFORMATION");
}

//==============================================================================

int ReadVTFData::getPosInCornerList(int iNodeID, int iPointID)
{
    int iPos = -1;

    if (m_tNodes[iNodeID].header.iWithID)
    {
        // perhaps it's easy to find the right point...
        int iFirstID = m_tNodes[iNodeID].point[0].ID;
        int iPosID = iPointID - iFirstID;

        if (iPosID >= 0 && iPosID <= m_tNodes[iNodeID].header.iNumNodes && m_tNodes[iNodeID].point[iPosID].ID == iPointID)
            iPos = m_tNodes[iNodeID].point[iPosID].iPosInCornerList;
        else // ...or it's not so easy:-(
        {
            int i = 0;

            while (m_tNodes[iNodeID].point[i].ID != iPointID)
            {
                i++;

                if (i > m_tNodes[iNodeID].header.iNumNodes)
                {
                    Covise::sendError("can't get PosInCornerList");
                    break;
                }
            }

            iPos = m_tNodes[iNodeID].point[i].iPosInCornerList;
        }
    }
    else
    {
        iPos = m_tNodes[iNodeID].point[iPointID - 1].iPosInCornerList;
    }

    return iPos;
}

//==============================================================================

void ReadVTFData::setPosInCornerList(int iNodeID, int iPointID, int iNumCorners)
{
    if (m_tNodes[iNodeID].header.iWithID)
    {
        // perhaps it's easy to find the right point...
        int iFirstID = m_tNodes[iNodeID].point[0].ID;
        int iPos = iPointID - iFirstID;

        if (iPos >= 0 && iPos <= m_tNodes[iNodeID].header.iNumNodes && m_tNodes[iNodeID].point[iPos].ID == iPointID)
        {
            if (m_tNodes[iNodeID].point[iPos].iPosInCornerList == -1)
                m_tNodes[iNodeID].point[iPos].iPosInCornerList = iNumCorners;
        }
        else // ...or it's not so easy:-(
        {
            int iCount = 0;

            while (m_tNodes[iNodeID].point[iCount].ID != iPointID)
            {
                iCount++;

                if (iCount > m_tNodes[iNodeID].header.iNumNodes)
                {
                    Covise::sendError("can't find place in corner list");
                    break;
                }
            }

            if (m_tNodes[iNodeID].point[iCount].iPosInCornerList == -1)
                m_tNodes[iNodeID].point[iCount].iPosInCornerList = iNumCorners;
        }
    }
    else
    {
        if (m_tNodes[iNodeID].point[iPointID - 1].iPosInCornerList == -1)
            m_tNodes[iNodeID].point[iPointID - 1].iPosInCornerList = iNumCorners;
    }
}

//==============================================================================

int ReadVTFData::getPosInElemList(int iElemID, int iUserID)
{
    int iPosInElemList = -1;
    int i = 0;
    bool bFlag = false;

    if (m_tElem[iElemID].header.iWithID)
    {
        while (i < m_tElem[iElemID].header.iNumElementTypes && !bFlag)
        {
            // perhaps it's easy to find the right point...
            int iFirstID = m_tElem[iElemID].data[i].tElement[0].UserID;
            int iPos = iUserID - iFirstID;

            if (iPos >= 0 && iPos <= m_tElem[iElemID].data[i].dataHeader.iNumElements && m_tElem[iElemID].data[i].tElement[iPos].UserID == iUserID)
            {
                iPosInElemList = iPos;
                bFlag = true;
            }
            else // ...or it's not so easy:-(
            {
                int j = 0;

                while (m_tElem[iElemID].data[i].tElement[j].UserID != iUserID && j < m_tElem[iElemID].data[i].dataHeader.iNumElements)
                {
                    j++;
                }

                if (m_tElem[iElemID].data[i].tElement[j].UserID == iUserID)
                {
                    iPosInElemList = j;
                    bFlag = true;
                }
            }
        }
    }
    else
    {
        iPosInElemList = iUserID - 1;
        bFlag = true;
    }

    if (!bFlag)
        Covise::sendError("can't find PosInElemList");

    return iPosInElemList;
}

//==============================================================================

void ReadVTFData::setPosInPointList(int iNodeID, int iPointID, int iNumPoints)
{
    if (m_tNodes[iNodeID].header.iWithID)
    {
        // perhaps it's easy to find the right point...
        int iFirstID = m_tNodes[iNodeID].point[0].ID;
        int iPos = iPointID - iFirstID;

        if (iPos >= 0 && iPos <= m_tNodes[iNodeID].header.iNumNodes && m_tNodes[iNodeID].point[iPos].ID == iPointID)
        {
            if (m_tNodes[iNodeID].point[iPos].iPosInPointList == -1)
                m_tNodes[iNodeID].point[iPos].iPosInPointList = iNumPoints;
        }
        else // ...or it's not so easy:-(
        {
            int iCount = 0;

            while (m_tNodes[iNodeID].point[iCount].ID != iPointID)
            {
                iCount++;

                if (iCount > m_tNodes[iNodeID].header.iNumNodes)
                {
                    Covise::sendError("can't find place in corner list");
                    break;
                }
            }

            if (m_tNodes[iNodeID].point[iCount].iPosInPointList == -1)
                m_tNodes[iNodeID].point[iCount].iPosInPointList = iNumPoints;
        }
    }
    else
    {
        if (m_tNodes[iNodeID].point[iPointID - 1].iPosInPointList == -1)
            m_tNodes[iNodeID].point[iPointID - 1].iPosInPointList = iNumPoints;
    }
}

//============================================================================

int ReadVTFData::getPosInPointList(int iNodeID, long int iPointID)
{
    int iPos = -1;
    long int iFirstID = 0;

    if (m_tNodes[iNodeID].header.iWithID && !ignoreIDs)
    {
        // perhaps it's easy to find the right point...
        iFirstID = m_tNodes[iNodeID].point[0].ID;
        int iPosID;
        iPosID = iPointID - iFirstID;

        if (iPosID < 0)
        {
            iPosID = -iPosID;
        }

        if (iPosID >= 0 && iPosID <= m_tNodes[iNodeID].header.iNumNodes && m_tNodes[iNodeID].point[iPosID].ID == iPointID)
        {
            iPos = m_tNodes[iNodeID].point[iPosID].iPosInPointList;
        }
        else // ...or it's not so easy:-(
        {
            int i = 0;

            while (m_tNodes[iNodeID].point[i].ID != iPointID)
            {
                i++;

                if (i > m_tNodes[iNodeID].header.iNumNodes)
                {
                    if (firstError)
                    {
                        Covise::sendError("can't get PosInCornerList");
                        firstError = false;
                    }

                    iPos = m_tNodes[iNodeID].point[iPointID - 1].iPosInPointList;
                    return iPos;
                    break;
                }
            }

            iPos = m_tNodes[iNodeID].point[i].iPosInPointList;
        }
    }
    else
    {
        iPos = m_tNodes[iNodeID].point[iPointID - 1].iPosInPointList;
    }

    return iPos;
}

//============================================================================
int ReadVTFData::getPosInPolyList(int iElemBlockID, int iUserID)
{
    int iPosInElemList = -1;
    int i = 0;
    bool bFlag = false;

    if (m_tElem[iElemBlockID].header.iWithID)
    {
        while (i < m_tElem[iElemBlockID].header.iNumElementTypes && !bFlag)
        {
            // perhaps it's easy to find the right point...
            int iFirstID = m_tElem[iElemBlockID].data[i].tElement[0].UserID;
            int iPos = iUserID - iFirstID;

            if (iPos >= 0 && iPos <= m_tElem[iElemBlockID].data[i].dataHeader.iNumElements && m_tElem[iElemBlockID].data[i].tElement[iPos].UserID == iUserID)
            {
                iPosInElemList = m_tElem[iElemBlockID].data[i].tElement[iPos].iPosInPolyList;
                bFlag = true;
            }
            else // ...or it's not so easy:-(
            {
                int j = 0;

                while (m_tElem[iElemBlockID].data[i].tElement[j].UserID != iUserID
                       && j < m_tElem[iElemBlockID].data[i].dataHeader.iNumElements)
                {
                    j++;
                }

                if (m_tElem[iElemBlockID].data[i].tElement[j].UserID == iUserID)
                {
                    iPosInElemList = m_tElem[iElemBlockID].data[i].tElement[j].iPosInPolyList;
                    bFlag = true;
                }
            }
        }
    }
    else
    {
        iPosInElemList = m_tElem[iElemBlockID].data[i].tElement[iUserID - 1].iPosInPolyList;
        bFlag = true;
    }

    if (!bFlag)
        Covise::sendError("can't find PosInElemList");

    return iPosInElemList;
}

//============================================================================

void ReadVTFData::deleteAll()
{
    // last thing to do ...is to delete everything
    // NODES
    int i = 0, j, k;

    for (i = 0; i < m_iNodesCounter; i++)
        delete[] m_tNodes[i].point;

    delete[] m_tNodes;

    //ELEMENT
    for (i = 0; i < m_iElemCounter; i++)
        for (j = 0; j < m_tElem[i].header.iNumElementTypes; j++)
        {
            if (m_tElem[i].data[j].dataHeader.eType != IFS)
            {
                for (k = 0; k < m_tElem[i].data[j].dataHeader.iNumElements; k++)
                    delete[] m_tElem[i].data[j].tElement[k].iNodeID;
            }
        }

    for (i = 0; i < m_iElemCounter; i++)
        for (int j = 0; j < m_tElem[i].header.iNumElementTypes; j++)
            delete[] m_tElem[i].data[j].tElement;

    for (i = 0; i < m_iElemCounter; i++)
        delete[] m_tElem[i].data;

    delete[] m_tElem;

    // RESULTS
    for (i = 0; i < m_iResCounter; i++)
        delete[] m_tRes[i].data;

    delete[] m_tRes;

    // GLVIEWSCALAR
    for (i = 0; i < m_iVSCounter; i++)
        for (j = 0; j < m_tViewS[i].header.iNumSteps; j++)
            delete[] m_tViewS[i].data[j].iResultBlockID;

    for (i = 0; i < m_iVSCounter; i++)
    {
        delete[] m_tViewS[i].data;
        delete[] m_tViewS[i].dataHeader;
    }

    //delete m_tViewS;
    // GLVIEWVECTOR
    for (i = 0; i < m_iVVCounter; i++)
        for (j = 0; j < m_tViewV[i].header.iNumSteps; j++)
            delete[] m_tViewV[i].data[j].iResultBlockID;

    for (i = 0; i < m_iVVCounter; i++)
    {
        delete[] m_tViewV[i].data;
        delete[] m_tViewV[i].dataHeader;
    }

    //delete m_tViewV;
    // GLVIEWGEOMETRY
    for (i = 0; i < m_iGeomCounter; i++)
        for (j = 0; j < m_tGeom[i].header.iNumSteps; j++)
        {
            delete[] m_tGeom[i].data[j].iElemBlockID;
            delete[] m_tGeom[i].data[j].iIFSBlock;
        }

    for (i = 0; i < m_iGeomCounter; i++)
    {
        delete[] m_tGeom[i].dataHeader;
        delete[] m_tGeom[i].data;
    }

    // TRANSFORMATIONS
    for (i = 0; i < m_iTransCounter; i++)
    {
        for (j = 0; j < m_tTrans[i].header.iNumSteps; j++)
        {

            delete[] m_tTrans[i].data[j].iElementBlockID;
            delete[] m_tTrans[i].data[j].iIFSID;

            for (k = 0; k < m_tTrans[i].dataHeader[j].iNumElementBlocks; k++)
            {
                delete[] m_tTrans[i].data[j].iElementTransMatrix[k];
            }

            for (k = 0; k < m_tTrans[i].dataHeader[j].iNumIFSBlocks; k++)
            {
                delete[] m_tTrans[i].data[j].iIFSTransMatrix[k];
            }
        }

        delete[] m_tTrans[i].data;
        delete[] m_tTrans[i].dataHeader;
    }

    //delete m_tGeom;
}
