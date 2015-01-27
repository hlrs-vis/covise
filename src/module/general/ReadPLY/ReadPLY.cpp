/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <do/coDoData.h>
#include <do/coDoPolygons.h>
#include <util/byteswap.h>

#include "ReadPLY.h"

#define LINE_BUFFER 1024
#define DEBUG_LEVEL 0

ReadPLY::ReadPLY(int argc, char *argv[])
    : coModule(argc, argv,
               "Read Stanford PLY files.")
{
    m_currentFileSection = FsUnspecified;
    m_lineCnt = 0;
    m_numVertices = 0;
    m_numFaces = 0;
    m_currentPoly = 0;
    m_numFiles = 0;
    m_numCurrentVertices = 0;
    m_numCurrentFaces = 0;
    m_vertexStructure = new std::vector<int>();
    m_faceStructure = new std::vector<int>();

    m_vx = NULL;
    m_vy = NULL;
    m_vz = NULL;
    m_rgba = NULL;
    m_nvx = NULL;
    m_nvy = NULL;
    m_nvz = NULL;

    m_indices = new std::vector<int>();
    m_polygons = new std::vector<int>();

    // Parameters.
    m_paramFilename = addFileBrowserParam("FilePath",
                                          "Single PLY file or directory that will be scanned for .ply files recursively");
    m_paramFilename->setValue("data/", "*");

    // Ports.
    m_polygonOutPort = addOutputPort("polygons", "Polygons", "geometry polygons");
    m_colorOutPort = addOutputPort("colors", "RGBA", "polygons colors");
    m_normalOutPort = addOutputPort("normals", "Vec3", "polygons normals");
}

ReadPLY::~ReadPLY()
{
    delete m_vertexStructure;
    delete m_faceStructure;

    delete[] m_vx;
    delete[] m_vy;
    delete[] m_vz;
    delete[] m_rgba;
    delete[] m_nvx;
    delete[] m_nvy;
    delete[] m_nvz;

    delete m_indices;
    delete m_polygons;
}

int ReadPLY::compute(const char *)
{
    coDoPolygons *polygonObject;
    coDoVec3 *normalObject;
    //coDoRGBA* colorObject;
    const char *polygonObjectName;
    const char *normalObjectName;
    //const char* colorObjectName;

    clearVertexArrays();
    m_indices->clear();
    m_polygons->clear();

    m_numVertices = 0;
    m_numFaces = 0;
    m_currentPoly = 0;
    m_numFiles = 0;
    m_numCurrentVertices = 0;
    m_numCurrentFaces = 0;

    const char *filename = m_paramFilename->getValue();

    coDirectory *dir = coDirectory::open(filename);

    // If dir is null, a single file rather than a directory was supplied.
    if (!dir)
    {
        if (!processSingleFile(filename))
        {
            return STOP_PIPELINE;
        }
    }
    else
    {
        processDirectory(dir, true);
    }

    // Copy the indices std::vector to a c-style array.
    int *indicesCopy = new int[m_indices->size()];
    copy(m_indices->begin(), m_indices->end(), indicesCopy);

    int *polygonsCopy = new int[m_polygons->size()];
    copy(m_polygons->begin(), m_polygons->end(), polygonsCopy);

    // Supply output ports with data.
    polygonObjectName = m_polygonOutPort->getObjName();
    polygonObject = new coDoPolygons(polygonObjectName, m_numVertices, m_vx, m_vy, m_vz, m_indices->size(),
                                     indicesCopy, m_numFaces, polygonsCopy);
    m_polygonOutPort->setCurrentObject(polygonObject);
    polygonObject->addAttribute("vertexOrder", "2");

    normalObjectName = m_normalOutPort->getObjName();
    normalObject = new coDoVec3(normalObjectName, m_numVertices, m_nvx, m_nvy, m_nvz);
    m_normalOutPort->setCurrentObject(normalObject);

    //colorObjectName = m_colorOutPort->getObjName();
    //colorObject = new coDoRGBA(colorObjectName, m_numVertices, m_rgba);

    delete[] indicesCopy;
    delete[] polygonsCopy;
#if DEBUG_LEVEL >= 1
    std::cout << std::endl;
    std::cout << "Parsed " << m_numFiles << " files" << std::endl;
    std::cout << "Parsed " << m_numVertices << " vertices in total" << std::endl;
    std::cout << "Parsed " << m_numFaces << " polygons in total" << std::endl;
#endif

    return CONTINUE_PIPELINE;
}

bool ReadPLY::processDirectory(coDirectory *dir, const bool recurse)
{
    for (int i = 0; i < dir->count(); ++i)
    {
        if ((strcmp(dir->name(i), ".") == 0) || (strcmp(dir->name(i), "..") == 0))
        {
            continue;
        }

        coDirectory *subDir = coDirectory::open(dir->full_name(i));

        // If subDir is null, it is a single file.
        if (subDir)
        {
            if (recurse)
            {
                processDirectory(subDir, recurse);
            }
        }
        else
        {
            processSingleFile(dir->full_name(i));
        }
    }
    return true;
}

bool ReadPLY::processSingleFile(const char *filename)
{
#if DEBUG_LEVEL >= 1
    std::cout << "Try to open file:" << filename << std::endl;
#endif

    if ((m_fileHandle = fopen(filename, "r")) == NULL)
    {
        sendError("ERROR: Can't open file >> %s", filename);
        return false;
    }

#if DEBUG_LEVEL >= 1
    std::cout << "Opened file" << std::endl;
#endif

    // Reset vectors storing header information.
    m_vertexStructure->clear();
    m_faceStructure->clear();

    // Go for it.
    m_currentFileSection = FsBeforeHeader;
    m_lineCnt = 0;

    char line[LINE_BUFFER];

    // Ascii processing.
    while (fgets(line, LINE_BUFFER, m_fileHandle) != NULL)
    {
        switch (m_currentFileSection)
        {
        case FsBeforeHeader:
            if (!findHeader(line))
            {
                return false;
            }
            break;
        case FsHeader:
            if (!processHeader(line))
            {
                return false;
            }
        default:
            break;
        }
        ++m_lineCnt;

        if (m_currentFileSection > FsHeader)
        {
            // Stop ascii header processing and switch to binary body.
            break;
        }
    }
#if DEBUG_LEVEL >= 1
    std::cout << "Parsed file header" << std::endl;
    std::cout << "Expecting " << m_numCurrentVertices << " vertices and " << m_numCurrentFaces << " polygons" << std::endl;
#endif

    // Binary processing.

    // Vertices.
    float *vx = new float[m_numCurrentVertices];
    float *vy = new float[m_numCurrentVertices];
    float *vz = new float[m_numCurrentVertices];
    // Per vertex colors.
    int *rgba = new int[m_numCurrentVertices];
    // Per vertex normals.
    float *nvx = new float[m_numCurrentVertices];
    float *nvy = new float[m_numCurrentVertices];
    float *nvz = new float[m_numCurrentVertices];
    // Indices references vertices. Polygons determines
    // of which indices a polygon is built up.
    std::vector<int> *indices = new std::vector<int>();
    std::vector<int> *polygons = new std::vector<int>();
    while (true)
    {
        switch (m_currentFileSection)
        {
        case FsVertices:
            if (!processVertices(vx, vy, vz, rgba, nvx, nvy, nvz))
            {
                return false;
            }
            break;
        case FsFaces:
            if (!processFaces(indices, polygons))
            {
                return false;
            }
            break;
        default:
            break;
        }

        // Eof ==> break.
        if (m_currentFileSection == FsUnspecified)
        {
            break;
        }
    }
#if DEBUG_LEVEL >= 1
    std::cout << "Successfully parsed file" << std::endl;
#endif

    // Append vertex data.
    int numVerticesOld = m_numVertices;
    m_numVertices += m_numCurrentVertices;

    concatArrays(m_vx, numVerticesOld, vx, m_numCurrentVertices);
    concatArrays(m_vy, numVerticesOld, vy, m_numCurrentVertices);
    concatArrays(m_vz, numVerticesOld, vz, m_numCurrentVertices);
    concatArrays(m_rgba, numVerticesOld, rgba, m_numCurrentVertices);
    concatArrays(m_nvx, numVerticesOld, nvx, m_numCurrentVertices);
    concatArrays(m_nvy, numVerticesOld, nvy, m_numCurrentVertices);
    concatArrays(m_nvz, numVerticesOld, nvz, m_numCurrentVertices);

    // Append indices and polygons.
    m_numFaces += m_numCurrentFaces;
    std::vector<int>::const_iterator vi;
    for (vi = indices->begin(); vi != indices->end(); ++vi)
    {
        m_indices->push_back(*vi);
    }

    std::vector<int>::const_iterator pi;
    for (pi = polygons->begin(); pi != polygons->end(); ++pi)
    {
        m_polygons->push_back(*pi);
    }
#if DEBUG_LEVEL >= 1
    std::cout << "All data appended" << std::endl;
#endif
    delete[] vx;
    delete[] vy;
    delete[] vz;
    delete[] rgba;
    delete[] nvx;
    delete[] nvy;
    delete[] nvz;
    delete indices;
    delete polygons;

    ++m_numFiles;
    fclose(m_fileHandle);
    return true;
}

bool ReadPLY::findHeader(char *line)
{
    // First line: find 'ply' or die.
    if (m_lineCnt == 0)
    {
        char *ply = new char[4];
        if (sscanf(line, "%s", ply) != 1)
        {
            sendError("ERROR: Invalid file header");
            m_currentFileSection = FsUnspecified;
            delete[] ply;
            return false;
        }

        if (strcmp(ply, "ply") != 0)
        {
            sendError("ERROR: Invalid file header");
            m_currentFileSection = FsUnspecified;
            delete[] ply;
            return false;
        }
        // Success.
        m_currentFileSection = FsHeader;
        delete[] ply;
    }
    return true;
}

bool ReadPLY::processHeader(char *line)
{
    HeaderElementPLY currentElement = HeUnspecified;
    do
    {
        // End section?
        if (strstr(line, "end_header") != NULL)
        {
            m_currentFileSection = FsVertices;
            return true;
        }

        // Comment?
        if (strstr(line, "comment") != NULL)
        {
            // Simply ignore.
            return false;
        }

        // Format: Binary or ascii? Currently the only supported format
        // is binary_big_endian.
        if (strstr(line, "format") != NULL)
        {
            if ((strstr(line, "binary_little_endian") != NULL) || (strstr(line, "ascii") != NULL))
            {
                sendError("ERROR: currently only binary_big_endian supported");
                return false;
            }
        }

        // Element.
        if (strstr(line, "element") != NULL)
        {
            bool recognized = false;
            if (strstr(line, "vertex") != NULL)
            {
                if (sscanf(line, "%*s %*s %d", &m_numCurrentVertices) == 0)
                {
                    sendError("ERROR: no vertex count specified");
                    return false;
                }
                else
                {
                    currentElement = HeVertex;
                    recognized = true;
                }
            }

            if (strstr(line, "face") != NULL)
            {
                if (sscanf(line, "%*s %*s %d", &m_numCurrentFaces) == 0)
                {
                    sendError("ERROR: no face count specified");
                    return false;
                }
                else
                {
                    currentElement = HeFace;
                    recognized = true;
                }
            }

            if (!recognized)
            {
                currentElement = HeUnspecified;
            }
        }

        // Property.
        if (strstr(line, "property") != NULL)
        {
            switch (currentElement)
            {
            case HeVertex:
                m_vertexStructure->push_back(parseDatatypeLengthFromProperty(line));
                break;
            case HeFace:
                if (strstr(line, "list"))
                {
                    // TODO: really parse these values.
                    m_vertexIndicesStructure[0] = 1;
                    m_vertexIndicesStructure[1] = 4;
                }
                else
                {
                    m_faceStructure->push_back(parseDatatypeLengthFromProperty(line));
                }
                break;
            default:
                break;
            }
        }
    } while (fgets(line, LINE_BUFFER, m_fileHandle) != NULL);
    return true;
}

bool ReadPLY::processVertices(float *&vx, float *&vy, float *&vz,
                              int *&rgba,
                              float *&nvx, float *&nvy, float *&nvz)
{
    VertexStructPLY ident = identifyVertexStructure();

    if (ident == VsUnspecified)
    {
        sendError("ERROR: format unrecognized.");
        return false;
    }

    if (ident == VsFloatXyzUcharRgbFloatNxNyNz)
    {
        for (int i = 0; i < m_numCurrentVertices; ++i)
        {
            float x, y, z;
            unsigned char r, g, b, a;
            float nx, ny, nz;

            unsigned char *bx = new unsigned char[4];
            unsigned char *by = new unsigned char[4];
            unsigned char *bz = new unsigned char[4];

            unsigned char *bnx = new unsigned char[4];
            unsigned char *bny = new unsigned char[4];
            unsigned char *bnz = new unsigned char[4];

            int chksum = 0;
            chksum += fread(bx, 4, 1, m_fileHandle);
            chksum += fread(by, 4, 1, m_fileHandle);
            chksum += fread(bz, 4, 1, m_fileHandle);

            chksum += fread(&r, 1, 1, m_fileHandle);
            chksum += fread(&g, 1, 1, m_fileHandle);
            chksum += fread(&b, 1, 1, m_fileHandle);
            a = static_cast<unsigned char>(255);

            chksum += fread(bnx, 4, 1, m_fileHandle);
            chksum += fread(bny, 4, 1, m_fileHandle);
            chksum += fread(bnz, 4, 1, m_fileHandle);

// toIEEE754 expects big endian bytes! Thus swap although swap isn't defined!
#ifndef BYTESWAP
            swapArray4(bx);
            swapArray4(by);
            swapArray4(bz);

            swapArray4(bnx);
            swapArray4(bny);
            swapArray4(bnz);
#endif
            x = toIEEE754(bx);
            y = toIEEE754(by);
            z = toIEEE754(bz);

            nx = toIEEE754(bnx);
            ny = toIEEE754(bny);
            nz = toIEEE754(bnz);

            delete[] bx;
            delete[] by;
            delete[] bz;
            delete[] bnx;
            delete[] bny;
            delete[] bnz;
#if DEBUG_LEVEL >= 2
            std::cout << x << " " << y << " " << z << std::endl;
            std::cout << (int)r << " " << (int)g << " " << (int)b << std::endl;
            std::cout << nx << " " << ny << " " << nz << std::endl;
#endif
            vx[i] = x;
            vy[i] = y;
            vz[i] = z;

            // Combine the rgba color components to a packed 32 bit color.
            rgba[i] = (r << 24) | (g << 16) | (b << 8) | a;

            nvx[i] = nx;
            nvy[i] = ny;
            nvz[i] = nz;
        }
    }
    m_currentFileSection = FsFaces;
    return true;
}

bool ReadPLY::processFaces(std::vector<int> *indices, std::vector<int> *polygons)
{
    FaceStructPLY ident = identifyFaceStructure();

    if (ident == FcUnspecified)
    {
        sendError("ERROR: format unrecognized.");
        return false;
    }

    if (ident == FcUcharRgb)
    {
        int chksum = 0;
        for (int i = 0; i < m_numCurrentFaces; ++i)
        {
            // Determine index count for this face.
            unsigned char idxCnt;
            chksum += fread(&idxCnt, 1, 1, m_fileHandle);

            for (int j = 0; j < idxCnt; ++j)
            {
                unsigned char *tmp = new unsigned char[4];
                unsigned int idx;
                chksum += fread(&idx, 4, 1, m_fileHandle);

#ifdef BYTESWAP
                byteSwap(idx);
#endif
                delete[] tmp;
                // When concatenating files, the index has to be updated according to the current idx count.
                indices->push_back(m_numVertices + static_cast<int>(idx));
            }
            // Store the index count of the polygon.
            m_currentPoly += static_cast<int>(idxCnt);
            polygons->push_back(m_currentPoly);
#if DEBUG_LEVEL >= 2
            std::cout << "Parsed polygon with " << static_cast<int>(idxCnt) << " indices" << std::endl;
#endif
            unsigned char r, g, b;

            chksum += fread(&r, 1, 1, m_fileHandle);
            chksum += fread(&g, 1, 1, m_fileHandle);
            chksum += fread(&b, 1, 1, m_fileHandle);
        }
    }
    m_currentFileSection = FsUnspecified;
    return true;
}

int ReadPLY::parseDatatypeLengthFromProperty(char *line)
{
    int result = -1;

    if ((strstr(line, "char") != NULL) || (strstr(line, "uchar") != NULL))
    {
        result = 1;
    }

    if ((strstr(line, "short") != NULL) || (strstr(line, "ushort") != NULL))
    {
        result = 2;
    }

    if ((strstr(line, "int") != NULL) || (strstr(line, "uint") != NULL)
        || (strstr(line, "float") != NULL))
    {
        result = 4;
    }

    if (strstr(line, "double") != NULL)
    {
        result = 8;
    }

    return result;
}

int ReadPLY::getVertexSize() const
{
    int result = 0;
    std::vector<int>::const_iterator i;

    for (i = m_vertexStructure->begin(); i != m_vertexStructure->end(); ++i)
    {
        result += (*i);
    }
    return result;
}

VertexStructPLY ReadPLY::identifyVertexStructure() const
{
    VertexStructPLY result = VsUnspecified;

    if (m_vertexStructure->size() == 9)
    {
        if ((m_vertexStructure->at(0) == 4) && (m_vertexStructure->at(1) == 4)
            && (m_vertexStructure->at(2) == 4)
            && (m_vertexStructure->at(3) == 1) && (m_vertexStructure->at(4) == 1)
            && (m_vertexStructure->at(5) == 1)
            && (m_vertexStructure->at(6) == 4) && (m_vertexStructure->at(7) == 4)
            && (m_vertexStructure->at(8) == 4))
        {
            result = VsFloatXyzUcharRgbFloatNxNyNz;
        }
    }

    return result;
}

FaceStructPLY ReadPLY::identifyFaceStructure() const
{
    FaceStructPLY result = FcUnspecified;

    if (m_faceStructure->size() == 3)
    {
        if ((m_faceStructure->at(0) == 1) && (m_faceStructure->at(1) == 1)
            && (m_faceStructure->at(2) == 1))
        {
            result = FcUcharRgb;
        }
    }

    return result;
}

float ReadPLY::toIEEE754(unsigned char *bytes)
{
    // IEEE-754 convertion taken (courtesy) from
    // Stefan Zellmann's 3D renderer.
    float f;

    if ((static_cast<int>(bytes[0]) == 0)
        && (static_cast<int>(bytes[1]) == 0)
        && (static_cast<int>(bytes[2]) == 0)
        && (static_cast<int>(bytes[3]) == 0))
    {
        f = 0.0f;
        return f;
    }

    float sign = 1.0f;
    if (bytes[0] & 0x80)
    {
        bytes[0] -= 0x80;
        sign = -1.0f;
    }

    float exp = bytes[0] << 1;
    if (bytes[1] & 0x80)
    {
        bytes[1] -= 0x80;
        exp += 1.0f;
    }
    exp -= 127;

    float mantissa = 0.0f;

    mantissa += ((bytes[1] & 0x40) >> 6) * 0.5f;
    mantissa += ((bytes[1] & 0x20) >> 5) * 0.25f;
    mantissa += ((bytes[1] & 0x10) >> 4) * 0.125f;
    mantissa += ((bytes[1] & 0x8) >> 3) * 0.0625f;
    mantissa += ((bytes[1] & 0x4) >> 2) * 0.03125f;
    mantissa += ((bytes[1] & 0x2) >> 1) * 0.015625f;
    mantissa += (bytes[1] & 0x1) * 0.0078125f;

    mantissa += ((bytes[2] & 0x80) >> 7) * 0.00390625f;
    mantissa += ((bytes[2] & 0x40) >> 6) * 0.001953125f;
    mantissa += ((bytes[2] & 0x20) >> 5) * 0.0009765625f;
    mantissa += ((bytes[2] & 0x10) >> 4) * 0.00048828125f;
    mantissa += ((bytes[2] & 0x8) >> 3) * 0.000244140625f;
    mantissa += ((bytes[2] & 0x4) >> 2) * 0.0001220703125f;
    mantissa += ((bytes[2] & 0x2) >> 1) * 0.00006103515625;
    mantissa += (bytes[2] & 0x1) * 0.000030517578125;

    mantissa += ((bytes[3] & 0x80) >> 7) * 0.0000152587890625f;
    mantissa += ((bytes[3] & 0x40) >> 6) * 0.00000762939453125f;
    mantissa += ((bytes[3] & 0x20) >> 5) * 0.000003814697265625f;
    mantissa += ((bytes[3] & 0x10) >> 4) * 0.0000019073486328125f;
    mantissa += ((bytes[3] & 0x8) >> 3) * 0.00000095367431640625f;
    mantissa += ((bytes[3] & 0x4) >> 2) * 0.00000047683715820312f;
    mantissa += ((bytes[3] & 0x2) >> 1) * 0.00000023841857910156f;
    mantissa += (bytes[3] & 0x1) * 0.00000011920928955078f;

    f = sign * (1.0f + mantissa) * static_cast<float>(pow(2.0f, exp));

    return f;
}

int ReadPLY::uchar4ToInt(unsigned char *bytes)
{
    int result = 0;

    for (int i = 0; i < 4; ++i)
    {
        result += static_cast<int>(bytes[i]) * (int)pow(16.0f, i);
    }

    return result;
}

void ReadPLY::swapArray4(unsigned char *&bytes)
{
    unsigned char tmp[4];
    for (int i = 0; i < 4; ++i)
    {
        tmp[i] = bytes[i];
    }

    for (int i = 0; i < 4; ++i)
    {
        bytes[3 - i] = tmp[i];
    }
}

void ReadPLY::concatArrays(int *&dest, const int numDest, int *src, const int numSrc)
{
    int *tmp = new int[numDest + numSrc];

    int iterator = 0;
    for (int i = 0; i < numDest; ++i)
    {
        tmp[iterator] = dest[i];
        ++iterator;
    }

    for (int i = 0; i < numSrc; ++i)
    {
        tmp[iterator] = src[i];
        ++iterator;
    }

    delete[] dest;
    dest = tmp;
}

void ReadPLY::concatArrays(float *&dest, const int numDest, float *src, const int numSrc)
{
    float *tmp = new float[numDest + numSrc];

    int iterator = 0;
    for (int i = 0; i < numDest; ++i)
    {
        tmp[iterator] = dest[i];
        ++iterator;
    }

    for (int i = 0; i < numSrc; ++i)
    {
        tmp[iterator] = src[i];
        ++iterator;
    }

    delete[] dest;
    dest = tmp;
}

void ReadPLY::clearVertexArrays()
{
    delete[] m_vx;
    delete[] m_vy;
    delete[] m_vz;
    delete[] m_rgba;
    delete[] m_nvx;
    delete[] m_nvy;
    delete[] m_nvz;

    m_vx = NULL;
    m_vy = NULL;
    m_vz = NULL;
    m_rgba = NULL;
    m_nvx = NULL;
    m_nvy = NULL;
    m_nvz = NULL;
}

MODULE_MAIN(IO, ReadPLY)
