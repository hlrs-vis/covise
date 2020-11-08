/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
    \brief  read Stanford PLY data

    \author Stefan Zellmann <zellmans@uni-koeln.de>
    \author (C) 2009 ZAIK, University of Cologne, Germany

    \data   08.10.2009
 */

#ifndef READPLY_H
#define READPLY_H

#include <api/coModule.h>
#include <util/coFileUtil.h>
using namespace covise;

enum VertexStructPLY
{
    VsFloatXyzUcharRgbFloatNxNyNz = 0,
    VsUnspecified
};

enum FaceStructPLY
{
    FcUcharRgb = 0,
    FcUnspecified
};

class ReadPLY : public coModule
{
public:
    ReadPLY(int argc, char *argv[]);
    virtual ~ReadPLY();

private:
    // Types.
    enum FileSectionPLY
    {
        FsBeforeHeader = 0,
        FsHeader,
        FsVertices,
        FsFaces,
        FsUnspecified
    };

    enum HeaderElementPLY
    {
        HeVertex = 0,
        HeFace,
        HeUnspecified
    };

    FILE *m_fileHandle;
    FileSectionPLY m_currentFileSection;
    int m_lineCnt;

    // Model specific parameters.
    int m_numCurrentVertices;
    int m_numCurrentFaces;
    int m_currentPoly;

    int m_numFiles;
    int m_numVertices;
    int m_numFaces;

    // Data structures for vertices, colors and normals.
    float *m_vx;
    float *m_vy;
    float *m_vz;

    int *m_rgba;

    float *m_nvx;
    float *m_nvy;
    float *m_nvz;

    // Data structures for vertex and polygon indices.
    std::vector<int> *m_indices;
    std::vector<int> *m_polygons;

    /*! \brief      Storage template for the vertex structure.

                  Vertices may be specified quite flexible. E.g. one
                  can declare vertices as follows:
                  float x, y, z + uchar red, green, blue + float nx, ny, nz.
                  This would result in the following vertex structure array:<br>
                  4|4|4|1|1|1|4|4|4<br>
                  Of course only structures that are recognized can be converted
                  into something useful. The above structure e.g. would be interpreted
                  so that x, y and z are the vertex coordinates while rgb are per vertex
                  colors and nx, ny and nz the components of the face normal.
   */
    std::vector<int> *m_vertexStructure;
    /*! \brief      Description for the vertex index list.

                  The first element tells how long the descriptor for the vertex / face
                  count is. The second element tells how long one single index is.
   */
    int m_vertexIndicesStructure[2];
    /*! \brief      Storage template for the face structure.

                  For a more in-deep description see documentation for \ref m_vertexStructure.
                  Note that the size of the vertex index list is already stored in
                  \ref m_vertexIndicesStructure and thus isn't stored here again.
   */
    std::vector<int> *m_faceStructure;

    // Parameters.
    coFileBrowserParam *m_paramFilename;

    // Ports.
    coOutputPort *m_polygonOutPort;
    coOutputPort *m_colorOutPort;
    coOutputPort *m_normalOutPort;

    // Methods.
    virtual int compute(const char *port);

    /*! \brief      Process files in the directory.

                  Process all .ply files in the specified directory.
                  If recurse is true, child directories are traversed.
      \param      dir The directory to parse.
      \param      recurse If true, dir is traversed recursively.
      \return     True on success, false otherwise.
   */
    bool processDirectory(coDirectory *dir, const bool recurse = true);
    /*! \brief      Process one single file.

                  Parse a single .ply file.
      \param      filename Full path to the file, including its full name.
      \return     True on success, false otherwise.
   */
    bool processSingleFile(const char *filename);
    /*! \brief      Find header and invalidate correctness.

                  Ply files start with an ascii header that
                  is followed by data that is either in
                  binary or ascii format. This method
                  searches the first line of the file for
                  the string "ply\n" that indicates the
                  start of the header.
      \return     True on success, false otherwise.
   */
    bool findHeader(char *line);
    /*! \brief      Process .ply file header.

                  Process the ascii file header of ply
                  files and determine model parameters
                  from that.
      \return     True on success, false otherwise.
   */
    bool processHeader(char *line);
    /*! \brief      Parse .ply file for vertices.

                  Parse .ply file for the amount of vertices
                  that is expected due to the header info.
      \param      vx A manipulatable array of vertex-x components.
      \param      vy A manipulatable array of vertex-y components.
      \param      vz A manipulatable array of vertex-z components.
      \param      rgba A manipulatable array of packed 32-bit per vertex rgba components.
      \param      nvx A manipulatable array of per vertex normal-x components.
      \param      nvy A manipulatable array of per vertex normal-y components.
      \param      nvz A manipulatable array of per vertex normal-z components.
      \return     True on success, false otherwise.
   */
    bool processVertices(float *&vx, float *&vy, float *&vz,
                         int *&rgba,
                         float *&nvx, float *&nvy, float *&nvz);
    bool processFaces(std::vector<int> *indices, std::vector<int> *polygons);
    /*! \brief      Return byte count of a datatype.

                  This method parses the line for one of the following types
                  and returns the byte count standing to the right of the
                  identifier:<br>
                  <pre>
                  char            1<br>
                  uchar           1<br>
                  short           2<br>
                  ushort          2<br>
                  int             4<br>
                  uint            4<br>
                  float           4<br>
                  double          8<br>
                  </pre>

      \param      line The ascii line to parse.
      \return     Length in byte.
   */
    int parseDatatypeLengthFromProperty(char *line);
    /*! \brief      Get size of a single vertex in byte.

                  Based upon the information provided by the .ply
                  file, parsed vertices can have different length.
                  E.g. if per vertex normals or colors are stored,
                  data structures will have to accomodate more data.
      \return     Byte size of a vertex.
   */
    int getVertexSize() const;
    /*! \brief      Identify the vertex specification.

                  Based on the header information, it is checked if the
                  vertex format is recognized and thus can be parsed.
   */
    VertexStructPLY identifyVertexStructure() const;
    /*! \brief      Identify the face specification.

                  Based on the header information, it is checked if the
                  face format is recognized and thus can be parsed.
   */
    FaceStructPLY identifyFaceStructure() const;
    /*! \brief      Convert BIG ENDIAN to float.

                  Convert 4 bytes in big endian order to float. Code
                  taken with friendly permission (all rights reserved)
                  from Stefan Zellmann's 3D renderer.
   */
    float toIEEE754(unsigned char *bytes);
    int uchar4ToInt(unsigned char *bytes);
    void swapArray4(unsigned char *&bytes);
    void concatArrays(int *&dest, const int numDest, int *src, const int numSrc);
    void concatArrays(float *&dest, const int numDest, float *src, const int numSrc);
    void clearVertexArrays();
};

#endif
