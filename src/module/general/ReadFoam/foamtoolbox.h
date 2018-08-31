/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FOAMTOOLBOX_H
#define FOAMTOOLBOX_H

#include <cstdlib>
#include <istream>
#include <vector>
#include <string>
#include <map>
#include <limits>

#include <memory>

// this header should contain typedefs for index_t and scalar_t
#include "foamtypes.h"

#if __cplusplus >= 201103L
#include <unordered_set>
typedef std::unordered_set<index_t> vertex_set;
#else
#include <set>
typedef std::set<index_t> vertex_set;
#endif

namespace fs {
class Model;
}

struct HeaderInfo
{
    std::string header;

    std::string version;
    std::string format;
    std::string fieldclass;
    std::string arch;
    std::string note;
    std::string location;
    std::string object;
    std::string dimensions;
    std::string internalField;
    index_t lines = 0;

    int numbits = 32;
    bool valid = false;
};

struct DimensionInfo
{
    DimensionInfo()
        : points(0)
        , cells(0)
        , faces(0)
        , internalFaces(0)
        , valid(false)
    {
    }
    index_t points;
    index_t cells;
    index_t faces;
    index_t internalFaces;
    bool valid;
};

class Boundary
{
public:
    Boundary(const std::string &name, const index_t s, const index_t num, const std::string &t, const index_t &ind)
        : name(name)
        , startFace(s)
        , numFaces(num)
        , type(t)
        , index(ind)
        , myProc(-1)
        , neighborProc(-1)
    {
    }

    std::string name;
    index_t startFace;
    index_t numFaces;
    std::string type;
    index_t index;
    int myProc;
    int neighborProc;
    std::vector<index_t> ghostVertices;
    std::vector<index_t> owner;
};

class Boundaries
{
public:
    Boundaries()
        : valid(false)
    {
    }

    void addBoundary(const Boundary &b)
    {

        if (b.type == "processor")
        {
            procboundaries.push_back(b);
        }
        else
        {
            boundaries.push_back(b);
        }
    }

    int findBoundaryIndexByName(const std::string &b)
    {
        int result = -1;
        for (size_t i = 0; i < boundaries.size(); ++i)
        {
            if (!b.compare(boundaries[i].name))
            {
                result = (int)i;
                break;
            }
        }
        return result;
    }

    int findBoundaryIndexForProc(int proc)
    {
        for (size_t i = 0; i < boundaries.size(); ++i)
        {
            const Boundary &b = procboundaries[i];
            if (b.neighborProc == proc)
                return (int)i;
        }
        return -1;
    }

    bool valid;

    std::vector<Boundary> boundaries;
    std::vector<Boundary> procboundaries;
};


struct CaseInfo
{

    CaseInfo()
        : numblocks(0)
        , varyingGrid(false)
        , varyingCoords(false)
        , hasParticles(false)
        , valid(false)
    {
    }

    std::string casedir;
    std::map<double, std::string> timedirs; //< Map of all the Time Directories
    std::map<double, std::string> completeMeshDirs; //< Map of most recent directory containing the full mesh (neighbour, owner, faces, points)
    std::map<std::string, int> varyingFields, constantFields, particleFields; //< name of all fields together with how often they appear
    std::string constantdir;
    std::string lagrangiandir; //< subdirectory of "lagrangian" which is used for particle data
    int numblocks;
    bool varyingGrid, varyingCoords;
    bool hasParticles = false;
    bool valid = false;
    bool archived = false;

    std::shared_ptr<std::istream> getStreamForFile(const std::string &base, const std::string &filename);
    Boundaries loadBoundary(const std::string &meshdir);

    std::map<int, std::shared_ptr<fs::Model>> archives;
};

CaseInfo getCaseInfo(const std::string &casedir, bool exact = false, bool verbose = false);
HeaderInfo readFoamHeader(std::istream &stream);
DimensionInfo parseDimensions(std::string header);

bool readIndexArray(const HeaderInfo &info, std::istream &stream, index_t *p, const size_t lines);
bool readIndexListArray(const HeaderInfo &info, std::istream &stream, std::vector<index_t> *p, const size_t lines);
bool readFloatArray(const HeaderInfo &info, std::istream &stream, scalar_t *p, const size_t lines);
bool readFloatVectorArray(const HeaderInfo &info, std::istream &stream, scalar_t *x, scalar_t *y, scalar_t *z, const size_t lines);
bool readParticleArray(const HeaderInfo &info, std::istream &stream, scalar_t *x, scalar_t *y, scalar_t *z, index_t *cell, const size_t lines);

index_t findVertexAlongEdge(const index_t point,
                            const index_t homeface,
                            const std::vector<index_t> &cellfaces,
                            const std::vector<std::vector<index_t> > &faces);
bool isPointingInwards(index_t face,
                       index_t cell,
                       index_t ninternalFaces,
                       const std::vector<index_t> &owners,
                       const std::vector<index_t> &neighbors);
vertex_set getVerticesForCell(const std::vector<index_t> &cellfaces,
                                        const std::vector<std::vector<index_t> > &faces);

#endif
