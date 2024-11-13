#include "OctUtils.h"
#include <array>
using namespace oct;

osg::Vec3 getNormal(const osg::Vec3Array& vertices, size_t vertexIndex, size_t numPointsPerCycle)
{
    using namespace osg;
    std::array<Vec3, 4> neigbors = {vertexIndex >= 1 ? vertices[vertexIndex - 1] : vertices[vertexIndex],
                                    vertexIndex >= numPointsPerCycle ? vertices[vertexIndex - numPointsPerCycle] : vertices[vertexIndex],
                                    vertexIndex  + 1 < vertices.size() ? vertices[vertexIndex + 1] : vertices[vertexIndex],
                                    vertexIndex + numPointsPerCycle < vertices.size() ? vertices[vertexIndex + numPointsPerCycle] : vertices[vertexIndex]                                  
                                    };
    Vec3 normal;

    for (size_t i = 0; i < neigbors.size(); i++)
    {
        auto last = i == 0 ? 3 : i - 1;
        auto x = vertices[vertexIndex] - neigbors[i] ^ vertices[vertexIndex] - neigbors[last];
        x.normalize();
        normal += x;
    }

    return normal;
}

osg::ref_ptr<osg::Vec3Array> oct::calculateNormals(osg::ref_ptr<osg::Vec3Array> &vertices, size_t numPointsPerCycle)
{
    using namespace osg;
    
    ref_ptr<Vec3Array> normals = new Vec3Array;
    
    for (size_t i = 0; i < vertices->size() - numPointsPerCycle - 1; i++)
        normals->push_back(getNormal(*vertices, i, numPointsPerCycle));
    return normals;
}