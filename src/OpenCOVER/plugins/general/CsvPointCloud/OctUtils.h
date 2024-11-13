#include <osg/Vec3>
#include <osg/Array>
#include <osg/ref_ptr>
namespace oct{

osg::ref_ptr<osg::Vec3Array> calculateNormals(osg::ref_ptr<osg::Vec3Array> &vertices, size_t numPointsPerCycle);


}