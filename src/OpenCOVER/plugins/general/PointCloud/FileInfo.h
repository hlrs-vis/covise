#ifndef POINTCLOUD_FILEINFO_H
#define POINTCLOUD_FILEINFO_H

#include <osg/ref_ptr>
#include <osg/Node>
#include <string>
#include <vector>

struct PointSet;

namespace opencover {
namespace ui {
class Button;
}
}

class NodeInfo
{
public:
    osg::ref_ptr<osg::Geode> node;
};

class FileInfo
{
public:
    std::string filename;
    std::vector<NodeInfo> nodes;
	osg::ref_ptr<osg::MatrixTransform> tranformMat;
    int pointSetSize;
    PointSet *pointSet;
	osg::Matrix prevMat;
	opencover::ui::Button *fileButton = nullptr;
};

#endif
