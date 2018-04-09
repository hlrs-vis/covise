#ifndef POINTCLOUD_FILEINFO_H
#define POINTCLOUD_FILEINFO_H

class NodeInfo
{
public:
    osg::Node *node;
};

class FileInfo
{
public:
    std::string filename;
    std::vector<NodeInfo> nodes;
    int pointSetSize;
    PointSet *pointSet;
};

#endif
