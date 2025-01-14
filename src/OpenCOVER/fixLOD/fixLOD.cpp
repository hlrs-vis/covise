#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <ogrsf_frmts.h>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ProxyNode>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>


int fixLOD(const std::string& filename)
{

    osg::Node *root=nullptr;
    root = osgDB::readNodeFile(filename.c_str());
    osg::PagedLOD *plod = 
    osgDB::writeNodeFile(*root, filename.c_str());
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 2)
    {
        std::string filename = argv[1];
        return fixLOD(filename);
    }
    else
    {
        std::cerr << "Usage: fixlod [file.osgb]" << std::endl;
    }

    return 0;
}
