#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <osg/MatrixTransform>
#include <osg/Geode>
#include <osg/ProxyNode>
#include <osgDB/Registry>
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osgDB/FileUtils>
#include <osgDB/FileNameUtils>
#include <osg/PagedLOD>
#include <filesystem>


int fixLOD(const std::string& filename)
{

    osgDB::Options *options= new osgDB::Options;

    options->setOptionString("WriteImageHint=IncludeFile");
    osg::Node *root=nullptr;
    std::filesystem::path p;
    p = filename;
    if(!p.parent_path().empty())
        std::filesystem::current_path(p.parent_path());
    std::string fn = p.filename().string();
    root = osgDB::readNodeFile(fn.c_str());
    osg::PagedLOD* plod = dynamic_cast<osg::PagedLOD*>(root);
    if (plod)
    {
        plod->setRangeMode(osg::LOD::RangeMode::DISTANCE_FROM_EYE_POINT);
        auto ranges = plod->getRangeList();
        osg::LOD::RangeList nrl;
        for (int i = (int)ranges.size() - 1; i >= 0; i--)
        {
            nrl.push_back(ranges[i]);
        }
        nrl[0].first = 4*plod->getRadius();
        nrl[0].second = 1.0E10;
        nrl[1].first = 0.0;
        nrl[1].second = 4*plod->getRadius();
        plod->setRangeList(nrl);
        osgDB::writeNodeFile(*root, fn.c_str(), options);
    }
    else
    {
        osg::Group* g = dynamic_cast<osg::Group*>(root);
        if (g)
        {
            bool found = true;
            for (size_t i = 0; i < g->getNumChildren(); i++)
            {
                osg::Node* node = g->getChild(i); 
                osg::PagedLOD* plod = dynamic_cast<osg::PagedLOD*>(node);
                if (plod)
                {
                    plod->setRangeMode(osg::LOD::RangeMode::DISTANCE_FROM_EYE_POINT);
                    auto ranges = plod->getRangeList();
                    osg::LOD::RangeList nrl;
                    for (int i = (int)ranges.size() - 1; i >= 0; i--)
                    {
                        nrl.push_back(ranges[i]);
                    }
                    nrl[0].first = plod->getRadius();
                    nrl[0].second = 1.0E10;
                    nrl[1].first = 0.0;
                    nrl[1].second = plod->getRadius();
                    plod->setRangeList(nrl);
                    found = true;
                }

            }
            if (found)
            {
                osgDB::writeNodeFile(*root, fn.c_str(), options);
            }
            else
            {
                fprintf(stderr, "no pagedLOD %s\n", filename.c_str());
            }
        }
        else
        {
            fprintf(stderr, "no pagedLOD %s\n", filename.c_str());
        }
    }
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
