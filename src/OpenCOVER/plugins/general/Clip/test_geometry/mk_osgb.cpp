// Assembles several OBJ files (photogrammetry "tiles") into ONE top-level
// object:  Group(name) -> [ Geode(tile0), Geode(tile1), ... ]  and writes it
// as .osgb (native compressed) so COVER loads it as a single multi-geode
// object, mimicking a photogrammetry model.
#include <osgDB/ReadFile>
#include <osgDB/WriteFile>
#include <osg/Group>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <iostream>

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "usage: " << argv[0] << " out.osgb name tile0.obj [tile1.obj ...]\n";
        return 1;
    }
    const std::string out = argv[1];
    const std::string name = argv[2];

    osg::ref_ptr<osg::Group> root = new osg::Group();
    root->setName(name);

    int gi = 0;
    int geodeCount = 0;
    for (int i = 3; i < argc; ++i)
    {
        osg::Node *node = osgDB::readNodeFile(argv[i]);
        if (!node)
        {
            std::cerr << "failed to read " << argv[i] << "\n";
            return 1;
        }
        // readNodeFile may return a Group with one geode; extract the geodes
        osg::Group *g = node->asGroup();
        if (g)
        {
            for (unsigned int c = 0; c < g->getNumChildren(); ++c)
            {
                osg::Geode *geode = g->getChild(c)->asGeode();
                if (geode)
                {
                    geode->setName("tile" + std::to_string(gi++));
                    root->addChild(geode);
                    geodeCount++;
                }
            }
        }
        else if (osg::Geode *geode = node->asGeode())
        {
            geode->setName("tile" + std::to_string(gi++));
            root->addChild(geode);
            geodeCount++;
        }
        else
        {
            std::cerr << "unexpected node type in " << argv[i] << ": " << node->className() << "\n";
            return 1;
        }
    }

    bool ok = osgDB::writeNodeFile(*root, out);
    std::cout << (ok ? "OK" : "FAILED") << " " << out << " geodes=" << geodeCount << "\n";
    return ok ? 0 : 1;
}
