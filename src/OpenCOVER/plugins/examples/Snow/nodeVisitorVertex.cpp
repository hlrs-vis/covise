#include "nodeVisitorVertex.h"
#include <osg/MatrixTransform>

nodeVisitorVertex::nodeVisitorVertex() :osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    localScene = new osg::Group;
    localScene->setName("RTCScene_Group");
    rtGeode = new osg::Geode;
    rtGeode->setName("RTScene");

    cover->getObjectsRoot()->addChild(localScene);
    localScene->addChild(rtGeode);

    childTransform.makeIdentity();

    vertexCoords = new osg::Vec3Array;

    blacklist.push_back("1.Name");                      //Names have to be hardcoded here
    blacklist.push_back("2.Name");                      //Names have to be hardcoded here
    blacklist.push_back("3.Name");                      //Names have to be hardcoded here
    blacklist.push_back("RTScene");
}


void nodeVisitorVertex::apply(osg::Node &node)
{
    //    std::clock_t begin = clock();

    if (checkBlacklist(&node))
    {
        return;
    }

    if (node.getName().compare(0, 8, "coNozzle") == 0)
    {
        coNozzleList.push_back(&node);
        return;
    }

    if (auto transform = dynamic_cast<osg::MatrixTransform*>(&node))
    {
        osg::Matrix saved = childTransform;

        childTransform = transform->getMatrix()*childTransform;
        traverse(node);
        childTransform = saved;
        return;
    }
    if (auto geode = dynamic_cast<osg::Geode*>(&node))
    {
        for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            Geometry *geom = geode->getDrawable(i)->asGeometry();
            if (geom)
            {
                osg::TriangleFunctor<nodeVisitTriangle> tfc;
                tfc.setNVV(this);
                geom->accept(tfc);

            }
        }
        return;
    }
    traverse(node);
}





void nodeVisitTriangle::operator()(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3, bool)const
{
    nvv_->fillVertexArray(v1*nvv_->getChildTransform(),
        v2*nvv_->getChildTransform(),
        v3*nvv_->getChildTransform());
    nvv_->numOfVertices += 3;

}

