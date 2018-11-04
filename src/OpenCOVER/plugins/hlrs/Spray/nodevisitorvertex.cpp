#include "nodevisitorvertex.h"

nodeVisitorVertex::nodeVisitorVertex():osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    localScene = new osg::Group;
    localScene->setName("RTCScene_Group");
    rtGeode = new osg::Geode;
    rtGeode->setName("RTScene");

    if(parser::instance()->getDisplayRTScene() == 1)
        displayRTScene = true;
    else
        displayRTScene = false;

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

    if(checkBlacklist(&node))
    {
        return;
    }

    if(node.getName().compare(0,8,"coNozzle") == 0)
    {
        coNozzleList.push_back(&node);
        return;
    }

    if(auto transform = dynamic_cast<osg::MatrixTransform*>(&node))
    {
        osg::Matrix saved = childTransform;

        childTransform = transform->getMatrix()*childTransform;
        traverse(node);
        childTransform = saved;
        return;
    }
    if(auto geode = dynamic_cast<osg::Geode*>(&node))
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
    nvv_->numOfVertices +=3;

    if(nvv_->displayRTScene)
    {
        osg::Geometry *geom = new osg::Geometry;

        osg::Vec3Array *vertices = new osg::Vec3Array;

        vertices->push_back(v1*nvv_->getChildTransform());
        vertices->push_back(v2*nvv_->getChildTransform());
        vertices->push_back(v3*nvv_->getChildTransform());

        geom->setVertexArray(vertices);

        osg::Vec4Array *colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(1,0,0,1));
        geom->setColorArray(colors);
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);

        osg::Vec3Array *normals = new osg::Vec3Array;
        normals->push_back(osg::Vec3(0,0,-1));
        geom->setNormalArray(normals);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

        geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,3));


        nvv_->returnGeode()->addDrawable(geom);
    }
}

