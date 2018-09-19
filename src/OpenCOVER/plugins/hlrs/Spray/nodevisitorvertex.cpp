#include "nodevisitorvertex.h"

nodeVisitorVertex::nodeVisitorVertex():osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    localScene = new osg::Group;
    localGeodeTriangle = new osg::Geode;
    localGeodeTriangle->setName("nopeTriangle");

    localScene->setName("test");
    cover->getObjectsRoot()->addChild(localScene);
    localScene->addChild(localGeodeTriangle);

    childTransform.makeIdentity();

    vertexCoords = new osg::Vec3Array;

    blacklist.push_back("1.Name");
    blacklist.push_back("2.Name");
    blacklist.push_back("3.Name");
    blacklist.push_back("nopeTriangle");
}


void nodeVisitorVertex::apply(osg::Node &node)
{    
    //    std::clock_t begin = clock();

    //std::cout << "Name of node " << node.getName() << std::endl;
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
        //std::clock_t begin = clock();
        std::cout << "Name of node " << node.getName() << std::endl;
        osg::Matrix saved = childTransform;

        childTransform = transform->getMatrix()*childTransform;
        traverse(node);
        childTransform = saved;
        return;

        //                std::clock_t end = clock();
        //                double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

        //                printf("elapsed time for mt %f\n", elapsed_secs);

    }
    if(auto geode = dynamic_cast<osg::Geode*>(&node))
    {
        //std::cout << "Name of node " << geode->getName() << std::endl;
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

    //    std::clock_t end = clock();
    //    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;

    //    std::cout << "elapsed time for traversing a node " << elapsed_secs << " name of node " << node.getName() << std::endl;

    traverse(node);
}


void nodeVisitorVertex::createTestFaces(int num, osg::Vec3Array::iterator coords, int type)
{
    osg::Geometry *geom = new osg::Geometry;

    osg::Vec3Array *vertices = new osg::Vec3Array;
    for(int it = 0; it < num; it++)
    {
        vertices->push_back(*coords);
        coords++;
    }

    geom->setVertexArray(vertices);

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(1,0,0,1));
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0,0,-1));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

    geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,num));


    localGeodeTriangle->addDrawable(geom);
}

void nodeVisitorVertex::createFaceSet(Vec3Array *coords, int type)
{
    osg::Geometry *geom = new osg::Geometry;

    if(visualize)
    {
        geom->setVertexArray(coords);

        osg::Vec4Array *colors = new osg::Vec4Array;
        colors->push_back(osg::Vec4(1,0,0,1));
        geom->setColorArray(colors);
        geom->setColorBinding(osg::Geometry::BIND_OVERALL);

        osg::Vec3Array *normals = new osg::Vec3Array;
        normals->push_back(osg::Vec3(0,0,-1));
        geom->setNormalArray(normals);
        geom->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);

        if(type == 0)
            geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::TRIANGLES,0,coords->size()));
        if(type == 1)
            geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::QUADS,0,coords->size()));


        localGeodeTriangle->addDrawable(geom);
    }
}

void nodeVisitTriangle::operator()(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3, bool)const
{
    nvv_->fillVertexArray(v1*nvv_->getChildTransform(),
                          v2*nvv_->getChildTransform(),
                          v3*nvv_->getChildTransform());
    nvv_->numOfVertices +=3;
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

void nodeVisitorVertex::_printPrimitiveType(osg::PrimitiveSet *pset)
{
    std::cout << (pset->getMode() == PrimitiveSet::POINTS ? "POINTS" :
                                                            pset->getMode() == PrimitiveSet::LINES ? "LINES" :
                                                                                                     pset->getMode() == PrimitiveSet::LINE_STRIP ? "LINE_STRIP" :
                                                                                                                                                   pset->getMode() == PrimitiveSet::LINE_LOOP ? "LINE_LOOP" :
                                                                                                                                                                                                pset->getMode() == PrimitiveSet::TRIANGLES ? "TRIANGLES" :
                                                                                                                                                                                                                                             pset->getMode() == PrimitiveSet::TRIANGLE_STRIP ? "TRIANGLE_STRIP" :
                                                                                                                                                                                                                                                                                               pset->getMode() == PrimitiveSet::TRIANGLE_FAN ? "TRIANGLE_FAN" :
                                                                                                                                                                                                                                                                                                                                               pset->getMode() == PrimitiveSet::QUADS ? "QUADS" :
                                                                                                                                                                                                                                                                                                                                                                                        pset->getMode() == PrimitiveSet::QUAD_STRIP ? "QUAD_STRIP" :
                                                                                                                                                                                                                                                                                                                                                                                                                                      pset->getMode() == PrimitiveSet::POLYGON ? "POLYGON" :
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 "Unknown") << std::endl;
}
