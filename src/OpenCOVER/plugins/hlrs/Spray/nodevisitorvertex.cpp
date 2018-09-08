#include "nodevisitorvertex.h"

nodeVisitorVertex::nodeVisitorVertex():osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{
    localScene = new osg::Group;
    localGeodeTriangle = new osg::Geode;
    localGeodeTriangleStrip = new osg::Geode;
    localGeodeTriangle->setName("nopeTriangle");
    localGeodeTriangleStrip->setName("nopeTriangleStrip");

    localScene->setName("test");
    cover->getObjectsRoot()->addChild(localScene);
    localScene->addChild(localGeodeTriangle);
    localScene->addChild(localGeodeTriangleStrip);

    vertexCoords = new osg::Vec3Array;

    blacklist.push_back("1.Name");
    blacklist.push_back("2.Name");
    blacklist.push_back("3.Name");
}

void nodeVisitorVertex::apply(osg::Node &node)
{
    if(checkBlacklist(&node))
    {
        traverse(node);
    }
    else
    if(node.getName().compare(0,8,"coNozzle") == 0)
    {
        coNozzleList.push_back(&node);
        traverse(node);
    }
else
    if (auto geode = dynamic_cast<osg::Geode *>(&node))
    {
        if(geode->getName().compare("nopeTriangle") == 0 || geode->getName().compare("nopeTriangleStrip") == 0)
            traverse(node);
        else

            for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
            {
                Geometry *geom = geode->getDrawable(i)->asGeometry();
                if (geom)
                {
                    //printf("Looking for Geometry in Geode: %s\n", geode->getName().c_str());
                    int numOfTriangles = 0;
                    int numOfQuads = 0;
                    int numOfTriangleStips = 0;
                    Vec3Array *coords = dynamic_cast<Vec3Array *>(geom->getVertexArray());
                    if (coords == 0L)
                    {
                        printf("No coords\n");
                        continue;
                    }

                    printf("%i\n", coords->getNumElements());

                    if(triFunc)
                    {
                        osg::TriangleFunctor<nodeVisitTriangle> tfc;
                        tfc.setNVV(this);
                        geom->accept(tfc);

                    }

                    else
                    {
                        Geometry::AttributeBinding binding = geom->getNormalBinding();
                        if (binding == Geometry::BIND_OFF)
                        {
                            printf("bind off\n");
                            continue;
                        }

                        if (binding == Geometry::BIND_OVERALL)
                        {
                            printf("bind overall \n");
                            continue;
                        }
                        else // BIND_PER_PRIMTIVE_SET, BIND_PER_PRIMTITIVE, BIND_PER_VERTEX

                        {
                            Geometry::PrimitiveSetList &primitiveSets = geom->getPrimitiveSetList();
                            Geometry::PrimitiveSetList::iterator itr;

                            Vec3Array::iterator coord_index = coords->begin();

                            for (itr = primitiveSets.begin(); itr != primitiveSets.end(); ++itr)
                            {
                                if (binding == Geometry::BIND_PER_PRIMITIVE_SET)
                                {

                                    printf("Bind per primitive set\n");
                                    continue;
                                }
                                else
                                {
                                    switch ((*itr)->getMode())
                                    {
                                    case (PrimitiveSet::LINES):
                                        for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                        {
                                            coord_index += 1;
                                            printf("lines\n");
                                        }
                                        break;

                                    case (PrimitiveSet::TRIANGLES):
                                        for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                        {
                                            if(raytracer::instance()->createFace(coord_index,0) == -1)
                                                std::cout << "An error occured during creating the embree geometry" <<std::endl;
                                            createTestFaces(3,coord_index, 0);

                                            coord_index +=3;

                                            numOfTriangles++;
                                            //printf("triangles\n");
                                        };
                                        printf("%i\n", (*itr)->getNumPrimitives());
                                        printf("%i\n", (*itr)->getNumIndices());
                                        break;

                                    case (PrimitiveSet::TRIANGLE_STRIP):
                                        for (unsigned int j = 0; j < (*itr)->getNumIndices() - 2; j++)
                                        {
                                            if(raytracer::instance()->createFace(coord_index,0) == -1)
                                                std::cout << "An error occured during creating the embree geometry" <<std::endl;
                                            createTestFaces(3,coord_index, 1);
                                            coord_index++;
                                            numOfTriangleStips++;
                                            //printf("triangle strips\n");
                                        }
                                        coord_index += 2;
                                        //                                    printf("%i\n", (*itr)->getNumIndices());

                                        //                                    for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                        //                                    {
                                        //                                        if(raytracer::instance()->createFace(coord_index,0) == -1)
                                        //                                           std::cout << "An error occured during creating the embree geometry" <<std::endl;
                                        //                                        createTestFaces(3,coord_index, 1);
                                        //                                        coord_index+=3;
                                        //                                        numOfTriangleStips++;
                                        //                                        //printf("triangle strips\n");
                                        //                                    };
                                        break;

                                    case (PrimitiveSet::TRIANGLE_FAN):
                                        printf("triangle fan\n");
                                        break;

                                    case (PrimitiveSet::QUADS):
                                        for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                        {
                                            //                                    coord_index += 4;
                                            //                                    normals_index += 4;
                                            if(raytracer::instance()->createFace(coord_index,1) == -1)
                                                std::cout << "An error occured during creating the embree geometry" <<std::endl;
                                            coord_index += 4;

                                            numOfQuads++;
                                            printf("quads\n");
                                        }
                                        break;

                                    case (PrimitiveSet::POLYGON):
                                    {
                                        DrawArrayLengths *draw = dynamic_cast<DrawArrayLengths *>(itr->get());
                                        if (!draw)
                                            break;
                                        for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                        {
                                            int nv = (*draw)[j];
                                            coord_index += nv;
                                            printf("polygon\n");
                                        }
                                    }
                                        break;
                                    case (PrimitiveSet::QUAD_STRIP):
                                        printf("quad strip\n");
                                        break;
                                    case (PrimitiveSet::PATCHES):
                                        printf("patches\n");
                                        break;
                                    case (PrimitiveSet::POINTS):
                                        printf("points\n");
                                        break;
                                    case (PrimitiveSet::LINES_ADJACENCY):
                                        printf("Lines adjacency\n");
                                        break;
                                    case (PrimitiveSet::LINE_LOOP):
                                        printf("Line loop\n");
                                        break;
                                    case (PrimitiveSet::LINE_STRIP_ADJACENCY):
                                        printf("Line strip adjacency\n");
                                        break;


                                    default:
                                        printf("default\n");
                                        break;
                                    }
                                }
                            }
                            printf("nothing found \n");
                        }
                    }
                }
                //nothing found
            }
    }
    //createFaceSet(vertexCoords, 0);
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


    if(type == 0)localGeodeTriangle->addDrawable(geom);
    if(type == 1)localGeodeTriangleStrip->addDrawable(geom);
}

void nodeVisitorVertex::createFaceSet(Vec3Array *coords, int type)
{
    osg::Geometry *geom = new osg::Geometry;

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

void nodeVisitTriangle::operator()(const osg::Vec3& v1, const osg::Vec3& v2, const osg::Vec3& v3, bool)const
{
    nvv_->fillVertexArray(v1,v2,v3);

        //raytracer::instance()->createFace(v1,v2,v3,0);
        osg::Geometry *geom = new osg::Geometry;

        osg::Vec3Array *vertices = new osg::Vec3Array;

        vertices->push_back(v1);
        vertices->push_back(v2);
        vertices->push_back(v3);

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
