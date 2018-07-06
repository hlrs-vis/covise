#include "nodevisitorvertex.h"

nodeVisitorVertex::nodeVisitorVertex():osg::NodeVisitor(TRAVERSE_ALL_CHILDREN)
{

}

void nodeVisitorVertex::apply(osg::Node &node)
{
    if (auto geode = dynamic_cast<osg::Geode *>(&node))
    {

    //std::cout << geode.getName() << std::endl;
    for (unsigned int i = 0; i < geode->getNumDrawables(); i++)
        {
            //Geometry *geom = dynamic_cast<Geometry *>(geode->getDrawable(i));
        Geometry *geom = geode->getDrawable(i)->asGeometry();
            if (geom)
            {
                printf("Looking for Geometry in Geode: %s\n", geode->getName().c_str());
                int numOfTriangles = 0;
                int numOfQuads = 0;
                Vec3Array *coords = dynamic_cast<Vec3Array *>(geom->getVertexArray());
                if (coords == 0L)
                {
                    printf("No coords\n");
                    continue;
                }

//                Vec3Array *normals = dynamic_cast<Vec3Array *>(geom->getNormalArray());
//                if (normals == 0L)
//                {
//                    printf("No normals\n");
//                    continue;
//                }

                Geometry::AttributeBinding binding = geom->getNormalBinding();
                if (binding == Geometry::BIND_OFF)
                {
                    printf("bind off\n");
                    continue;
                }

                Geode *g = geode;
//                Matrix *mat = cover->getWorldCoords(g);
//                Matrix nmat = *mat;
//                nmat.setTrans(Vec3(0., 0., 0.));

                if (binding == Geometry::BIND_OVERALL)
                {
//                    Vec3 v(0, 0, 0);
//                    Vec3 n = normals->front();

//                    Vec3Array::iterator coord_index = coords->begin();
//                    while (coord_index != coords->end())
//                        v += *(coord_index++);
//                    v /= (float)(coords->size());
//                    v = v * *mat;

//                    n *= _normal_scale;
//                    n = n * nmat;
//                    _local_coords->push_back(v);
//                    _local_coords->push_back((v + n));
                    printf("bind overall \n");
                }
                else // BIND_PER_PRIMTIVE_SET, BIND_PER_PRIMTITIVE, BIND_PER_VERTEX
                {
                    Geometry::PrimitiveSetList &primitiveSets = geom->getPrimitiveSetList();
                    Geometry::PrimitiveSetList::iterator itr;

                    Vec3Array::iterator coord_index = coords->begin();
                    //Vec3Array::iterator normals_index = normals->begin();

                    for (itr = primitiveSets.begin(); itr != primitiveSets.end(); ++itr)
                    {
    #ifdef DEBUG
                        _printPrimitiveType((*itr).get());
    #endif
                        if (binding == Geometry::BIND_PER_PRIMITIVE_SET)
                        {
//                            Vec3 v(0, 0, 0);
//                            Vec3 n = *(normals_index++);
//                            int ni = (*itr)->getNumIndices();
//                            for (int i = 0; i < ni; i++)
//                                v += *(coord_index++);
//                            v /= (float)(ni);
//                            v = v * *mat;

//                            n *= _normal_scale;
//                            n = n * nmat;
//                            _local_coords->push_back(v);
//                            _local_coords->push_back((v + n));
                            printf("Bind per primitive set\n");
                        }
                        else
                        {
                            switch ((*itr)->getMode())
                            {
                            case (PrimitiveSet::LINES):
                                for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                {
//                                    _processPrimitive(1, coord_index,
//                                                      normals_index, binding, *mat, nmat);
//                                    coord_index += 1;
//                                    normals_index += 1;
                                    printf("lines\n");
                                }
                                break;

                            case (PrimitiveSet::TRIANGLES):
                                for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                {
//                                    _processPrimitive(3, coord_index,
//                                                      normals_index, binding, *mat, nmat);
//                                    coord_index += 3;
//                                    normals_index += 3;
                                    if(raytracer::instance()->createFace(coord_index,0) == -1)
                                        std::cout << "An error occured during creating the embree geometry" <<std::endl;

                                    for(int h = 0; h < 3; h++)
                                    {

                                        Vec3 t = *coord_index;
                                        std::cout << t.x() << " " << t.y() << " " << t.z() << std::endl;
                                        coord_index++;
                                    }


                                    numOfTriangles++;
                                    printf("triangles\n");
                                }
                                break;

                            case (PrimitiveSet::TRIANGLE_STRIP):
                                for (unsigned int j = 0; j < (*itr)->getNumIndices() - 2; j++)
                                {
//                                    _processPrimitive(3, coord_index,
//                                                      normals_index, binding, *mat, nmat);
//                                    coord_index++;
//                                    normals_index++;
                                    printf("triangle strips\n");
                                }
//                                coord_index += 2;
//                                if (binding == Geometry::BIND_PER_VERTEX)
//                                    normals_index += 2;
                                break;

                            case (PrimitiveSet::TRIANGLE_FAN):
                                printf("triangle fan\n");
                                break;

                            case (PrimitiveSet::QUADS):
                                for (unsigned int j = 0; j < (*itr)->getNumPrimitives(); j++)
                                {
//                                    _processPrimitive(4, coord_index,
//                                                      normals_index, binding, *mat, nmat);
//                                    coord_index += 4;
//                                    normals_index += 4;
                                    if(raytracer::instance()->createFace(coord_index,1) == -1)
                                        std::cout << "An error occured during creating the embree geometry" <<std::endl;
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
//                                    int nv = (*draw)[j];
//                                    _processPrimitive(nv, coord_index,
//                                                      normals_index, binding, *mat, nmat);
//                                    coord_index += nv;
//                                    normals_index += nv;
                                    printf("polygon\n");
                                }
                            }
                            break;
                            case (PrimitiveSet::QUAD_STRIP):
                                printf("quad strip\n");
                                break;
                            default:
                                printf("default\n");
                                break;
                            }
                        }
                    }
                    printf("nothing found \n");
                }
                std::cout << numOfTriangles << std::endl;
            }
            else printf("Geometry not found in Geode: %s\n", geode->getName().c_str());
        }
    }

        traverse(node);
}

//void nodeVisitorVertex::_processPrimitive(unsigned int nv,
//                                                    Vec3Array::iterator coords,
//                                                    Vec3Array::iterator normals,
//                                                    Geometry::AttributeBinding binding,
//                                                    Matrix &mat, Matrix &nmat)
//{

//    Vec3 v(0, 0, 0);
//    Vec3 n(0, 0, 0);
//    if (_mode == SurfaceNormals)
//    {
//        if (binding == Geometry::BIND_PER_VERTEX)
//        {
//            for (unsigned int i = 0; i < nv; i++)
//                n += *(normals++);
//            n /= (float)(nv);
//        }

//        for (unsigned int i = 0; i < nv; i++)
//            v += *(coords++);
//        v /= (float)(nv);
//        v = v * mat;

//        n *= _normal_scale;
//        n = n * nmat;

//        _local_coords->push_back(v);
//        _local_coords->push_back((v + n));
//    }
//    else if (_mode == VertexNormals)
//    {
//        for (unsigned int i = 0; i < nv; i++)
//        {
//            v = *(coords++);
//            v = v * mat;

//            n = *(normals++);
//            n *= _normal_scale;
//            n = n * nmat;

//            _local_coords->push_back(v);
//            _local_coords->push_back((v + n));
//        }
//    }
//}
