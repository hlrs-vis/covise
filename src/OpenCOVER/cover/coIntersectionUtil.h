/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COINTERSECTIONUTIL_H
#define COINTERSECTIONUTIL_H

#include <osg/TriangleFunctor>
#include <osg/Geometry>
#include <osgUtil/IntersectVisitor>

namespace opencover
{
namespace Private
{

    template <class T>
    class ParallelTriangleFunctor : public osg::TriangleFunctor<T>
    {

        virtual void drawArrays(GLenum mode, GLint first, GLsizei count)
        {
            if (this->_vertexArrayPtr == 0 || count == 0)
                return;

            const osg::Vec3 *vfirst = &this->_vertexArrayPtr[first];
            switch (mode)
            {
            case (GL_TRIANGLES):
            {
#ifdef _OPENMP
#pragma omp parallel for if (count > 20)
#endif
                for (int i = 2; i < count; i += 3)
                {
                    const osg::Vec3 *vptr = vfirst + i - 2;
                    //std::cerr << "TR " << omp_get_thread_num() << std::endl;
                    this->operator()(*(vptr), *(vptr + 1), *(vptr + 2), this->_treatVertexDataAsTemporary);
                }
                break;
            }
            case (GL_TRIANGLE_STRIP):
            {
#ifdef _OPENMP
#pragma omp parallel for if (count > 20)
#endif
                for (GLsizei i = 2; i < count; ++i)
                {
                    //std::cerr << "TS " << omp_get_thread_num() << std::endl;
                    const osg::Vec3 *vptr = vfirst + i - 2;
                    if ((i % 2))
                        this->operator()(*(vptr), *(vptr + 2), *(vptr + 1), this->_treatVertexDataAsTemporary);
                    else
                        this->operator()(*(vptr), *(vptr + 1), *(vptr + 2), this->_treatVertexDataAsTemporary);
                }
                break;
            }
            case (GL_QUADS):
            {
#ifdef _OPENMP
#pragma omp parallel for if (count > 20)
#endif
                for (GLsizei i = 3; i < count; i += 4)
                {
                    //std::cerr << "QD " << omp_get_thread_num() << std::endl;
                    const osg::Vec3 *vptr = vfirst + i - 3;
                    this->operator()(*(vptr), *(vptr + 1), *(vptr + 2), this->_treatVertexDataAsTemporary);
                    this->operator()(*(vptr), *(vptr + 2), *(vptr + 3), this->_treatVertexDataAsTemporary);
                }
                break;
            }
            case (GL_QUAD_STRIP):
            {
                // Vertices 2n-1, 2n, 2n+2, and 2n+1 define quadrilateral n (counting from 1)
#ifdef _OPENMP
#pragma omp parallel for if (count > 20)
#endif
                for (GLsizei i = 3; i < count; i += 2)
                {
                    //std::cerr << "QS " << omp_get_thread_num() << std::endl;
                    const osg::Vec3 *vptr = vfirst + i - 3;
                    this->operator()(*(vptr), *(vptr + 1), *(vptr + 3), this->_treatVertexDataAsTemporary);
                    this->operator()(*(vptr), *(vptr + 3), *(vptr + 2), this->_treatVertexDataAsTemporary);
                }
                break;
            }
            case (GL_POLYGON): // treat polygons as GL_TRIANGLE_FAN
            case (GL_TRIANGLE_FAN):
            {
#ifdef _OPENMP
#pragma omp parallel for if (count > 20)
#endif
                for (GLsizei i = 2; i < count; ++i)
                {
                    //std::cerr << "TF " << omp_get_thread_num() << std::endl;
                    const osg::Vec3 *vptr = vfirst + i - 1;
                    this->operator()(*(vfirst), *(vptr), *(vptr + 1), this->_treatVertexDataAsTemporary);
                }
                break;
            }
            case (GL_POINTS):
            case (GL_LINES):
            case (GL_LINE_STRIP):
            case (GL_LINE_LOOP):
            default:
                // can't be converted into to triangles.
                break;
            }
        }

        virtual void drawElements(GLenum mode, GLsizei count, const GLubyte *indices)
        {
            osg::TriangleFunctor<T>::drawElements(mode, count, indices);
            //std::cerr << "D";
        }

        virtual void drawElements(GLenum mode, GLsizei count, const GLushort *indices)
        {
            osg::TriangleFunctor<T>::drawElements(mode, count, indices);
            //std::cerr << "D";
        }

        virtual void drawElements(GLenum mode, GLsizei count, const GLuint *indices)
        {
            osg::TriangleFunctor<T>::drawElements(mode, count, indices);
            //std::cerr << "D";
        }
    };

    class coIntersectionSubVisitor;

    class coIntersectionVisitor : public osgUtil::IntersectVisitor
    {

    public:
        virtual void apply(osg::Geode &geode);
        virtual bool intersect(osg::Drawable &drawable);
        virtual void intersect(osg::Geometry *drawable, osg::PrimitiveFunctor &functor);

    private:
        //   std::vector<coIntersectionSubVisitor> subVisitors;
    };

    //class coIntersectionSubVisitor : public osgUtil::IntersectVisitor
    //{
    //public:
    //   coIntersectionSubVisitor() {}
    //   coIntersectionSubVisitor(const coIntersectionVisitor & visitor);
    //   virtual void intersect(osg::Drawable * drawable);
    //};
}
}

#endif // COINTERSECTIONUTIL_H
