#ifndef COVISE_OCT_SURFACE_PRIMITIVE_SET_H
#define COVISE_OCT_SURFACE_PRIMITIVE_SET_H

#include <osg/PrimitiveSet>
#include <osg/State>
#include <cassert>
#include <iostream>

class SurfacePrimitiveSet : public osg::DrawElementsUInt
{
public:
    using DrawElementsUInt::DrawElementsUInt;
    void setRange(size_t begin, size_t num)
    {
        auto factor = getNumIndices() / getNumPrimitives() ;
        m_begin = begin * factor;
        m_num = num * factor;
        assert(getNumPrimitives() >= begin + num);
    }
    
    void draw(osg::State& state, bool useVertexBufferObjects) const
    {
        GLenum mode = _mode;
#if defined(OSG_GLES1_AVAILABLE) || defined(OSG_GLES2_AVAILABLE) || defined(OSG_GLES3_AVAILABLE)
        if (mode == GL_POLYGON) mode = GL_TRIANGLE_FAN;
        if (mode == GL_QUAD_STRIP) mode = GL_TRIANGLE_STRIP;
#endif

        if (useVertexBufferObjects)
        {
            osg::GLBufferObject* ebo = getOrCreateGLBufferObject(state.getContextID());

            if (ebo)
            {
                state.getCurrentVertexArrayState()->bindElementBufferObject(ebo);
                glDrawElements(mode, m_num, GL_UNSIGNED_INT, (const GLvoid*)(ebo->getOffset(getBufferIndex()) + m_begin * sizeof(unsigned int)));
            }
        }
    }

private:
    size_t m_begin = 0;
    size_t m_num = 0;
};

#endif //COVISE_OCT_SURFACE_PRIMITIVE_SET_H
