/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef SORTLASTMASTER_H
#define SORTLASTMASTER_H

#include "SortLastImplementation.h"

#include <GL/gl.h>

#include <cassert>
#include <ctype.h>
#include <vector>
#include <mpi.h>

#define SL_DEPTH_TEXTURE_MODE_F32
//#define SL_DEPTH_TEXTURE_MODE_I32
//#define SL_DEPTH_TEXTURE_MODE_I24

template <typename T>
struct BufferTypeTraits
{
};

template <>
struct BufferTypeTraits<float>
{
    MPI_Datatype mpiType;
    static const float bufferInit = 1.0f;

    BufferTypeTraits()
    {
        mpiType = MPI_FLOAT;
    }
};

template <>
struct BufferTypeTraits<unsigned char>
{
    MPI_Datatype mpiType;
    static const unsigned char bufferInit = 255;

    BufferTypeTraits()
    {
        mpiType = MPI_BYTE;
    }
};

class SortLastMaster : public SortLastImplementation
{
public:
    SortLastMaster(const std::string &nodename, int session);
    virtual ~SortLastMaster();

    virtual bool init();
    virtual void preSwapBuffers(int windowNumber);

    virtual bool initialiseAsMaster();
    virtual bool initialiseAsSlave()
    {
        assert(0);
    }
    virtual bool createContext(const std::list<std::string> &hostlist, int groupIdentifier);

private:
    void initTextures();
    GLuint makeShader(GLuint program, GLuint type, const char *source, GLuint oldShader = 0);

    void gatherFrames();
    void deleteBuffers();

    void compositeSimpleReadback();
    void compositeSimpleShader();

    template <typename T>
    struct Buffer : BufferTypeTraits<T>
    {
        Buffer()
        {
            size = 0;
            data = 0;
        }
        Buffer(int s)
        {
            size = s;
            data = new T[s];
            componentSize = 1;
        }
        Buffer(int s, int cs)
        {
            size = s;
            data = new T[s];
            componentSize = cs;
        }

        int size; /// Size in elements
        T *data;
        int componentSize; /// Component size in elements, eg. RGB24 in unsigned char = 3

        typedef T DataType;
    };

    typedef Buffer<unsigned char> FrameBuffer;
    typedef Buffer<float> DepthBuffer;

    FrameBuffer **frameBuffers;
    DepthBuffer **depthBuffers;

    GLuint *textures;
    GLsizei numTextures;
    GLuint program;

    std::string fragmentSource;
    GLuint fragmentShader;

    std::vector<int> hostlist;

    int frameCtr;
    int session;

    bool initPending;
};

#endif // SORTLASTMASTER_H
