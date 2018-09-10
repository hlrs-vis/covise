// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <GL/glew.h>

#include "vvibrclient.h"

#include "gl/handle.h"
#include "gl/util.h"
#include "math/serialization.h"
#include "private/vvgltools.h"
#include "private/vvibrimage.h"
#include "private/vvmessages.h"
#include "private/vvtimer.h"
#include "vvibr.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvvoldesc.h"

#include <boost/thread/mutex.hpp>

#include <cstdio>
#include <cstdlib>

namespace gl = virvo::gl;
using virvo::aabb;
using virvo::mat4;
using virvo::recti;
using virvo::vec3;
using virvo::vec4;

using virvo::makeMessage;
using virvo::Message;

#define TIME_FPS 0
#define TIME_READS 0
#define TIME_WRITES 0
#define TIME_NEW_IMAGES 1

struct GLClientState
{
    GLboolean colorArray;
    GLboolean edgeFlagArray;
    GLboolean indexArray;
    GLboolean normalArray;
    GLboolean textureCoordArray;
    GLboolean vertexArray;

    GLClientState()
        : colorArray(glIsEnabled(GL_COLOR_ARRAY))
        , edgeFlagArray(glIsEnabled(GL_EDGE_FLAG_ARRAY))
        , indexArray(glIsEnabled(GL_INDEX_ARRAY))
        , normalArray(glIsEnabled(GL_NORMAL_ARRAY))
        , textureCoordArray(glIsEnabled(GL_TEXTURE_COORD_ARRAY))
        , vertexArray(glIsEnabled(GL_VERTEX_ARRAY))
    {
    }

    ~GLClientState()
    {
        reset(GL_COLOR_ARRAY, colorArray);
        reset(GL_EDGE_FLAG_ARRAY, edgeFlagArray);
        reset(GL_INDEX_ARRAY, indexArray);
        reset(GL_NORMAL_ARRAY, normalArray);
        reset(GL_TEXTURE_COORD_ARRAY, textureCoordArray);
        reset(GL_VERTEX_ARRAY, vertexArray);
    }

    void reset(GLenum name, GLboolean value)
    {
        if (value)
            glEnableClientState(name);
        else
            glDisableClientState(name);
    }
};

struct vvIbrClient::Impl
{
    // The mutex to protect the members below
    boost::mutex lock;
    // The current image to render
    std::auto_ptr<virvo::IbrImage> curr;
    // The next image
    std::auto_ptr<virvo::IbrImage> next;
    // OpenGL objects
    gl::Buffer pointVBO;
    // OpenGL objects
    gl::Texture texRGBA;
    // OpenGL objects
    gl::Texture texDepth;
    // The IBR shader
    std::auto_ptr<vvShaderProgram> shader;
    // The current viewport
    recti viewport;
    // Current image matrix
    mat4 imgMatrix;

    Impl()
        : pointVBO(gl::createBuffer())
        , texRGBA(gl::createTexture())
        , texDepth(gl::createTexture())
        , shader(vvShaderFactory().createProgram("ibr", "", "ibr"))
    {
    }

    void setNextImage(virvo::IbrImage* image)
    {
        boost::unique_lock<boost::mutex> guard(lock);

        next.reset(image);
    }

    bool fetchNextImage()
    {
        boost::unique_lock<boost::mutex> guard(lock);

        if (next.get() == 0)
            return false;

        curr.reset(next.release());
        return true;
    }
};

vvIbrClient::vvIbrClient(vvVolDesc *vd, vvRenderState renderState,
        boost::shared_ptr<virvo::Connection> conn, std::string const& filename)
    : vvRemoteClient(REMOTE_IBR, vd, renderState, conn, filename)
    , impl_(new Impl)
{
    conn->set_handler(boost::bind(&vvIbrClient::handler, this, _1, _2, _3));

    init();
    init_connection(conn);
}

vvIbrClient::~vvIbrClient()
{
}

bool vvIbrClient::render()
{
#if TIME_FPS
    static virvo::FrameCounter counter;

    printf("IBR-client: FPS: %.2f\n", counter.registerFrame());
#endif

    // Send a new request
    if (conn_)
        conn_->write(makeMessage(Message::CameraMatrix, virvo::messages::CameraMatrix(view(), proj())));

    if (impl_->fetchNextImage())
    {
        // Decompress
        if (!impl_->curr->decompress())
        {
            throw std::runtime_error("decompression failed");
        }

        // Re-Initialize OpenGL textures and buffers
        initIbrFrame();
    }

    if (impl_->curr.get() == 0)
        return true;

#if 0

    virvo::gl::blendTexture(impl_->texRGBA.get(), GL_ONE, GL_ZERO);

#else

    // Render the current image
    virvo::IbrImage const& image = *impl_->curr.get();

    // Draw boundary lines
    if (_boundaries)
    {
        vec3 size = vd->getSize(); // volume size [world coordinates]
        drawBoundingBox(size, vd->pos, _boundColor);
    }

    // Get the current projection and model-view matrices
    mat4 currentPr = gl::getProjectionMatrix();
    mat4 currentMv = gl::getModelviewMatrix();
    mat4 currentMatrix = currentPr * currentMv;

    float drMin = 0.0f;
    float drMax = 0.0f;

    aabb box = vd->getBoundingBox();

    virvo::ibr::calcDepthRange(currentPr, currentMv, box, drMin, drMax);

    recti vp = gl::getViewport();

    mat4 currentImgMatrix = virvo::ibr::calcImgMatrix(currentPr, currentMv, vp, drMin, drMax);

    bool matrixChanged = currentImgMatrix != impl_->imgMatrix;

    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    mat4 reprojectionMatrix;
    if (!matrixChanged)
    {
        reprojectionMatrix = mat4::identity();
    }
    else
    {
        mat4 invOld = inverse(image.projMatrix() * image.viewMatrix());
        reprojectionMatrix = currentMatrix * invOld;
    }

    mat4 invMv = inverse(currentMv);

    vec4 viewerObj(0.f, 0.f, 0.f, 1.f);
    viewerObj = invMv * viewerObj;
    viewerObj = image.viewMatrix() * viewerObj;

    bool closer = viewerObj[2] > 0.f; // inverse render order if viewer has moved closer

    // project current viewer onto original image along its normal
    viewerObj = image.projMatrix() * viewerObj;

    float splitX = (viewerObj[0] / viewerObj[3] + 1.f) * image.viewport()[2] * 0.5f;
    float splitY = (viewerObj[1] / viewerObj[3] + 1.f) * image.viewport()[3] * 0.5f;

    splitX = ts_clamp(splitX, 0.f, float(image.viewport()[2] - 1));
    splitY = ts_clamp(splitY, 0.f, float(image.viewport()[3] - 1));

    GLboolean depthMask = GL_TRUE;

    glGetBooleanv(GL_DEPTH_WRITEMASK, &depthMask);

    glDepthMask(GL_FALSE);

    GLboolean pointSmooth = GL_FALSE;

    glGetBooleanv(GL_POINT_SMOOTH, &pointSmooth);

    //glEnable(GL_POINT_SMOOTH);

    glEnable(GL_POINT_SPRITE);
    glTexEnvf(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);

    impl_->shader->enable();

    impl_->shader->setParameter1f("vpWidth", static_cast<float>(vp[2]));
    impl_->shader->setParameter1f("vpHeight", static_cast<float>(vp[3]));
    impl_->shader->setParameter1f("imageWidth", static_cast<float>(image.viewport()[2]));
    impl_->shader->setParameter1f("imageHeight", static_cast<float>(image.viewport()[3]));
    impl_->shader->setParameterTex2D("rgbaTex", impl_->texRGBA.get());
    impl_->shader->setParameterTex2D("depthTex", impl_->texDepth.get());
    impl_->shader->setParameter1f("splitX", splitX);
    impl_->shader->setParameter1f("splitY", splitY);
    impl_->shader->setParameter1f("depthMin", image.depthMin());
    impl_->shader->setParameter1f("depthRange", image.depthMax() - image.depthMin());
    impl_->shader->setParameter1i("closer", closer);
    impl_->shader->setParameterMatrix4f("reprojectionMatrix" , reprojectionMatrix);

    //// begin ellipsoid test code - temporary

    // hardwired parameters for now
    /*  float v_i[16] = {
    0.70710678f, 0.70710678f, 0.0f, 0.0f,
    -0.70710678f, 0.70710678f, 0.0f, 0.0f,
    0.0f,        0.0f, 1.0f, 0.0f,
    0.0f,        0.0f, 0.0f, 1.0f
    };*/
    float v_i[16] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 1.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    };
    mat4 V_i = inverse(mat4(v_i));

    impl_->shader->setParameter1f("si", 1.0);
    impl_->shader->setParameter1f("sj", 1.0);
    impl_->shader->setParameter1f("sk", 1.0);
    impl_->shader->setParameterMatrix4f("V_i" , V_i);

    //// end ellipsoid test code - temporary

    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // push the client state and temporarily disable every
    // client state but vertex arrays
    {
        GLClientState state_saver;

        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_COLOR_ARRAY);
        glDisableClientState(GL_EDGE_FLAG_ARRAY);
        glDisableClientState(GL_INDEX_ARRAY);
        glDisableClientState(GL_NORMAL_ARRAY);
        glDisableClientState(GL_TEXTURE_COORD_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, impl_->pointVBO.get());
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 0, NULL);

        glDrawArrays(GL_POINTS, 0, image.viewport()[2] * image.viewport()[3]);

        glDisableClientState(GL_VERTEX_ARRAY);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    }

    impl_->shader->disable();

    if(depthMask)
        glDepthMask(GL_TRUE);
    if(!pointSmooth)
        glDisable(GL_POINT_SMOOTH);

#endif

    return true;
}

void vvIbrClient::handler(virvo::Connection::Reason reason, virvo::MessagePointer message, boost::system::error_code const& e)
{
    if (e)
    {
        // Connection lost!?
        conn_->close();
        conn_.reset();
        return;
    }

    switch (reason)
    {
    case virvo::Connection::Read:
        on_read(message);
        break;
    case virvo::Connection::Write:
        on_write(message);
        break;
    }
}

void vvIbrClient::on_read(virvo::MessagePointer message)
{
#if TIME_READS
    static virvo::FrameCounter counter;

    printf("IBR-client: reads/sec: %.2f\n", counter.registerFrame());
#endif

    switch (message->type())
    {
    case virvo::Message::IbrImage:
        processIbrImage(message);
        break;
    }
}

void vvIbrClient::on_write(virvo::MessagePointer message)
{
#if TIME_WRITES
    static virvo::FrameCounter counter;

    printf("IBR-client: writes/sec: %.2f\n", counter.registerFrame());
#endif
}

void vvIbrClient::processIbrImage(virvo::MessagePointer message)
{
#if TIME_NEW_IMAGES
    static virvo::FrameCounter counter;

    printf("IBR-client: images/sec %.2f\n", counter.registerFrame());
#endif

    // Create a new image
    std::auto_ptr<virvo::IbrImage> image(new virvo::IbrImage);

    // Extract the image from the message
    //
    // FIXME:
    // Move into render() ??
    //
    if (!message->deserialize(*image))
    {
        throw std::runtime_error("deserialization failed");
    }

    // Update the next image
    impl_->setNextImage(image.release());
}

void vvIbrClient::initIbrFrame()
{
    virvo::IbrImage& image = *impl_->curr;

    impl_->imgMatrix = virvo::ibr::calcImgMatrix(
        image.projMatrix(), image.viewMatrix(), image.viewport(), image.depthMin(), image.depthMax());

    int h = image.height();
    int w = image.width();

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    glBindTexture(GL_TEXTURE_2D, impl_->texRGBA.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

    virvo::PixelFormatInfo cf = mapPixelFormat(image.colorBuffer().format());

    glTexImage2D(GL_TEXTURE_2D, 0, cf.internalFormat, w, h, 0, cf.format, cf.type,
        image.colorBuffer().data().ptr());

    glBindTexture(GL_TEXTURE_2D, impl_->texDepth.get());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

    virvo::PixelFormatInfo df = mapPixelFormat(image.depthBuffer().format());

    glTexImage2D(GL_TEXTURE_2D, 0, df.internalFormat, w, h, 0, df.format, df.type,
        image.depthBuffer().data().ptr());

    if (impl_->viewport[2] != w || impl_->viewport[3] != h)
    {
        impl_->viewport = image.viewport();

        std::vector<GLfloat> points(w * h * 3);

        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                points[y*w*3 + x*3 + 0] = static_cast<float>(x);
                points[y*w*3 + x*3 + 1] = static_cast<float>(y);
                points[y*w*3 + x*3 + 2] = 0.f;
            }
        }

        // VBO for points
        glBindBuffer(GL_ARRAY_BUFFER, impl_->pointVBO.get());
        glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), &points[0], GL_STATIC_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}
