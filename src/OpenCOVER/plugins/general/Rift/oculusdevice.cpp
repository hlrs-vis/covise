/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*
 * oculusdevice.cpp
 *
 *  Created on: Jul 03, 2013
 *      Author: Bjorn Blissing
 */

#include "oculusdevice.h"
#include <cover/input/input.h>
#include <cover/input/dev/rift/RIFTDriver.h>

#include <osg/Geometry>

const std::string OculusDevice::m_warpVertexShaderSource(
    "#version 110\n"

    "uniform vec2 EyeToSourceUVScale;\n"
    "uniform vec2 EyeToSourceUVOffset;\n"

    "attribute vec2 Position;\n"
    "attribute vec4 Color;\n"
    "attribute vec2 TexCoord0;\n"
    "attribute vec2 TexCoord1;\n"
    "attribute vec2 TexCoord2;\n"

    "varying vec4 oColor;\n"
    "varying vec2 oTexCoord0;\n"
    "varying vec2 oTexCoord1;\n"
    "varying vec2 oTexCoord2;\n"

    "void main()\n"
    "{\n"
    "   gl_Position.x = Position.x;\n"
    "   gl_Position.y = Position.y;\n"
    "   gl_Position.z = 0.5;\n"
    "   gl_Position.w = 1.0;\n"
    "   // Vertex inputs are in TanEyeAngle space for the R,G,B channels (i.e. after chromatic aberration and distortion).\n"
    "   // Scale them into the correct [0-1],[0-1] UV lookup space (depending on eye)\n"
    "   oTexCoord0 = TexCoord0 * EyeToSourceUVScale + EyeToSourceUVOffset;\n"
    "   oTexCoord0.y = 1.0-oTexCoord0.y;\n"
    "   oTexCoord1 = TexCoord1 * EyeToSourceUVScale + EyeToSourceUVOffset;\n"
    "   oTexCoord1.y = 1.0-oTexCoord1.y;\n"
    "   oTexCoord2 = TexCoord2 * EyeToSourceUVScale + EyeToSourceUVOffset;\n"
    "   oTexCoord2.y = 1.0-oTexCoord2.y;\n"
    "   oColor = Color; // Used for vignette fade.\n"
    "}\n");

const std::string OculusDevice::m_warpWithTimewarpVertexShaderSource(
    "#version 110\n"

    "uniform vec2 EyeToSourceUVScale;\n"
    "uniform vec2 EyeToSourceUVOffset;\n"
    "uniform mat4 EyeRotationStart;\n"
    "uniform mat4 EyeRotationEnd;\n"

    "attribute vec2 Position;\n"
    "attribute vec4 Color;\n"
    "attribute vec2 TexCoord0;\n"
    "attribute vec2 TexCoord1;\n"
    "attribute vec2 TexCoord2;\n"

    "varying vec4 oColor;\n"
    "varying vec2 oTexCoord0;\n"
    "varying vec2 oTexCoord1;\n"
    "varying vec2 oTexCoord2;\n"

    "void main()\n"
    "{\n"
    "   gl_Position.x = Position.x;\n"
    "   gl_Position.y = Position.y;\n"
    "   gl_Position.z = 0.0;\n"
    "   gl_Position.w = 1.0;\n"

    "    // Vertex inputs are in TanEyeAngle space for the R,G,B channels (i.e. after chromatic aberration and distortion).\n"
    "    // These are now real world vectors in direction(x, y, 1) relative to the eye of the HMD.\n"
    "   vec3 TanEyeAngleR = vec3 ( TexCoord0.x, TexCoord0.y, 1.0 );\n"
    "   vec3 TanEyeAngleG = vec3 ( TexCoord1.x, TexCoord1.y, 1.0 );\n"
    "   vec3 TanEyeAngleB = vec3 ( TexCoord2.x, TexCoord2.y, 1.0 );\n"

    "   mat3 EyeRotation;\n"
    "   EyeRotation[0] = mix ( EyeRotationStart[0], EyeRotationEnd[0], Color.a ).xyz;\n"
    "   EyeRotation[1] = mix ( EyeRotationStart[1], EyeRotationEnd[1], Color.a ).xyz;\n"
    "   EyeRotation[2] = mix ( EyeRotationStart[2], EyeRotationEnd[2], Color.a ).xyz;\n"
    "   vec3 TransformedR   = EyeRotation * TanEyeAngleR;\n"
    "   vec3 TransformedG   = EyeRotation * TanEyeAngleG;\n"
    "   vec3 TransformedB   = EyeRotation * TanEyeAngleB;\n"

    "    // Project them back onto the Z=1 plane of the rendered images.\n"
    "   float RecipZR = 1.0 / TransformedR.z;\n"
    "   float RecipZG = 1.0 / TransformedG.z;\n"
    "   float RecipZB = 1.0 / TransformedB.z;\n"
    "   vec2 FlattenedR = vec2 ( TransformedR.x * RecipZR, TransformedR.y * RecipZR );\n"
    "   vec2 FlattenedG = vec2 ( TransformedG.x * RecipZG, TransformedG.y * RecipZG );\n"
    "   vec2 FlattenedB = vec2 ( TransformedB.x * RecipZB, TransformedB.y * RecipZB );\n"

    "    // These are now still in TanEyeAngle space.\n"
    "    // Scale them into the correct [0-1],[0-1] UV lookup space (depending on eye)\n"
    "   vec2 SrcCoordR = FlattenedR * EyeToSourceUVScale + EyeToSourceUVOffset;\n"
    "   vec2 SrcCoordG = FlattenedG * EyeToSourceUVScale + EyeToSourceUVOffset;\n"
    "   vec2 SrcCoordB = FlattenedB * EyeToSourceUVScale + EyeToSourceUVOffset;\n"
    "   oTexCoord0 = SrcCoordR;\n"
    "   oTexCoord0.y = 1.0-oTexCoord0.y;\n"
    "   oTexCoord1 = SrcCoordG;\n"
    "   oTexCoord1.y = 1.0-oTexCoord1.y;\n"
    "   oTexCoord2 = SrcCoordB;\n"
    "   oTexCoord2.y = 1.0-oTexCoord2.y;\n"
    "   oColor = vec4(Color.r, Color.r, Color.r, Color.r); // Used for vignette fade.\n"
    "}\n");

const std::string OculusDevice::m_warpFragmentShaderSource(
    "#version 110\n"
    "    \n"
    "uniform sampler2D Texture;\n"
    "    \n"
    "varying vec4 oColor;\n"
    "varying vec2 oTexCoord0;\n"
    "varying vec2 oTexCoord1;\n"
    "varying vec2 oTexCoord2;\n"
    "    \n"
    "void main()\n"
    "{\n"
    "   gl_FragColor.r = oColor.r * texture2D(Texture, oTexCoord0).r;\n"
    "   gl_FragColor.g = oColor.g * texture2D(Texture, oTexCoord1).g;\n"
    "   gl_FragColor.b = oColor.b * texture2D(Texture, oTexCoord2).b;\n"
    "   gl_FragColor.a = 1.0;\n"
    "}\n");

OculusDevice::OculusDevice(float nearClip, float farClip, float pixelsPerDisplayPixel, bool useTimewarp)
    : m_hmdDevice(0)
    , m_nearClip(nearClip)
    , m_farClip(farClip)
    , m_useTimeWarp(useTimewarp)
    , m_position(osg::Vec3(0.0f, 0.0f, 0.0f))
    , m_orientation(osg::Quat(0.0f, 0.0f, 0.0f, 1.0f))
{
    
    opencover::InputDevice *riftInput = opencover::Input::instance()->getDevice("RiftDevice");
    if(riftInput != NULL)
    {
        trackingDriver = (RIFTDriver *)riftInput;
    }

    ovr_Initialize();

    // Enumerate HMD devices
    int numberOfDevices = ovrHmd_Detect();
    osg::notify(osg::DEBUG_INFO) << "Number of connected devices: " << numberOfDevices << std::endl;

    // Get first available HMD
    m_hmdDevice = ovrHmd_Create(0);

    // If no HMD is found try an emulated device
    if (!m_hmdDevice)
    {
        osg::notify(osg::WARN) << "Warning: No device could be found. Creating emulated device " << std::endl;
        m_hmdDevice = ovrHmd_CreateDebug(ovrHmd_DK1);
        ovrHmd_ResetFrameTiming(m_hmdDevice, 0);
    }

    if (m_hmdDevice)
    {
        // Print out some information about the HMD
        osg::notify(osg::ALWAYS) << "Product:         " << m_hmdDevice->ProductName << std::endl;
        osg::notify(osg::ALWAYS) << "Manufacturer:    " << m_hmdDevice->Manufacturer << std::endl;
        osg::notify(osg::ALWAYS) << "VendorId:        " << m_hmdDevice->VendorId << std::endl;
        osg::notify(osg::ALWAYS) << "ProductId:       " << m_hmdDevice->ProductId << std::endl;
        osg::notify(osg::ALWAYS) << "SerialNumber:    " << m_hmdDevice->SerialNumber << std::endl;
        osg::notify(osg::ALWAYS) << "FirmwareVersion: " << m_hmdDevice->FirmwareMajor << "." << m_hmdDevice->FirmwareMinor << std::endl;

        // Get more details about the HMD.
        m_resolution = m_hmdDevice->Resolution;

        // Compute recommended render texture size
        if (pixelsPerDisplayPixel > 1.0f)
        {
            osg::notify(osg::WARN) << "Warning: Pixel per display pixel is set to a value higher than 1.0." << std::endl;
        }

        ovrSizei recommenedLeftTextureSize = ovrHmd_GetFovTextureSize(m_hmdDevice, ovrEye_Left, m_hmdDevice->DefaultEyeFov[0], pixelsPerDisplayPixel);
        ovrSizei recommenedRightTextureSize = ovrHmd_GetFovTextureSize(m_hmdDevice, ovrEye_Right, m_hmdDevice->DefaultEyeFov[1], pixelsPerDisplayPixel);

        // Compute size of render target
        m_renderTargetSize.w = recommenedLeftTextureSize.w + recommenedRightTextureSize.w;
        m_renderTargetSize.h = osg::maximum(recommenedLeftTextureSize.h, recommenedRightTextureSize.h);

        // Initialize ovrEyeRenderDesc struct.
        m_eyeRenderDesc[0] = ovrHmd_GetRenderDesc(m_hmdDevice, ovrEye_Left, m_hmdDevice->DefaultEyeFov[0]);
        m_eyeRenderDesc[1] = ovrHmd_GetRenderDesc(m_hmdDevice, ovrEye_Right, m_hmdDevice->DefaultEyeFov[1]);

        ovrVector3f leftEyeAdjust = m_eyeRenderDesc[0].HmdToEyeViewOffset;
        m_leftEyeAdjust.set(leftEyeAdjust.x, leftEyeAdjust.y, leftEyeAdjust.z);
        ovrVector3f rightEyeAdjust = m_eyeRenderDesc[1].HmdToEyeViewOffset;
        m_rightEyeAdjust.set(rightEyeAdjust.x, rightEyeAdjust.y, rightEyeAdjust.z);

        bool isRightHanded = true;

        ovrMatrix4f leftEyeProjectionMatrix = ovrMatrix4f_Projection(m_eyeRenderDesc[0].Fov, m_nearClip, m_farClip, isRightHanded);
        // Transpose matrix
        m_leftEyeProjectionMatrix.set(leftEyeProjectionMatrix.M[0][0], leftEyeProjectionMatrix.M[1][0], leftEyeProjectionMatrix.M[2][0], leftEyeProjectionMatrix.M[3][0],
                                      leftEyeProjectionMatrix.M[0][1], leftEyeProjectionMatrix.M[1][1], leftEyeProjectionMatrix.M[2][1], leftEyeProjectionMatrix.M[3][1],
                                      leftEyeProjectionMatrix.M[0][2], leftEyeProjectionMatrix.M[1][2], leftEyeProjectionMatrix.M[2][2], leftEyeProjectionMatrix.M[3][2],
                                      leftEyeProjectionMatrix.M[0][3], leftEyeProjectionMatrix.M[1][3], leftEyeProjectionMatrix.M[2][3], leftEyeProjectionMatrix.M[3][3]);

        ovrMatrix4f rightEyeProjectionMatrix = ovrMatrix4f_Projection(m_eyeRenderDesc[1].Fov, m_nearClip, m_farClip, isRightHanded);
        // Transpose matrix
        m_rightEyeProjectionMatrix.set(rightEyeProjectionMatrix.M[0][0], rightEyeProjectionMatrix.M[1][0], rightEyeProjectionMatrix.M[2][0], rightEyeProjectionMatrix.M[3][0],
                                       rightEyeProjectionMatrix.M[0][1], rightEyeProjectionMatrix.M[1][1], rightEyeProjectionMatrix.M[2][1], rightEyeProjectionMatrix.M[3][1],
                                       rightEyeProjectionMatrix.M[0][2], rightEyeProjectionMatrix.M[1][2], rightEyeProjectionMatrix.M[2][2], rightEyeProjectionMatrix.M[3][2],
                                       rightEyeProjectionMatrix.M[0][3], rightEyeProjectionMatrix.M[1][3], rightEyeProjectionMatrix.M[2][3], rightEyeProjectionMatrix.M[3][3]);

        // Start the sensor which provides the Rift<92>s pose and motion.
        ovrHmd_ConfigureTracking(m_hmdDevice, ovrTrackingCap_Orientation | ovrTrackingCap_MagYawCorrection | ovrTrackingCap_Position, 0);

        beginFrameTiming();
    }
}

OculusDevice::~OculusDevice()
{
    ovrHmd_Destroy(m_hmdDevice);
    ovr_Shutdown();
}

unsigned int OculusDevice::screenResolutionWidth() const
{
    return m_hmdDevice->Resolution.w;
}

unsigned int OculusDevice::screenResolutionHeight() const
{
    return m_hmdDevice->Resolution.h;
}

unsigned int OculusDevice::renderTargetWidth() const
{
    return m_renderTargetSize.w;
}

unsigned int OculusDevice::renderTargetHeight() const
{
    return m_renderTargetSize.h;
}

osg::Matrix OculusDevice::projectionMatrixCenter() const
{
    osg::Matrix projectionMatrixCenter;
    projectionMatrixCenter = m_leftEyeProjectionMatrix.operator*(0.5) + m_rightEyeProjectionMatrix.operator*(0.5);
    return projectionMatrixCenter;
}

osg::Matrix OculusDevice::projectionMatrixLeft() const
{
    return m_leftEyeProjectionMatrix;
}

osg::Matrix OculusDevice::projectionMatrixRight() const
{
    return m_rightEyeProjectionMatrix;
}

osg::Matrix OculusDevice::projectionOffsetMatrixLeft() const
{
    osg::Matrix projectionOffsetMatrix;
    float offset = m_leftEyeProjectionMatrix(2, 0);
    projectionOffsetMatrix.makeTranslate(osg::Vec3(-offset, 0.0, 0.0));
    return projectionOffsetMatrix;
}

osg::Matrix OculusDevice::projectionOffsetMatrixRight() const
{
    osg::Matrix projectionOffsetMatrix;
    float offset = m_rightEyeProjectionMatrix(2, 0);
    projectionOffsetMatrix.makeTranslate(osg::Vec3(-offset, 0.0, 0.0));
    return projectionOffsetMatrix;
}

osg::Matrix OculusDevice::viewMatrixLeft() const
{
    osg::Matrix viewMatrix;
    viewMatrix.makeTranslate(m_leftEyeAdjust);
    return viewMatrix;
}

osg::Matrix OculusDevice::viewMatrixRight() const
{
    osg::Matrix viewMatrix;
    viewMatrix.makeTranslate(m_rightEyeAdjust);
    return viewMatrix;
}

void OculusDevice::updatePose(unsigned int frameIndex)
{
    // Ask the API for the times when this frame is expected to be displayed.
    m_frameTiming = ovrHmd_GetFrameTiming(m_hmdDevice, frameIndex);

    // Query the HMD for the current tracking state.
    ovrTrackingState ts = ovrHmd_GetTrackingState(m_hmdDevice, m_frameTiming.ScanoutMidpointSeconds);
    ovrPoseStatef headpose = ts.HeadPose;
    ovrPosef pose = headpose.ThePose;
    m_position.set(-pose.Position.x, -pose.Position.y, -pose.Position.z);
    m_orientation.set(pose.Orientation.x, pose.Orientation.y, pose.Orientation.z, -pose.Orientation.w);

    // Get head pose for both eyes (used for time warp
    for (int eyeIndex = 0; eyeIndex < ovrEye_Count; ++eyeIndex)
    {
        ovrEyeType eye = m_hmdDevice->EyeRenderOrder[eyeIndex];
        m_headPose[eye] = ovrHmd_GetHmdPosePerEye(m_hmdDevice, eye);
    }
}

void OculusDevice::resetSensorOrientation() const
{
    ovrHmd_RecenterPose(m_hmdDevice);
}

int OculusDevice::renderOrder(Eye eye) const
{
    for (int eyeIndex = 0; eyeIndex < ovrEye_Count; ++eyeIndex)
    {
        ovrEyeType ovrEye = m_hmdDevice->EyeRenderOrder[eyeIndex];
        if (ovrEye == ovrEye_Left && eye == LEFT)
        {
            return eyeIndex;
        }
        if (ovrEye == ovrEye_Right && eye == RIGHT)
        {
            return eyeIndex;
        }
    }
    return 0;
}

osg::Geode *OculusDevice::distortionMesh(Eye eye, osg::Program *program, int x, int y, int w, int h, bool splitViewport)
{
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    // Allocate & generate distortion mesh vertices.
    ovrDistortionMesh meshData;
    ovrHmd_CreateDistortionMesh(m_hmdDevice, m_eyeRenderDesc[eye].Eye, m_eyeRenderDesc[eye].Fov, ovrDistortionCap_Chromatic | ovrDistortionCap_TimeWarp, &meshData);

    // Now parse the vertex data and create a render ready vertex buffer from it
    ovrDistortionVertex *ov = meshData.pVertexData;
    osg::Vec2Array *positionArray = new osg::Vec2Array;
    osg::Vec4Array *colorArray = new osg::Vec4Array;
    osg::Vec2Array *textureRArray = new osg::Vec2Array;
    osg::Vec2Array *textureGArray = new osg::Vec2Array;
    osg::Vec2Array *textureBArray = new osg::Vec2Array;

    for (unsigned vertNum = 0; vertNum < meshData.VertexCount; ++vertNum)
    {
        if (splitViewport)
        {
            // Positions need to be scaled and translated if we are using one viewport per eye
            if (eye == LEFT)
            {
                positionArray->push_back(osg::Vec2f(2 * ov[vertNum].ScreenPosNDC.x + 1.0, ov[vertNum].ScreenPosNDC.y));
            }
            else if (eye == RIGHT)
            {
                positionArray->push_back(osg::Vec2f(2 * ov[vertNum].ScreenPosNDC.x - 1.0, ov[vertNum].ScreenPosNDC.y));
            }
        }
        else
        {
            positionArray->push_back(osg::Vec2f(ov[vertNum].ScreenPosNDC.x, ov[vertNum].ScreenPosNDC.y));
        }

        colorArray->push_back(osg::Vec4f(ov[vertNum].VignetteFactor, ov[vertNum].VignetteFactor, ov[vertNum].VignetteFactor, ov[vertNum].TimeWarpFactor));
        textureRArray->push_back(osg::Vec2f(ov[vertNum].TanEyeAnglesR.x, ov[vertNum].TanEyeAnglesR.y));
        textureGArray->push_back(osg::Vec2f(ov[vertNum].TanEyeAnglesG.x, ov[vertNum].TanEyeAnglesG.y));
        textureBArray->push_back(osg::Vec2f(ov[vertNum].TanEyeAnglesB.x, ov[vertNum].TanEyeAnglesB.y));
    }

    // Get triangle indicies
    osg::UShortArray *indexArray = new osg::UShortArray;
    unsigned short *index = meshData.pIndexData;
    for (unsigned indexNum = 0; indexNum < meshData.IndexCount; ++indexNum)
    {
        indexArray->push_back(index[indexNum]);
    }

    // Deallocate the mesh data
    ovrHmd_DestroyDistortionMesh(&meshData);

    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
    geometry->setUseDisplayList(false);
    geometry->setUseVertexBufferObjects(true);
    osg::ref_ptr<osg::DrawElementsUShort> drawElement = new osg::DrawElementsUShort(osg::PrimitiveSet::TRIANGLES, indexArray->size(), (GLushort *)indexArray->getDataPointer());
    geometry->addPrimitiveSet(drawElement);

    GLuint positionLoc = 0;
    GLuint colorLoc = 1;
    GLuint texCoord0Loc = 2;
    GLuint texCoord1Loc = 3;
    GLuint texCoord2Loc = 4;

    program->addBindAttribLocation("Position", positionLoc);
    geometry->setVertexAttribArray(positionLoc, positionArray);
    geometry->setVertexAttribBinding(positionLoc, osg::Geometry::BIND_PER_VERTEX);

    program->addBindAttribLocation("Color", colorLoc);
    geometry->setVertexAttribArray(colorLoc, colorArray);
    geometry->setVertexAttribBinding(colorLoc, osg::Geometry::BIND_PER_VERTEX);

    program->addBindAttribLocation("TexCoord0", texCoord0Loc);
    geometry->setVertexAttribArray(texCoord0Loc, textureRArray);
    geometry->setVertexAttribBinding(texCoord0Loc, osg::Geometry::BIND_PER_VERTEX);

    program->addBindAttribLocation("TexCoord1", texCoord1Loc);
    geometry->setVertexAttribArray(texCoord1Loc, textureGArray);
    geometry->setVertexAttribBinding(texCoord1Loc, osg::Geometry::BIND_PER_VERTEX);

    program->addBindAttribLocation("TexCoord2", texCoord2Loc);
    geometry->setVertexAttribArray(texCoord2Loc, textureBArray);
    geometry->setVertexAttribBinding(texCoord2Loc, osg::Geometry::BIND_PER_VERTEX);

    // Compute UV scale and offset
    ovrRecti eyeRenderViewport;
    eyeRenderViewport.Pos.x = x;
    eyeRenderViewport.Pos.y = y;
    eyeRenderViewport.Size.w = w;
    eyeRenderViewport.Size.h = h;
    ovrSizei renderTargetSize;
    renderTargetSize.w = m_renderTargetSize.w / 2;
    renderTargetSize.h = m_renderTargetSize.h;
    ovrHmd_GetRenderScaleAndOffset(m_eyeRenderDesc[eye].Fov, renderTargetSize, eyeRenderViewport, m_UVScaleOffset[eye]);
    geode->addDrawable(geometry);
    return geode.release();
}

osg::Vec2f OculusDevice::eyeToSourceUVScale(Eye eye) const
{
    osg::Vec2f uvScale(m_UVScaleOffset[eye][0].x, m_UVScaleOffset[eye][0].y);
    return uvScale;
}
osg::Vec2f OculusDevice::eyeToSourceUVOffset(Eye eye) const
{
    osg::Vec2f uvOffset(m_UVScaleOffset[eye][1].x, m_UVScaleOffset[eye][1].y);
    return uvOffset;
}

osg::Matrixf OculusDevice::eyeRotationStart(Eye eye) const
{
    osg::Matrixf rotationStart;

    ovrMatrix4f rotationMatrix = m_timeWarpMatrices[eye][0];
    // Transpose matrix
    rotationStart.set(rotationMatrix.M[0][0], rotationMatrix.M[1][0], rotationMatrix.M[2][0], rotationMatrix.M[3][0],
                      rotationMatrix.M[0][1], rotationMatrix.M[1][1], rotationMatrix.M[2][1], rotationMatrix.M[3][1],
                      rotationMatrix.M[0][2], rotationMatrix.M[1][2], rotationMatrix.M[2][2], rotationMatrix.M[3][2],
                      rotationMatrix.M[0][3], rotationMatrix.M[1][3], rotationMatrix.M[2][3], rotationMatrix.M[3][3]);

    return rotationStart;
}

osg::Matrixf OculusDevice::eyeRotationEnd(Eye eye) const
{
    osg::Matrixf rotationEnd;

    ovrMatrix4f rotationMatrix = m_timeWarpMatrices[eye][1];
    // Transpose matrix
    rotationEnd.set(rotationMatrix.M[0][0], rotationMatrix.M[1][0], rotationMatrix.M[2][0], rotationMatrix.M[3][0],
                    rotationMatrix.M[0][1], rotationMatrix.M[1][1], rotationMatrix.M[2][1], rotationMatrix.M[3][1],
                    rotationMatrix.M[0][2], rotationMatrix.M[1][2], rotationMatrix.M[2][2], rotationMatrix.M[3][2],
                    rotationMatrix.M[0][3], rotationMatrix.M[1][3], rotationMatrix.M[2][3], rotationMatrix.M[3][3]);

    return rotationEnd;
}

void OculusDevice::beginFrameTiming(unsigned int frameIndex)
{
    m_frameTiming = ovrHmd_BeginFrameTiming(m_hmdDevice, frameIndex);
}

void OculusDevice::endFrameTiming() const
{
    ovrHmd_EndFrameTiming(m_hmdDevice);
}

void OculusDevice::waitTillTime()
{
    // Wait till time-warp point to reduce latency.
    ovr_WaitTillTime(m_frameTiming.TimewarpPointSeconds);

    // Get time warp properties
    for (int eyeIndex = 0; eyeIndex < ovrEye_Count; ++eyeIndex)
    {
        ovrHmd_GetEyeTimewarpMatrices(m_hmdDevice, (ovrEyeType)eyeIndex, m_headPose[eyeIndex], m_timeWarpMatrices[eyeIndex]);
    }
}

osg::Camera *OculusDevice::createRTTCamera(osg::Texture *texture, OculusDevice::Eye eye, osg::Transform::ReferenceFrame referenceFrame, osg::GraphicsContext *gc) const
{
    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setClearColor(osg::Vec4(0.2f, 0.2f, 0.4f, 1.0f));
    camera->setClearMask(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLenum buffer = (gc && gc->getTraits()->doubleBuffer) ? GL_BACK : GL_FRONT;
    camera->setDrawBuffer(buffer);
    camera->setReadBuffer(buffer);

    camera->setRenderTargetImplementation(osg::Camera::FRAME_BUFFER_OBJECT);
    camera->setRenderOrder(osg::Camera::PRE_RENDER, renderOrder(eye));
    camera->setAllowEventFocus(false);
    camera->setReferenceFrame(referenceFrame);

    if (gc)
    {
        camera->setGraphicsContext(gc);
    }

    if (texture)
    {
        texture->setFilter(osg::Texture2D::MIN_FILTER, osg::Texture2D::LINEAR);
        texture->setFilter(osg::Texture2D::MAG_FILTER, osg::Texture2D::LINEAR);
        camera->setViewport(0, 0, texture->getTextureWidth(), texture->getTextureHeight());
        camera->attach(osg::Camera::COLOR_BUFFER, texture, 0, 0, false, 4, 4);
    }

    return camera.release();
}

osg::Camera *OculusDevice::createWarpOrthoCamera(double left, double right, double bottom, double top, osg::GraphicsContext *gc) const
{
    osg::ref_ptr<osg::Camera> camera = new osg::Camera;
    camera->setClearColor(osg::Vec4(0.0f, 0.0f, 0.0f, 1.0f));
    camera->setClearMask(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
    camera->setRenderOrder(osg::Camera::POST_RENDER);
    camera->setAllowEventFocus(false);

    camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    camera->setProjectionMatrix(osg::Matrix::ortho2D(left, right, bottom, top));
    camera->getOrCreateStateSet()->setMode(GL_LIGHTING, osg::StateAttribute::OFF);

    if (gc)
    {
        camera->setGraphicsContext(gc);
    }
    return camera.release();
}

osg::Program *OculusDevice::createShaderProgram() const
{
    // Set up shaders from the Oculus SDK documentation
    osg::ref_ptr<osg::Program> program = new osg::Program;
    osg::ref_ptr<osg::Shader> vertexShader = new osg::Shader(osg::Shader::VERTEX);

    if (m_useTimeWarp)
    {
        vertexShader->setShaderSource(m_warpWithTimewarpVertexShaderSource);
    }
    else
    {
        vertexShader->setShaderSource(m_warpVertexShaderSource);
    }

    osg::ref_ptr<osg::Shader> fragmentShader = new osg::Shader(osg::Shader::FRAGMENT);
    fragmentShader->setShaderSource(m_warpFragmentShaderSource);
    program->addShader(vertexShader);
    program->addShader(fragmentShader);
    return program.release();
}

void OculusDevice::applyShaderParameters(osg::StateSet *stateSet, osg::Program *program, osg::Texture2D *texture, OculusDevice::Eye eye) const
{
    stateSet->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    stateSet->setAttributeAndModes(program, osg::StateAttribute::ON);
    stateSet->addUniform(new osg::Uniform("Texture", 0));
    stateSet->addUniform(new osg::Uniform("EyeToSourceUVScale", eyeToSourceUVScale(eye)));
    stateSet->addUniform(new osg::Uniform("EyeToSourceUVOffset", eyeToSourceUVOffset(eye)));

    // Uniforms needed for time warp
    if (m_useTimeWarp)
    {
        osg::ref_ptr<osg::Uniform> eyeRotationStart = new osg::Uniform("EyeRotationStart", this->eyeRotationStart(eye));
        osg::ref_ptr<osg::Uniform> eyeRotationEnd = new osg::Uniform("EyeRotationEnd", this->eyeRotationEnd(eye));
        stateSet->addUniform(eyeRotationStart);
        stateSet->addUniform(eyeRotationEnd);
        eyeRotationStart->setUpdateCallback(new EyeRotationCallback(EyeRotationCallback::START, this, eye));
        eyeRotationEnd->setUpdateCallback(new EyeRotationCallback(EyeRotationCallback::END, this, eye));
    }
}

void WarpCameraPreDrawCallback::operator()(osg::RenderInfo &) const
{
    // Wait till time - warp point to reduce latency.
    m_device->waitTillTime();
}
void EyeRotationCallback::operator()(osg::Uniform *uniform, osg::NodeVisitor *)
{
    if (m_mode == START)
    {
        uniform->set(m_device->eyeRotationStart(m_eye));
    }
    else if (m_mode == END)
    {
        uniform->set(m_device->eyeRotationEnd(m_eye));
    }
}
