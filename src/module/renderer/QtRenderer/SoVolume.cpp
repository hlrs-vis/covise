/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// for class documentation see SoVolume.h

#ifdef INVENTORRENDERER
#include "InvCoviseViewer.h"
#else
#ifndef YAC
#include "InvMain.h"
#else
#include "InvMain_yac.h"
#endif
#include "InvViewer.h"
#endif

#include <config/CoviseConfig.h>
#include <util/unixcompat.h>

#include <Inventor/SbBox.h>
#include <Inventor/SoPickedPoint.h>
#include <Inventor/SoPrimitiveVertex.h>
#include <Inventor/actions/SoGLRenderAction.h>
#include <Inventor/actions/SoRayPickAction.h>
#include <Inventor/bundles/SoMaterialBundle.h>
#include <Inventor/elements/SoComplexityElement.h>
#include <Inventor/elements/SoComplexityTypeElement.h>
#include <Inventor/elements/SoGLTextureCoordinateElement.h>
#include <Inventor/elements/SoGLTextureEnabledElement.h>
#include <Inventor/elements/SoLightModelElement.h>
#include <Inventor/elements/SoMaterialBindingElement.h>
#include <Inventor/elements/SoModelMatrixElement.h>
#include <Inventor/elements/SoCacheElement.h>
#include <Inventor/misc/SoState.h>

#include "InvDefs.h"

#include "SoVolumeDetail.h"
#include "SoVolume.h"

#include <util/coTypes.h>
#include <virvo/math/math.h>
#include <virvo/vvopengl.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvrendererfactory.h>

#include <util/unixcompat.h>

using std::cerr;
using std::endl;

//using namespace covise;

int SoVolume::numberOfInstances = 0;

SO_NODE_SOURCE(SoVolume);

//----------------------------------------------------------------------
/// Constructor
SoVolume::SoVolume()
{
    numberOfInstances++;
    useGlobalLut = false;

    float sq2;

    SO_NODE_CONSTRUCTOR(SoVolume);

    SO_NODE_ADD_FIELD(width, (2));
    SO_NODE_ADD_FIELD(height, (2));
    SO_NODE_ADD_FIELD(depth, (2));

    isBuiltIn = FALSE;

    renderer = NULL;
    vd = NULL;

    if (SO_NODE_IS_FIRST_INSTANCE())
    {
        // Initialize corner coordinate values
        coords[0].setValue(-1.0, 1.0, -1.0); // Left  Top    Back
        coords[1].setValue(1.0, 1.0, -1.0); // Right Top    Back
        coords[2].setValue(-1.0, -1.0, -1.0); // Left  Bottom Back
        coords[3].setValue(1.0, -1.0, -1.0); // Right Bottom Back
        coords[4].setValue(-1.0, 1.0, 1.0); // Left  Top    Front
        coords[5].setValue(1.0, 1.0, 1.0); // Right Top    Front
        coords[6].setValue(-1.0, -1.0, 1.0); // Left  Bottom Front
        coords[7].setValue(1.0, -1.0, 1.0); // Right Bottom Front

        // Initialize face vertices to point into coords. The order of
        // vertices around the faces is chosen so that the texture
        // coordinates match up: texture coord (0,0) is at the first
        // vertex and (1,1) is at the third. The vertices obey the
        // right-hand rule for each face.
        verts[1][2] = verts[2][3] = verts[4][3] = &coords[0];
        verts[1][3] = verts[3][2] = verts[4][2] = &coords[1];
        verts[1][1] = verts[2][0] = verts[5][0] = &coords[2];
        verts[1][0] = verts[3][1] = verts[5][1] = &coords[3];
        verts[0][3] = verts[2][2] = verts[4][0] = &coords[4];
        verts[0][2] = verts[3][3] = verts[4][1] = &coords[5];
        verts[0][0] = verts[2][1] = verts[5][3] = &coords[6];
        verts[0][1] = verts[3][0] = verts[5][2] = &coords[7];

        // Initialize texture coordinates. These are for the 4 corners of
        // each face, starting at the lower left corner
        texCoords[0].setValue(0.0, 0.0);
        texCoords[1].setValue(1.0, 0.0);
        texCoords[2].setValue(1.0, 1.0);
        texCoords[3].setValue(0.0, 1.0);

        // Initialize face normals
        normals[0].setValue(0.0, 0.0, 1.0); // Front
        normals[1].setValue(0.0, 0.0, -1.0); // Back
        normals[2].setValue(-1.0, 0.0, 0.0); // Left
        normals[3].setValue(1.0, 0.0, 0.0); // Right
        normals[4].setValue(0.0, 1.0, 0.0); // Top
        normals[5].setValue(0.0, -1.0, 0.0); // Bottom

        // Initialize edge normals. These are used when drawing simple
        // wire-frame versions of the cube. The order of these matters,
        // since the rendering routine relies on it. Each normal is the
        // average of the face normals of the two adjoining faces, so the
        // edge is fairly-well lit in any forward-facing orientation.
        sq2 = sqrt(2.0) / 2.0;

        edgeNormals[0].setValue(0.0, -sq2, sq2); // Bottom front
        edgeNormals[1].setValue(sq2, 0.0, sq2); // Right  front
        edgeNormals[2].setValue(0.0, sq2, sq2); // Top    front
        edgeNormals[3].setValue(-sq2, 0.0, sq2); // Left   front
        edgeNormals[4].setValue(0.0, -sq2, -sq2); // Bottom rear
        edgeNormals[5].setValue(-sq2, 0.0, -sq2); // Left   rear
        edgeNormals[6].setValue(0.0, sq2, -sq2); // Top    rear
        edgeNormals[7].setValue(sq2, 0.0, -sq2); // Top    rear
        edgeNormals[8].setValue(-sq2, -sq2, 0.0); // Bottom left
        edgeNormals[9].setValue(sq2, -sq2, 0.0); // Bottom right
        edgeNormals[10].setValue(sq2, sq2, 0.0); // Top    right
        edgeNormals[11].setValue(-sq2, sq2, 0.0); // Top    left
    }
}

void SoVolume::initClass()
{
#ifdef __COIN__
    SO_NODE_INIT_CLASS(SoVolume, SoShape, "Shape");
#endif
}

//----------------------------------------------------------------------
/** Code needed for volume rendering: initialize a new volume object
    when new 3D scalar data arrives.
*/
void SoVolume::init(int xsize, int ysize, int zsize,
                    float xmin, float xmax, float ymin, float ymax, float zmin, float zmax,
                    int colorpacking, float *r, float *g, float *b, uchar *pc, uchar *byteData,
                    int no_of_lut_entries, const uchar *rgbalut)
{

    (void)no_of_lut_entries;
    (void)rgbalut;

    virvo::vec3f pos;
    std::string rendererName;

    //vvDebugMsg::setDebugLevel(vvDebugMsg::NO_MESSAGES);
    //vvDebugMsg::setDebugLevel(vvDebugMsg::ALL_MESSAGES);

    if (vd != NULL)
        delete vd;
    vd = NULL;
    if (renderer != NULL)
    {
        renderState = *renderer;
        delete renderer;
    }
    renderer = NULL;

    if (colorpacking == INV_RGBA && pc)
    {
        vvDebugMsg::msg(1, "RGBA data received.");
        vd = new vvVolDesc("COVISE", xsize, ysize, zsize, 1, 1, 4, &pc);
    }
    else if (r && g && b)
    {
        vvDebugMsg::msg(1, "RGB data received.");
        vd = new vvVolDesc("COVISE", xsize, ysize, zsize, 1, &r, &g, &b);
    }
    else if (r || byteData)
    {
        vvDebugMsg::msg(1, "Density data received.");
        if (byteData)
            vd = new vvVolDesc("COVISE", xsize, ysize, zsize, 1, 1, 1, &byteData);
        else
            vd = new vvVolDesc("COVISE", xsize, ysize, zsize, 1, &r);

#ifdef INVENTORRENDERER
        if (vd->tf[0].isEmpty())
        {
            vd->tf[0].setDefaultColors((vd->getBPV() < 3) ? 0 : 3, 0.0, 1.0);
            vd->tf[0].setDefaultAlpha(0, 0.0, 1.0);
        }
#else
#ifdef OLD_VIRVO
        if (no_of_lut_entries > 0 && rgbalut)
        {
            vd->tf[0].setLUT(no_of_lut_entries, rgbalut);
        }

        if (vd->tf[0].getType() == vvTransFunc::PINS_1D && vd->tf[0].isEmpty())
        {
#endif
            no_of_lut_entries = ::renderer->viewer->getNumGlobalLutEntries();
            rgbalut = ::renderer->viewer->getGlobalLut();
#ifdef OLD_VIRVO
            useGlobalLut = true;
            fprintf(stderr, "using global LUT when available\n");
            if (no_of_lut_entries > 0 && rgbalut)
            {
                vd->tf[0].setLUT(no_of_lut_entries, rgbalut);
            }
        }
#endif
#endif
    }
    else
    {
        vvDebugMsg::msg(1, "No data received!");
    }

    if (!vd)
    {
        vvDebugMsg::msg(1, "Volume description is NULL!");
        return;
    }

    vd->pos[0] = (xmax + xmin) / 2.;
    vd->pos[1] = (ymax + ymin) / 2.;
    vd->pos[2] = (zmax + zmin) / 2.;

    vd->dist[0] = (xmax - xmin) / xsize;
    vd->dist[1] = (ymax - ymin) / ysize;
    vd->dist[2] = (zmax - zmin) / zsize;

    std::cerr << "voldesc: vox=(" << vd->vox[0] << " " << vd->vox[1] << " " << vd->vox[2]
              << "), dist=(" << vd->dist[0] << " " << vd->dist[1] << " " << vd->dist[2] << ")" << std::endl;

    width.setValue(vd->vox[0] * vd->dist[0]);
    height.setValue(vd->vox[1] * vd->dist[1]);
    depth.setValue(vd->vox[2] * vd->dist[2]);

    vvDebugMsg::msg(1, "Byte per voxel: ", static_cast<int>(vd->getBPV()));

// Set default transfer function if none is present yet:
#ifdef OLD_VIRVO
    if (vd->tf[0].getType() == vvTransFunc::PINS_1D && vd->tf[0].isEmpty())
    {
#endif
        vvDebugMsg::msg(1, "No transfer function specified -- choosing default one.");
        vd->tf[0].setDefaultColors((vd->getBPV() < 3) ? 0 : 3, 0.0, 1.0);
        vd->tf[0].setDefaultAlpha(0, 0.0, 1.0);
#ifdef OLD_VIRVO
    }
#endif

    pos = virvo::vec3f((xmax + xmin) * 0.5f,
                       (ymax + ymin) * 0.5f,
                       (zmax + zmin) * 0.5f);
    vd->pos = pos;

    rendererName = covise::coCoviseConfig::getEntry("Renderer.VolumeRenderer");

    const char *vox = NULL;
    if (strncasecmp(rendererName.c_str(), "preint", 6) == 0)
        vox = "arb";
    else if (strncasecmp(rendererName.c_str(), "fragprog", 8) == 0)
        vox = "arb";

    renderer = vvRendererFactory::create(vd, renderState, rendererName.c_str(), vox);

    if (strncasecmp(rendererName.c_str(), "PreInt", 6) == 0)
    {
        renderer->setParameter(vvRenderer::VV_PREINT, 1.0f);
    }
    renderer->setParameter(vvRenderState::VV_BOUNDARIES, false);
    //renderer->setParameter(vvRenderer::VV_SLICEORIENT, vvTexRend::VV_VIEWPLANE);
    float quality = 2.f;

#ifndef INVENTORRENDERER
    quality = ::renderer->viewer->getVolumeSamplingAccuracy();
    ::renderer->viewer->enableRightWheelSampleControl(true);
#endif

    renderer->setParameter(vvRenderState::VV_QUALITY, quality);
}

//----------------------------------------------------------------------
/// Destructor
SoVolume::~SoVolume()
{
    delete renderer;
    renderer = NULL;

    delete vd;
    vd = NULL;

    numberOfInstances--;

#ifndef INVENTORRENDERER
    if (numberOfInstances <= 0)
    {
        ::renderer->viewer->enableRightWheelSampleControl(false);
    }
#endif
}

//----------------------------------------------------------------------
/// Performs GL rendering of a volume dataset.
void SoVolume::GLRender(SoGLRenderAction *action)
{
    SoCacheElement::invalidate(action->getState());
#ifndef INVENTORRENDERER
#ifdef OLD_VIRVO
    if (useGlobalLut && ::renderer->viewer->isGlobalLutUpdated())
    {
        if (vd)
        {
            vd->tf[0].setLUT(::renderer->viewer->getNumGlobalLutEntries(),
                             ::renderer->viewer->getGlobalLut());
            renderer->updateTransferFunction();
        }
    }
#endif
#endif

    // First see if the object is visible and should be rendered now
    if (!shouldGLRender(action))
        return;

    // See if texturing is enabled
    SbBool doTextures = SoGLTextureEnabledElement::get(action->getState());

    // Render the cube. The GLRenderGeneric() method handles any
    // case. The GLRenderNvertTnone() handles the case where we are
    // outputting normals but no texture coordinates. This case is
    // handled separately since it occurs often and warrants its own
    // method.
    SbBool sendNormals = (SoLightModelElement::get(action->getState()) != SoLightModelElement::BASE_COLOR);
    GLRenderGeneric(action, sendNormals, doTextures);
}

//----------------------------------------------------------------------
/** Implements ray picking. We could just use the default mechanism,
    generating primitives, but this would be inefficient if the
    complexity is above 0.5. Therefore, we make sure that the
    complexity is low and then use the primitive generation.
*/
void SoVolume::rayPick(SoRayPickAction *action)
{
    // First see if the object is pickable
    if (!shouldRayPick(action))
        return;

    // Save the state so we don't affect the real complexity
    action->getState()->push();

    // Change the complexity
    SoComplexityElement::set(action->getState(), 0.0);
    SoComplexityTypeElement::set(action->getState(),
                                 SoComplexityTypeElement::OBJECT_SPACE);

    // Pick using primitive generation. Make sure we know that we are
    // really picking on a real cube, not just a bounding box of
    // another shape.
    pickingBoundingBox = FALSE;
    SoShape::rayPick(action);

    // Restore the state
    action->getState()->pop();
}

//----------------------------------------------------------------------
/// Computes bounding box of volume dataset.
void SoVolume::computeBBox(SoAction *, SbBox3f &box, SbVec3f &center)
{
    virvo::vec3f pos(0.0, 0.0, 0.0);
    virvo::vec3f size(0.0, 0.0, 0.0);
    virvo::vec3f size2(0.0, 0.0, 0.0); // half size

    if (renderer)
    {
        pos = renderer->getPosition();
        size = vd->getSize();
    }
    size2 = size * virvo::vec3f(0.5f);
    box.setBounds(pos[0] - size2[0], pos[1] - size2[1], pos[2] - size2[2],
                  pos[0] + size2[0], pos[1] + size2[1], pos[2] + size2[2]);
    center.setValue(pos[0], pos[1], pos[2]);
}

//----------------------------------------------------------------------
/// Generates triangles representing a cube.
void SoVolume::generatePrimitives(SoAction *action)
{
    SbBool materialPerFace;
    int numDivisions, face, vert;
    float s;
    SbVec3f pt;
    float w, h, d;
    SbVec4f tex(0., 0., 0., 0.);
    SbBool genTexCoords;
    SoPrimitiveVertex pv;
    SoVolumeDetail detail;
    const SoTextureCoordinateElement *tce = NULL;

    materialPerFace = isMaterialPerFace(action);
    numDivisions = computeNumDivisions(action);

    pv.setDetail(&detail);

    // Determine whether we should generate our own texture coordinates
    switch (SoTextureCoordinateElement::getType(action->getState()))
    {
    case SoTextureCoordinateElement::EXPLICIT:
        genTexCoords = TRUE;
        break;
    case SoTextureCoordinateElement::FUNCTION:
        genTexCoords = FALSE;
        break;
    default:
        fprintf(stderr, "SoVolume::generatePrimitives(): genTexCoords used uninitialized\n");
        genTexCoords = TRUE;
        break;
    }

    // If we're not generating our own coordinates, we'll need the
    // texture coordinate element to get coords based on points/normals.
    if (!genTexCoords)
        tce = SoTextureCoordinateElement::getInstance(action->getState());
    else
    {
        tex[2] = 0.0;
        tex[3] = 1.0;
    }

    getSize(w, h, d);

    for (face = 0; face < 6; face++)
    {

        if (face == 0 || materialPerFace)
            pv.setMaterialIndex(face);
        pv.setNormal(normals[face]);
        detail.setPart(face);

        // Simple case of one polygon per face
        if (numDivisions == 1)
        {
            beginShape(action, TRIANGLE_STRIP);
            vert = 3;
            pt.setValue((*verts[face][vert])[0] * w,
                        (*verts[face][vert])[1] * h,
                        (*verts[face][vert])[2] * d);
            if (genTexCoords)
            {
                tex[0] = texCoords[vert][0];
                tex[1] = texCoords[vert][1];
            }
            else
                tex = tce->get(pt, normals[face]);
            pv.setPoint(pt);
            pv.setTextureCoords(tex);
            shapeVertex(&pv);
            vert = 0;
            pt.setValue((*verts[face][vert])[0] * w,
                        (*verts[face][vert])[1] * h,
                        (*verts[face][vert])[2] * d);
            if (genTexCoords)
            {
                tex[0] = texCoords[vert][0];
                tex[1] = texCoords[vert][1];
            }
            else
                tex = tce->get(pt, normals[face]);
            pv.setPoint(pt);
            pv.setTextureCoords(tex);
            shapeVertex(&pv);
            vert = 2;
            pt.setValue((*verts[face][vert])[0] * w,
                        (*verts[face][vert])[1] * h,
                        (*verts[face][vert])[2] * d);
            if (genTexCoords)
            {
                tex[0] = texCoords[vert][0];
                tex[1] = texCoords[vert][1];
            }
            else
                tex = tce->get(pt, normals[face]);
            pv.setPoint(pt);
            pv.setTextureCoords(tex);
            shapeVertex(&pv);
            vert = 1;
            pt.setValue((*verts[face][vert])[0] * w,
                        (*verts[face][vert])[1] * h,
                        (*verts[face][vert])[2] * d);
            if (genTexCoords)
            {
                tex[0] = texCoords[vert][0];
                tex[1] = texCoords[vert][1];
            }
            else
                tex = tce->get(pt, normals[face]);
            pv.setPoint(pt);
            pv.setTextureCoords(tex);
            shapeVertex(&pv);
            endShape();
        }

        // More than one polygon per face
        else
        {
            float di = 1.0 / numDivisions;
            SbVec3f topPoint, botPoint, nextBotPoint;
            SbVec3f horizSpace, vertSpace;
            int strip, rect;

            botPoint = *verts[face][0];

            // Compute spacing between adjacent points in both directions
            horizSpace = di * (*verts[face][1] - botPoint);
            vertSpace = di * (*verts[face][3] - botPoint);

            // For each horizontal strip
            for (strip = 0; strip < numDivisions; strip++)
            {

                // Compute current top point. Save it to use as bottom
                // of next strip
                nextBotPoint = topPoint = botPoint + vertSpace;

                beginShape(action, TRIANGLE_STRIP);

                // Send points at left end of strip
                s = 0.0;
                pt = topPoint;
                pt[0] *= w;
                pt[1] *= h;
                pt[2] *= d;
                if (genTexCoords)
                {
                    tex[0] = s;
                    tex[1] = (strip + 1) * di;
                }
                else
                    tex = tce->get(pt, normals[face]);
                pv.setPoint(pt);
                pv.setTextureCoords(tex);
                shapeVertex(&pv);
                pt = botPoint;
                pt[0] *= w;
                pt[1] *= h;
                pt[2] *= d;
                if (genTexCoords)
                {
                    tex[0] = s;
                    tex[1] = strip * di;
                }
                else
                    tex = tce->get(pt, normals[face]);
                pv.setPoint(pt);
                pv.setTextureCoords(tex);
                shapeVertex(&pv);

                // For each rectangular piece of strip
                for (rect = 0; rect < numDivisions; rect++)
                {

                    // Go to next rect
                    topPoint += horizSpace;
                    botPoint += horizSpace;
                    s += di;

                    // Send points at right side of rect
                    pt = topPoint;
                    pt[0] *= w;
                    pt[1] *= h;
                    pt[2] *= d;
                    if (genTexCoords)
                    {
                        tex[0] = s;
                        tex[1] = (strip + 1) * di;
                    }
                    else
                        tex = tce->get(pt, normals[face]);
                    pv.setPoint(pt);
                    pv.setTextureCoords(tex);
                    shapeVertex(&pv);
                    pt = botPoint;
                    pt[0] *= w;
                    pt[1] *= h;
                    pt[2] *= d;
                    if (genTexCoords)
                    {
                        tex[0] = s;
                        tex[1] = strip * di;
                    }
                    else
                        tex = tce->get(pt, normals[face]);
                    pv.setPoint(pt);
                    pv.setTextureCoords(tex);
                    shapeVertex(&pv);
                }

                endShape();

                // Get ready for next strip
                botPoint = nextBotPoint;
            }
        }
    }
}

//----------------------------------------------------------------------
/// Macro to multiply out coordinates to avoid extra GL calls
#define SCALE(pt) (tmp[0] = (pt)[0] * scale[0], tmp[1] = (pt)[1] * scale[1], \
                   tmp[2] = (pt)[2] * scale[2], tmp)

//----------------------------------------------------------------------
/** Generic rendering of volume with or without normals, with or
    without texture coordinates.
*/
void SoVolume::GLRenderGeneric(SoGLRenderAction *action,
                               SbBool sendNormals, SbBool doTextures)
{
    (void)action;
    (void)sendNormals;
    (void)doTextures;
    InvViewer::DrawStyle drawStyle;
#ifdef INVENTORRENDERER
    if (coviseViewer->getCurViewer()->getInteractiveCount() > 0)
    {
        drawStyle = coviseViewer->getCurViewer()->getDrawStyle(InvViewer::INTERACTIVE);
    }
    else
    {
        drawStyle = coviseViewer->getCurViewer()->getDrawStyle(InvViewer::STILL);
    }
#else
    if (::renderer->viewer->getInteractiveCount() > 0)
    {
        drawStyle = ::renderer->viewer->getDrawStyle(InvViewer::INTERACTIVE);
    }
    else
    {
        drawStyle = ::renderer->viewer->getDrawStyle(InvViewer::STILL);
    }
#endif

    if (renderer)
    {
        float q = .5f;
#ifdef INVENTORRENDERER
        if (drawStyle == InvViewer::VIEW_LOW_VOLUME)
            q = .5f;
        else
        {
            q = coviseViewer->getCurViewer()->getSamplingRate();
        }
#else
        q = ::renderer->viewer->getVolumeSamplingAccuracy();
        (void)drawStyle;
#endif
        renderer->setParameter(vvRenderState::VV_QUALITY, q);
        glMatrixMode(GL_MODELVIEW_MATRIX);
        glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);
        renderer->renderVolumeGL();
    }
}

//----------------------------------------------------------------------
/// Renders volume with normals and without texture coordinates.
void SoVolume::GLRenderNvertTnone(SoGLRenderAction *action)
{
    (void)action;
    InvViewer::DrawStyle drawStyle;
#ifdef INVENTORRENDERER
    if (coviseViewer->getCurViewer()->getInteractiveCount() > 0)
    {
        drawStyle = coviseViewer->getCurViewer()->getDrawStyle(InvViewer::INTERACTIVE);
    }
    else
    {
        drawStyle = coviseViewer->getCurViewer()->getDrawStyle(InvViewer::STILL);
    }
#else
    if (::renderer->viewer->getInteractiveCount() > 0)
    {
        drawStyle = ::renderer->viewer->getDrawStyle(InvViewer::INTERACTIVE);
    }
    else
    {
        drawStyle = ::renderer->viewer->getDrawStyle(InvViewer::STILL);
    }
#endif

    if (renderer)
    {
        float q = .5f;
#ifdef INVENTORRENDERER
        if (drawStyle == InvViewer::VIEW_LOW_VOLUME)
            q = .5f;
        else
        {
            q = coviseViewer->getCurViewer()->getSamplingRate();
        }
#else
        q = ::renderer->viewer->getVolumeSamplingAccuracy();
        (void)drawStyle;
#endif

        renderer->setParameter(vvRenderState::VV_QUALITY, q);
        glMatrixMode(GL_MODELVIEW_MATRIX);
        glTranslatef(vd->pos[0], vd->pos[1], vd->pos[2]);
        renderer->renderVolumeGL();
    }
}

//----------------------------------------------------------------------
/** Overrides standard method to create an SoVolumeDetail instance
    representing a picked intersection with a triangle that is half
    of the face of a cube.
*/
SoDetail *SoVolume::createTriangleDetail(SoRayPickAction *,
                                         const SoPrimitiveVertex *v1, const SoPrimitiveVertex *,
                                         const SoPrimitiveVertex *, SoPickedPoint *)
{
    SoVolumeDetail *detail;

    // Don't create a detail if the pick operation was performed on a
    // bounding box cube, not a real cube
    if (pickingBoundingBox)
        return NULL;

    detail = new SoVolumeDetail;

    // The part code should be the same in all three details, so just use one
    detail->setPart(((const SoVolumeDetail *)v1->getDetail())->getPart());

    return detail;
}

//----------------------------------------------------------------------
/// Returns TRUE if per face materials are specified.
SbBool SoVolume::isMaterialPerFace(SoAction *action) const
{
    SoMaterialBindingElement::Binding binding;

    binding = SoMaterialBindingElement::get(action->getState());

    return (binding == SoMaterialBindingElement::PER_PART || binding == SoMaterialBindingElement::PER_PART_INDEXED || binding == SoMaterialBindingElement::PER_FACE || binding == SoMaterialBindingElement::PER_FACE_INDEXED);
}

//----------------------------------------------------------------------
/// Computes number of divisions per side based on complexity.
int SoVolume::computeNumDivisions(SoAction *action) const
{
    int numDivisions;
    float complexity;

    switch (SoComplexityTypeElement::get(action->getState()))
    {
    case SoComplexityTypeElement::OBJECT_SPACE:
        // In object space, the number of divisions is greater than 1
        // only for complexity values > 0.5. The maximum value is 16,
        // when complexity = 1.
        complexity = SoComplexityElement::get(action->getState());
        numDivisions = (complexity <= 0.5 ? 1 : -14 + (int)(complexity * 30.0));
        break;

    case SoComplexityTypeElement::SCREEN_SPACE:
        // In screen space, the number of divisions is based on the
        // complexity and the size of the cube when projected onto the
        // screen.
        short maxSize;
        {
            SbVec3f p;
            SbVec2s rectSize;

            getSize(p[0], p[1], p[2]);
            getScreenSize(action->getState(), SbBox3f(-p, p), rectSize);
            maxSize = (rectSize[0] > rectSize[1] ? rectSize[0] : rectSize[1]);
        }

        // Square complexity to get a more even increase in the number
        // of tesselation squares. Maximum bound is 1/4 the number of
        // pixels per side.
        complexity = SoComplexityElement::get(action->getState());
        numDivisions = 1 + (int)(0.25 * maxSize * complexity * complexity);
        break;

    case SoComplexityTypeElement::BOUNDING_BOX:
        // Most shapes do not have to handle this case, since it is
        // handled for them. However, since it is handled by drawing
        // the shape as a cube, the SoVolume class has to handle it.
        numDivisions = 1;
        break;
    default:
        fprintf(stderr, "SoVolume::computeNumDivisions(): unhandled complexity type\n");
        numDivisions = 1;
        break;
    }

    return numDivisions;
}

//----------------------------------------------------------------------
/// Computes real half-width, -height, -depth.
void SoVolume::getSize(float &hWidth, float &hHeight, float &hDepth) const
{
    hWidth = (width.isIgnored() ? 1.0 : width.getValue() / 2.0);
    hHeight = (height.isIgnored() ? 1.0 : height.getValue() / 2.0);
    hDepth = (depth.isIgnored() ? 1.0 : depth.getValue() / 2.0);
}

//----------------------------------------------------------------------
/** Does GL rendering of a volume representing the given bounding box.
    This is used for BOUNDING_BOX complexity. It does the minimum
    work necessary.
*/
void SoVolume::GLRenderBoundingBox(SoGLRenderAction *action, const SbBox3f &bbox)
{
    int face, vert;
    SoMaterialBundle mb(action);
    SbVec3f scale, tmp;

    // Make sure textures are disabled, just to speed things up
    action->getState()->push();
    SoGLTextureEnabledElement::set(action->getState(), FALSE);

    // Make sure first material is sent if necessary
    mb.sendFirst();

    // Scale and translate the cube to the correct spot
    const SbVec3f &translate = bbox.getCenter();
    SbVec3f size;
    bbox.getSize(size[0], size[1], size[2]);
    scale = 0.5f * size;

    for (face = 0; face < 6; face++)
    {
        if (!mb.isColorOnly())
            glNormal3fv(normals[face].getValue());

        glBegin(GL_POLYGON);

        for (vert = 0; vert < 4; vert++)
            glVertex3fv((SCALE(*verts[face][vert]) + translate).getValue());

        glEnd();
    }

    // Restore state
    action->getState()->pop();
}

//----------------------------------------------------------------------
/** Does picking of a cube representing the given bounding box. This
    is used for BOUNDING_BOX complexity. It uses the same code as
    for rayPick(), except that it makes sure the cube is transformed
    first to where the bounding box is.
*/
void SoVolume::rayPickBoundingBox(SoRayPickAction *action, const SbBox3f &bbox)
{
    // Save the state so we don't affect the real complexity
    action->getState()->push();

    // Change the complexity
    SoComplexityElement::set(action->getState(), 0.0);
    SoComplexityTypeElement::set(action->getState(),
                                 SoComplexityTypeElement::OBJECT_SPACE);

    // Change the current matrix to scale and translate the cube to the
    // correct spot. (We can't just use an extra matrix passed to
    // computeObjectSpaceRay(), since the points generated by
    // generatePrimitives() have to be transformed, not just the ray.)
    SbVec3f size;
    bbox.getSize(size[0], size[1], size[2]);

    // If any of the dimensions is 0, beef it up a little bit to avoid
    // scaling by 0
    if (size[0] == 0.0)
        size[0] = 0.00001f;
    if (size[1] == 0.0)
        size[1] = 0.00001f;
    if (size[2] == 0.0)
        size[2] = 0.00001f;

    SoModelMatrixElement::translateBy(action->getState(), this, bbox.getCenter());
    SoModelMatrixElement::scaleBy(action->getState(), this, 0.5f * size);

    // Compute the picking ray in the space of the shape
    computeObjectSpaceRay(action);

    // Pick using primitive generation. Make sure we know that we are
    // picking on just a bounding box of another shape, so details
    // won't be created.
    pickingBoundingBox = TRUE;
    generatePrimitives(action);

    // Restore the state
    action->getState()->pop();
}
