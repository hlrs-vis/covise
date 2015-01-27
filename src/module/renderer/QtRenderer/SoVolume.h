/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _SO_VOLUME_
#define _SO_VOLUME_

#include <util/coTypes.h>
#include <Inventor/SbLinear.h>
#include <Inventor/fields/SoSFFloat.h>
#include <Inventor/nodes/SoShape.h>
#include <virvo/vvrenderer.h>
#include <virvo/vvvoldesc.h>

/** Class: SoVolume
    This class implements a volume rendering node for Inventor.
    It is part of the COVISE system.
    The original version of this file is 'SoCube'.
    Default size is -1 to +1 in all 3 dimensions, but the
    width, height, and depth fields can be used to change these.
    @author Uwe Woessner
    @author Juergen Schulze-Doebold
*/
class SoVolume : public SoShape
{
    SO_NODE_HEADER(SoVolume);

public:
    // Fields
    SoSFFloat width; ///< Size in x dimension
    SoSFFloat height; ///< Size in y dimension
    SoSFFloat depth; ///< Size in z dimension

    /// Constructor
    SoVolume();

    SoEXTENDER public :
        // Implements actions
        virtual void
        GLRender(SoGLRenderAction *action);
    virtual void rayPick(SoRayPickAction *action);
    virtual void init(int, int, int,
                      float, float, float, float, float, float,
                      int, float *, float *, float *, uchar *pc, uchar *byteData,
                      int no_of_lut_entries = 0, const uchar *rgbalut = NULL);

    SoINTERNAL public : static void initClass();

protected:
    /// Generates triangles representing a cube
    virtual void generatePrimitives(SoAction *action);

    /// Computes bounding box of cube
    virtual void computeBBox(SoAction *action, SbBox3f &box,
                             SbVec3f &center);

    /// Overrides standard method to create an SoVolumeDetail instance
    virtual SoDetail *createTriangleDetail(SoRayPickAction *action,
                                           const SoPrimitiveVertex *v1,
                                           const SoPrimitiveVertex *v2,
                                           const SoPrimitiveVertex *v3,
                                           SoPickedPoint *pp);

    virtual ~SoVolume();

private:
    static int numberOfInstances; ///< Number of SoVolume instances
    bool useGlobalLut; ///< use viewer global lut
    SbVec3f coords[8]; ///< Corner coordinates
    SbVec2f texCoords[4]; ///< Face corner texture coordinates
    SbVec3f normals[6]; ///< Face normals
    SbVec3f edgeNormals[12]; ///< Edge normals (for wire-frame)
    SbVec3f *verts[6][4]; ///< Vertex references to coords
    vvRenderer *renderer; ///< volume renderer
    vvRenderState renderState; ///< state of volume renderer
    vvVolDesc *vd; ///< volume description

    /** This flag indicates whether picking is done on a real cube or a
          cube that is just a bounding box representing another shape. If
          this flag is TRUE, a pick on the cube should not generate a
          detail, since the bounding box is not really in the picked path.
      */
    SbBool pickingBoundingBox;

    /// Returns TRUE if per face materials are specified
    SbBool isMaterialPerFace(SoAction *action) const;

    /// Computes number of divisions per side based on complexity
    int computeNumDivisions(SoAction *action) const;

    /// Computes real half-width, -height, -depth
    void getSize(float &hWidth,
                 float &hHeight,
                 float &hDepth) const;

    /// These render the volume
    void GLRenderGeneric(SoGLRenderAction *action,
                         SbBool sendNormals, SbBool doTextures);
    void GLRenderNvertTnone(SoGLRenderAction *action);

    /** Renders or picks cube representing given bounding box. These
          are used by SoShape to implement BOUNDING_BOX complexity.
      */
    void GLRenderBoundingBox(SoGLRenderAction *action,
                             const SbBox3f &bbox);
    void rayPickBoundingBox(SoRayPickAction *action,
                            const SbBox3f &bbox);

    /// SoShape needs to get at the above methods
    friend class SoShape;
};
#endif
