/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <util/coTypes.h>
#include <Inventor/engines/SoSubEngine.h>
#include <Inventor/fields/SoSFVec3f.h>
#include <Inventor/fields/SoSFPlane.h>

class InvComposePlane : public SoEngine
{

    SO_ENGINE_HEADER(InvComposePlane);

public:
    // Input fields: a normal and a point
    SoSFVec3f normal;
    SoSFVec3f point;

    // The output is a vector
    SoEngineOutput plane; // (SoSFPlane) plane

    // Initializes this class for use in scene graphs. This
    // should be called after database initialization and before
    // any instance of this engine is constructed.
    static void initClass();

    // Constructor
    InvComposePlane();

private:
    // Destructor. Since engines are never deleted explicitly,
    // this can be private.
    virtual ~InvComposePlane();

    // Evaluation method
    virtual void evaluate();
};
