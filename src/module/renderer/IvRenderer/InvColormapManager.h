/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _INV_COLORMAP_MANAGER_H
#define _INV_COLORMAP_MANAGER_H

//**************************************************************************
//
// * Description    : Interctive COVISE desktop renderer
//
// * Class(es)      :
//
// * inherited from :
//
// * Author  : Dirk Rantzau
//
// * History : 24.07.97 V 1.0
//
//**************************************************************************

//
// OpenInventor stuff
//
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoNormalBinding.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoPackedColor.h>
#include <Inventor/nodes/SoMaterialBinding.h>
#include <Inventor/nodes/SoCamera.h>
#include <Inventor/nodes/SoGroup.h>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoPointSet.h>
#include <Inventor/nodes/SoDrawStyle.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoLabel.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoNormalBinding.h>
#include <Inventor/nodes/SoLightModel.h>
#include <Inventor/nodes/SoIndexedLineSet.h>
#include <Inventor/nodes/SoFaceSet.h>
#include <Inventor/nodes/SoQuadMesh.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <Inventor/nodes/SoIndexedTriangleStripSet.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/nodes/SoAnnotation.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/Xt/viewers/SoXtExaminerViewer.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoText3.h>
#include <Inventor/nodes/SoText2.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoFont.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoAsciiText.h>
#include <Inventor/nodes/SoCallback.h>
//
// renderer stuff
//
#include "InvObjectList.h"

class InvColormapManager
{

public:
    InvColormapManager();
    static SoCallbackCB updateCallback;
    void addColormap(const char *format, const char *name, int length,
                     float *r, float *g, float *b, float *a,
                     float min, float max, int ramps, char *annotation,
                     float x_0, float y_0, float size);
    int removeColormap(const char *name);
    void hideAllColormaps();
    void showColormap(const char *name, SoXtExaminerViewer *);
    void updateColormaps(void *, float x_0, float y_0, float size);
    char *currentColormap();
    SoNode *getRootNode();
    ~InvColormapManager();
    InvObjectList *getColormapList()
    {
        return colormap_list;
    }
    int getNumColormaps()
    {
        return numMaps;
    }

private:
    SoSeparator *colormap_root;
    SoSwitch *colormap_switch;
    SoOrthographicCamera *colormap_camera;
    SoCallback *colormap_callback;
    SoGroup *colormap_group;
    SoMaterial *colormap_material;
    SoTransform *colormap_transform;
    static short first_time;
    InvObjectList *colormap_list;
    char current_map[255];
    int numMaps;
    int maxColorLegend_;
};
#endif // _INV_COLORMAP_MANAGER_H
