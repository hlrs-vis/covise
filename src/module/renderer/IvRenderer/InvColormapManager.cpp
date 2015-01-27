/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


//**************************************************************************
//
// * Description    : Inventor interactive COVISE renderer
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

#include <config/CoviseConfig.h>
#include <stdlib.h> // for system() and getenv()
#include <unistd.h> // for access()
#include <string.h>

#include "InvColormapManager.h"
#include "InvObjects.h"

//static const char *fontName = "Utopia-Regular";
//static const char *fontName = "Helvetica";
//static const char *fontName = "times.ttf";
static const char *fontName = "arial.ttf";

short InvColormapManager::first_time = 1;

//=========================================================================
//
//=========================================================================
InvColormapManager::InvColormapManager()
    : maxColorLegend_(8)
{
    // if we are in here the first time create the root node
    // under which all colormaps will be placed
    if (first_time == 1)
    {
        first_time = 0;
        colormap_root = new SoSeparator();
        colormap_switch = new SoSwitch();
        colormap_switch->whichChild.setValue(SO_SWITCH_NONE);
        strcpy(current_map, "NONE");
        colormap_camera = new SoOrthographicCamera;
        colormap_callback = new SoCallback;
        colormap_callback->setCallback(updateCallback, this);
        colormap_material = new SoMaterial;
        colormap_material->diffuseColor.set1Value(0, 0.5, 1.0, 1.0);
        colormap_group = new SoGroup;
        colormap_root->addChild(colormap_camera);
        colormap_root->addChild(colormap_callback);
        colormap_root->addChild(colormap_material);
        colormap_root->addChild(colormap_group);
        colormap_group->addChild(colormap_switch);
        colormap_list = new InvObjectList();
        numMaps = 0;
    }
    // read whats needed for colormaps from the Renderer section of the config file

    // read from config file, leave alone if no entry
    maxColorLegend_ = covise::coCoviseConfig::getInt("Renderer.MaxColorLegend", maxColorLegend_);

    //cerr << "InvColormapManager::InvColormapManager() got MaxColorLegend " << maxColorLegend_ << endl;
}

void InvColormapManager::updateCallback(void *userdata, SoAction *ac)
{
    // fix colormap to lower left corner
    const SoGLRenderAction *gl = dynamic_cast<SoGLRenderAction *>(ac);
    if (!gl)
        return;
    const InvColormapManager *cm = static_cast<InvColormapManager *>(userdata);
    if (!cm)
        return;
    if (!cm->colormap_switch)
        return;
    if (!cm->colormap_camera)
        return;
    const SbViewportRegion vp = gl->getViewportRegion();
    const float aspect = vp.getViewportAspectRatio();
    const SbVec2s vpSize = vp.getViewportSizePixels();
    const int xMargin = 40; // 20 Pixels at each side as our rectangle has margins on both sides ...
    const int yMargin = 60; // 30 Pixels at each side as our rectangle has margins on both sides ...
    float xMarginRel = 0.;
    float yMarginRel = 0.;
    if (vpSize[0] > 2 * xMargin)
        xMarginRel = (float)(vpSize[0] - xMargin) / (vpSize[0]);
    if (vpSize[1] > 2 * yMargin)
        yMarginRel = (float)(vpSize[1] - yMargin) / (vpSize[1]);

    for (int i = 0; i < cm->colormap_switch->getNumChildren(); ++i)
    {
        const SoGroup *root = dynamic_cast<SoGroup *>(cm->colormap_switch->getChild(i));
        if (!root)
            continue;
        if (root->getNumChildren() < 1)
            continue;
        SoTranslation *trans = dynamic_cast<SoTranslation *>(root->getChild(0));
        if (!trans)
            continue;
        if (aspect > 1.)
            trans->translation.setValue(-aspect * xMarginRel, -0.95 * yMarginRel, 0.0);
        else
            trans->translation.setValue(-1.0 * xMarginRel, -0.95 / aspect * yMarginRel, 0.0);
    }
}

//=========================================================================
//
//=========================================================================
void InvColormapManager::updateColormaps(void * /*data*/, float x_0, float y_0, float size)
{
    (void)x_0;
    (void)y_0;
    (void)size;
    SbViewVolume volume;
    //SoXtExaminerViewer *viewer = (SoXtExaminerViewer *)data;

    if (colormap_switch->whichChild.getValue() != SO_SWITCH_NONE)
    {

        // cerr << "Switching colormap to new position" << endl;
    }
}

//=========================================================================
//
//=========================================================================
SoNode *InvColormapManager::getRootNode()
{
    return colormap_root;
}

//=========================================================================
//
//=========================================================================
void InvColormapManager::addColormap(const char *format, const char *name,
                                     int length,
                                     float *r, float *g, float *b, float *a,
                                     float min, float max, int ramps,
                                     char *annotation,
                                     float x_0, float y_0, float size)
{

    char medc[40] = "";
    char minc[20] = "";
    char maxc[20] = "";
    const char *numFormat = NULL;

    if ((NULL == format) || ('\0' == *format))
    {
        if (((max - min) < 0.1) || ((max - min) > 1e5))
        {
            numFormat = "%10.5e";
        }
        else
        {
            numFormat = "%.3f";
        }
        snprintf(minc, sizeof(minc), numFormat, min);
        snprintf(maxc, sizeof(maxc), numFormat, max);

        // check whether min and max look same in color bar
        float testMin, testMax;
        int ret = sscanf(minc, "%f", &testMin);
        if (ret != 1)
        {
            cerr << "InvColormapManager::addColormap: sscanf1 failed" << endl;
        }
        ret = sscanf(maxc, "%f", &testMax);
        if (ret != 1)
        {
            cerr << "InvColormapManager::addColormap: sscanf2 failed" << endl;
        }
        if (testMin == testMax)
        {
            numFormat = "%.3e";
            sprintf(minc, numFormat, min);
            sprintf(maxc, numFormat, max);
        }
        ret = sscanf(minc, "%f", &testMin);
        if (ret != 1)
        {
            cerr << "InvColormapManager::addColormap: sscanf3 failed" << endl;
        }
        ret = sscanf(maxc, "%f", &testMax);
        if (ret != 1)
        {
            cerr << "InvColormapManager::addColormap: sscanf4 failed" << endl;
        }
        if (testMin == testMax)
        {
            numFormat = "%g";
        }
    }
    else
    {
        //Format is user defined
        numFormat = format;
    }
    sprintf(minc, numFormat, min);
    sprintf(maxc, numFormat, max);

    SoSeparator *root = new SoSeparator;
    SoTranslation *trans = new SoTranslation;
    SoLabel *objName = new SoLabel;
    SoDrawStyle *drawstyle = new SoDrawStyle;
    SoLightModel *lightmodel = new SoLightModel;
    SoMaterialBinding *matbind = new SoMaterialBinding;
    SoMaterial *material = new SoMaterial;
    SoCoordinate3 *coord = new SoCoordinate3;
    SoQuadMesh *quadmesh = new SoQuadMesh;
    SoShapeHints *shapehints = new SoShapeHints;

    SoSeparator *minSep = new SoSeparator;
    SoSeparator *medSep;
    SoSeparator *annSep = new SoSeparator;
    SoMaterialBinding *minmatb = new SoMaterialBinding;
    SoMaterialBinding *medmatb;
    SoMaterialBinding *maxmatb = new SoMaterialBinding;
    SoMaterialBinding *annmatb = new SoMaterialBinding;
    SoMaterial *minmat = new SoMaterial;
    SoMaterial *medmat;
    SoMaterial *maxmat = new SoMaterial;
    SoMaterial *annmat = new SoMaterial;
    SoFont *minFont = new SoFont;
    SoFont *medFont;
    SoFont *maxFont = new SoFont;
    SoFont *annFont = new SoFont;
    SoTranslation *minTrans = new SoTranslation;
    SoTranslation *medTrans;
    SoScale *medScale;
    SoTranslation *maxTrans = new SoTranslation;
    SoTranslation *annTrans = new SoTranslation;
    SoText2 *maxText = new SoText2;
    SoText2 *medText;
    SoText2 *minText = new SoText2;
    SoText2 *annText = new SoText2;

    // positioning is now done in updateCallback
    x_0 = y_0 = 0.;

    float fontSize = 14.0;
    //float fontSize=64.0;
    /*
   if ( maxColorLegend_ <= 10 )                              fontSize = 12.0;
   if (( maxColorLegend_ > 10 ) && (maxColorLegend_ <= 20) ) fontSize = 10.0;
   if (  maxColorLegend_ > 20 )                              fontSize = 8.0;
*/
    minFont->size.setValue(fontSize);
    maxFont->size.setValue(fontSize);
    annFont->size.setValue(fontSize);

    minFont->name.setValue(fontName);
    maxFont->name.setValue(fontName);
    annFont->name.setValue(fontName);

    objName->label.setValue(name);
    drawstyle->style = SoDrawStyle::FILLED;
    matbind->value = SoMaterialBinding::PER_FACE;
    lightmodel->model = SoLightModel::BASE_COLOR;

    minmatb->value = SoMaterialBinding::OVERALL;
    maxmatb->value = SoMaterialBinding::OVERALL;
    annmatb->value = SoMaterialBinding::OVERALL;

    minmat->diffuseColor.set1Value(0, 1.0, 1.0, 1.0);
    maxmat->diffuseColor.set1Value(0, 1.0, 1.0, 1.0);
    annmat->diffuseColor.set1Value(0, 1.0, 1.0, 1.0);

    annTrans->translation.setValue(x_0 + 0.0, y_0 - size * 0.10, 0.0);
    minTrans->translation.setValue(x_0 + 0.12, y_0 + 0.01, 0.0);
    maxTrans->translation.setValue(x_0 + 0.12, y_0 + size * 0.98, 0.0);

    minText->string.setValue(minc);
    maxText->string.setValue(maxc);
    annText->justification = SoText2::LEFT;
    //fprintf(stderr, "annotation=%s\n", annotation);
    annText->string.setValue(annotation);

    trans->translation.setValue(x_0, y_0, 0.);
    root->addChild(trans);

    root->addChild(objName);
    root->addChild(drawstyle);
    root->addChild(lightmodel);
    root->addChild(matbind);
    root->addChild(material);
    root->addChild(coord);
    root->addChild(shapehints);
    root->addChild(quadmesh);

    root->addChild(minSep);
    minSep->addChild(minmatb);
    minSep->addChild(minmat);
    minSep->addChild(minFont);
    minSep->addChild(minTrans);
    minSep->addChild(minText);

    SoSeparator *lineSep = NULL;

    int numLeg = maxColorLegend_;
    if (length <= maxColorLegend_)
        numLeg = length;

    if ((ramps == 0) || (ramps > 10)) // continous colormap /aw:or fine-grained
    {

        int ii;
        int nn = numLeg;
        float delty = (size / nn);
        float ypos = y_0;

        float vert_x[2];
        float vert_y[2];
        float vert_z[2];
        vert_x[0] = x_0 + 0.08;
        vert_y[0] = ypos;
        vert_z[0] = 0;
        vert_x[1] = x_0 + 0.11;
        vert_y[1] = ypos;
        vert_z[1] = 0;
        int vertices[2];
        vertices[0] = 0;
        vertices[1] = 1;

        int indices[2];
        indices[0] = 0;
        indices[1] = 1;
        InvLine *tick = new InvLine;
        tick->setCoords(1, 2,
                        2, vert_x, vert_y, vert_z,
                        vertices, indices);
        float zeroo = 0;
        tick->setColors(1, &zeroo, &zeroo, &zeroo);

        lineSep = tick->getSeparator();

        root->addChild(lineSep);

        for (ii = 0; ii < numLeg; ++ii)
        {
            medSep = new SoSeparator;
            medmatb = new SoMaterialBinding;
            medmat = new SoMaterial;
            medFont = new SoFont;
            medTrans = new SoTranslation;
            medScale = new SoScale;
            medText = new SoText2;
            //cerr << " ramps : " << ramps << endl;
            //cerr << "--->set coordinates" << endl;

            //fprintf(stderr, "format=%s\n", numFormat);
            snprintf(medc, sizeof(medc), numFormat, min + (ii + 1) * (max - min) / nn);
            //medFont->name.setValue("Helvetica");

            medFont->name.setValue(fontName);
            //medFont->name.setValue("");
            medFont->size.setValue(15);

            medmatb->value = SoMaterialBinding::OVERALL;
            medmat->diffuseColor.set1Value(0, 1.0, 1.0, 1.0);
            ypos += delty;
            medTrans->translation.setValue(x_0 + 0.12, ypos + 0.01, 0.0);
            medScale->scaleFactor.setValue(0.004, 0.004, 0.004);
            medText->string.setValue(medc);
            //medText->string.setValue("0123456789ABCDEFGabcdefg");

            float vert_x[2];
            float vert_y[2];
            float vert_z[2];
            vert_x[0] = x_0 + 0.08;
            vert_y[0] = ypos;
            vert_z[0] = 0;
            vert_x[1] = x_0 + 0.11;
            vert_y[1] = ypos;
            vert_z[1] = 0;
            int vertices[2];
            vertices[0] = 0;
            vertices[1] = 1;

            int indices[2];
            indices[0] = 0;
            indices[1] = 1;
            InvLine *tick = new InvLine;
            tick->setCoords(1, 2,
                            2, vert_x, vert_y, vert_z,
                            vertices, indices);
            float zeroo = 0;
            tick->setColors(1, &zeroo, &zeroo, &zeroo);
            lineSep = tick->getSeparator();

            root->addChild(medSep);
            medSep->addChild(medmatb);
            medSep->addChild(medmat);
            medSep->addChild(medFont);
            medSep->addChild(medTrans);
            medSep->addChild(medScale);
            medSep->addChild(medText);
            root->addChild(lineSep);
        }
    }

    else if (ramps > 1) // colormap with distinct ramps
    {

        for (int i = 0; i < ramps - 1; i++)
        {

            float val_delta = (max - min) / ramps;
            float trans_delta = size / ramps;

            medSep = new SoSeparator;
            medmatb = new SoMaterialBinding;
            medmat = new SoMaterial;
            medFont = new SoFont;
            medTrans = new SoTranslation;
            medText = new SoText2;

            snprintf(medc, sizeof(medc), numFormat, min + (i + 1) * val_delta);
            medFont->size.setValue(fontSize);
            medFont->name.setValue(fontName);
            medmatb->value = SoMaterialBinding::OVERALL;
            medmat->diffuseColor.set1Value(0, 1.0, 1.0, 1.0);
            medTrans->translation.setValue(x_0 + 0.12, y_0 + (i + 1) * trans_delta, 0.0);
            medText->string.setValue(medc);

            medSep->addChild(medmatb);
            medSep->addChild(medmat);
            medSep->addChild(medFont);
            medSep->addChild(medTrans);
            medSep->addChild(medText);
            root->addChild(medSep);
        }
        length = ramps;
    }

    // SoSeparator *maxSep         = new SoSeparator;
    //   root->addChild(maxSep);
    //   maxSep->addChild(maxmatb);
    //   maxSep->addChild(maxmat);
    //   maxSep->addChild(maxFont);
    //   maxSep->addChild(maxTrans);
    //   maxSep->addChild(maxText);

    annSep->addChild(annmatb);
    annSep->addChild(annmat);
    annSep->addChild(annFont);
    annSep->addChild(annTrans);
    annSep->addChild(annText);
    root->addChild(annSep);

    // create quadmesh with colors
    //
    //
    //
    //           _________   M A X TEXT
    //      yi+1 |   i   |
    //           |       |
    //       yi  ---------                         MED_TEXT
    //                                                .
    //              ...                               .
    //       y2  _________   M E D TEXT    or      MED_TEXT
    //           |   1   |
    //           |       |
    //       y1  ---------                         MED_TEXT
    //           |   0   |
    //           |       |
    //       y0  ---------   M I N TEXT
    //          x0       x1   (zi is always 0)
    //
    //
    //           A N N O T A T I O N
    //
    //

    float *vertices = new float[3 * 2 * (length + 1)];
    float *vPtr = vertices;
    float dy = size / length;
    float x_1 = x_0 + 0.1;

    for (int i = 0; i <= length; i++)
    {
        float y_i = y_0 + i * dy;

        // left point
        *vPtr = x_0;
        vPtr++;
        *vPtr = y_i;
        vPtr++;
        *vPtr = 0.0;
        vPtr++;

        // right point
        *vPtr = x_1;
        vPtr++;
        *vPtr = y_i;
        vPtr++;
        *vPtr = 0.0;
        vPtr++;
    }
    coord->point.setValues(0, 2 * (length + 1), (float(*)[3])vertices);
    quadmesh->verticesPerRow = 2;
    quadmesh->verticesPerColumn = length + 1;
    delete[] vertices;

    //
    // assign color to each of the quads
    //
    for (int i = 0; i < length; i++)
    {
        material->diffuseColor.set1Value(i, r[i], g[i], b[i]);
        material->transparency.set1Value(i, 1.0 - a[i]);
    }

    // put into list of colormaps

    colormap_switch->addChild(root);

    InvObject *new_map = new InvObject(name, root);
    colormap_list->add(new_map);
    numMaps++;
}

//=========================================================================
//
//=========================================================================
int InvColormapManager::removeColormap(const char *name)
{
    InvObject *the_map;

    // cerr << "Name of colormap to remove is " << name << endl;

    // find the map in the list
    colormap_list->resetToFirst();
    the_map = colormap_list->search(name);

    // nuke the whole map starting at'root' seperator
    if (the_map != NULL)
    {

        // prior to removing the map we have to set the value of whichChild to
        // correctly we have to take into account that removing a node from the switch
        // induces may introduce a re-indexing
        int switchIdx = colormap_switch->whichChild.getValue();
        int idxToRemove = colormap_switch->findChild((SoNode *)the_map->getObject());

        if (idxToRemove < switchIdx)
        {
            switchIdx--;
            colormap_switch->whichChild.setValue(switchIdx);
            const char *newSwNm = colormap_switch->getChild(switchIdx)->getName().getString();
            strcpy(current_map, newSwNm);
        }
        else if (idxToRemove == switchIdx)
        {
            colormap_switch->whichChild.setValue(SO_SWITCH_NONE);
            strcpy(current_map, "NONE");
        }

        colormap_switch->removeChild((SoNode *)(the_map->getObject()));
        // remove information from the list
        colormap_list->remove(the_map);
        numMaps--;
        return 1;
    }
    else
        return 0; // nothing to delete found

    //cerr << "ypos " << ypos << endl;
}

//=========================================================================
//
//=========================================================================
void InvColormapManager::hideAllColormaps()
{
    colormap_switch->whichChild.setValue(SO_SWITCH_NONE);
    strcpy(current_map, "NONE");
}

//=========================================================================
//
//=========================================================================
void InvColormapManager::showColormap(const char *name, SoXtExaminerViewer *viewer)
{
    int num_map;
    InvObject *the_map;
    (void)viewer;

    // cerr << "Name of colormap to show on screen is " << name << endl;

    if (strcmp(name, "NONE") != 0)
    {

        //                            1,1
        //     |-----------------------|
        //     |                       |
        //     | ||                    |
        //     | ||                    |
        //     | ||                    |
        //     | ||                    |
        //     |-----------------------|
        //    0,0
        //

        // find the map in the list
        colormap_list->resetToFirst();
        the_map = colormap_list->searchX(name);

        if (the_map != NULL)
        {
            num_map = colormap_switch->findChild((SoNode *)(the_map->getObject()));
            colormap_switch->whichChild.setValue(num_map);
            //     cerr << "      --> InvColormapManager::showColormap(..) Switched to colormap no. " << num_map << "  for display " << the_map->getName() << endl;

            strcpy(current_map, the_map->getName());
            /// colormap_camera->viewAll(colormap_switch,viewer->getViewportRegion());
            ///     colormap_camera->position.setValue(0.0,0.0,5.0);
            ///     SbVec3f trans = colormap_transform->translation.getValue();
            ///     SbVec3f camPos = colormap_camera->position.getValue();
            //  colormap_transform->translation.setValue(colormap_camera->aspectRatio.getValue()*(camPos[0]-1.0),camPos[1]-0.9,camPos[2]);
            // colormap_transform->translation.setValue(camPos[0]-0.9,camPos[1]-0.9,camPos[2]);
            ///     colormap_transform->scaleFactor.setValue(1.0,0.5,1.0);

            ///     cerr << "Colormap Camera Position     = " << camPos[0] << " " << camPos[1] << " " << camPos[2]<< endl;
            ///     cerr << "Colormap Camera Aspect Ratio = " << colormap_camera->aspectRatio.getValue() << endl;
            ///     cerr << "Distance To Near Clip Plane  = " << colormap_camera->nearDistance.getValue() << endl;
            ///     cerr << "Distance To Far Clip Plane   = " << colormap_camera->farDistance.getValue() << endl;
            ///     cerr << "Colormap Transform Position  = " << trans[0] << " " << trans[1] << " " << trans[2] << endl;
        }
    }
}

//=========================================================================
//
//=========================================================================
char *InvColormapManager::currentColormap()
{

    // cerr << "Name of current shown colormap is " << current_map << endl;
    return current_map;
}

//=========================================================================
//
//=========================================================================
InvColormapManager::~InvColormapManager()
{
    // stuff to delete...

    // finally delete the list
    delete colormap_list;
}
