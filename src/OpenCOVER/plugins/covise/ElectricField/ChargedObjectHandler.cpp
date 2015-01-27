/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <osgText/Text>

#include <cover/coVRFileManager.h>

#include "cover/VRViewer.h"

#include "cover/coTranslator.h"

#include "ChargedObjectHandler.h"
#include "ElectricFieldPlugin.h"

ChargedObjectHandler *ChargedObjectHandler::instance_ = NULL;

ChargedObjectHandler::ChargedObjectHandler()
{
    // create grid
    grid_min = -1.5;
    grid_max = 1.5;
    grid_steps = 51; // odd number, so we have a grid position at 0.0 (seems to be nescessary if we have just one plate in the origin)

    field_u = new float[grid_steps * grid_steps * grid_steps];
    field_v = new float[grid_steps * grid_steps * grid_steps];
    field_w = new float[grid_steps * grid_steps * grid_steps];
    field_potential = new float[grid_steps * grid_steps * grid_steps];

    // create points
    for (int i = 0; i < 4; ++i)
    {
        std::stringstream out;
        out << "Punkt " << (i + 1);
        chargedObjects.push_back(new ChargedPoint(out.str(), 100.0));
    }

    // create plate
    chargedPlates[0] = new ChargedPlate("Platte A", 100.0);
    chargedPlates[1] = new ChargedPlate("Platte B", -100.0);
    chargedPlates[0]->setOtherPlate(chargedPlates[1]);
    chargedPlates[1]->setOtherPlate(chargedPlates[0]);
    chargedObjects.push_back(chargedPlates[0]);
    chargedObjects.push_back(chargedPlates[1]);

    // GUI-Message

    osg::Geode *geode = new osg::Geode();
    osg::StateSet *stateset = geode->getOrCreateStateSet();
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    textCamera = new osg::Camera;
    textCamera->setProjectionMatrix(osg::Matrix::ortho2D(0.0, 800.0, 0.0, 800.0)); // virtual screen is 0-800/0-800
    textCamera->setComputeNearFarMode(osg::CullSettings::DO_NOT_COMPUTE_NEAR_FAR);
    textCamera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
    textCamera->setViewMatrix(osg::Matrix::translate(osg::Vec3(0, 0, 100)));
    textCamera->setViewMatrix(osg::Matrix::identity());
    textCamera->setClearMask(GL_DEPTH_BUFFER_BIT);
    textCamera->setRenderOrder(osg::Camera::POST_RENDER);
    textCamera->addChild(geode);

    // text
    osgText::Text *text = new osgText::Text;
    text->setFont(coVRFileManager::instance()->getFontFile(NULL));
    text->setPosition(osg::Vec3(400.0, 400.0, 0.2));
    text->setText(coTranslator::coTranslate("Feld wird berechnet ..."), osgText::String::ENCODING_UTF8);
    text->setColor(osg::Vec4(0.5451, 0.7020, 0.2431, 1.0));
    text->setCharacterSize(20);
    text->setAlignment(osgText::Text::CENTER_BOTTOM_BASE_LINE);
    geode->addDrawable(text);

    // geometry
    osg::Geometry *geom = new osg::Geometry;
    osg::Vec3Array *vertices = new osg::Vec3Array;
    osg::StateSet *stateset2 = geom->getOrCreateStateSet();
    stateset2->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset2->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

    vertices->push_back(osg::Vec3(280.0, 430.0, 0.1));
    vertices->push_back(osg::Vec3(280.0, 390.0, 0.1));
    vertices->push_back(osg::Vec3(520.0, 390.0, 0.1));
    vertices->push_back(osg::Vec3(520.0, 430.0, 0.1));
    geom->setVertexArray(vertices);

    osg::Vec3Array *normals = new osg::Vec3Array;
    normals->push_back(osg::Vec3(0.0f, 0.0f, 1.0f));
    geom->setNormalArray(normals);
    geom->setNormalBinding(osg::Geometry::BIND_OVERALL);

    osg::Vec4Array *colors = new osg::Vec4Array;
    colors->push_back(osg::Vec4(0.0f, 0.0, 0.0f, 0.8f));
    geom->setColorArray(colors);
    geom->setColorBinding(osg::Geometry::BIND_OVERALL);

    geom->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
    geode->addDrawable(geom);
}

ChargedObjectHandler *ChargedObjectHandler::Instance()
{
    if (instance_ == NULL)
        instance_ = new ChargedObjectHandler();
    return instance_;
}

int ChargedObjectHandler::getActiveObjectsCount(unsigned int types)
{
    int num(0);
    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
    {
        if ((((*it)->getTypeId() & types) > 0) && (*it)->isActive())
            ++num;
    }
    return num;
}

// note: dont call this function too often (e.g. for every grid point) because it's too expensive for that
bool ChargedObjectHandler::fieldIsValid()
{
    // check if we have any objects at all
    if (getActiveObjectsCount(TYPE_POINT | TYPE_PLATE) == 0)
        return false;
    // check some "random" points
    if (getFieldAt(osg::Vec3(grid_min / 2.0, grid_min / 2.0, grid_min / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_max / 2.0, grid_min / 2.0, grid_min / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_min / 2.0, grid_max / 2.0, grid_min / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_max / 2.0, grid_max / 2.0, grid_min / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_min / 2.0, grid_min / 2.0, grid_max / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_max / 2.0, grid_min / 2.0, grid_max / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_min / 2.0, grid_max / 2.0, grid_max / 2.0)).length() > 0.0001)
        return true;
    if (getFieldAt(osg::Vec3(grid_max / 2.0, grid_max / 2.0, grid_max / 2.0)).length() > 0.0001)
        return true;
    return false;
}

void ChargedObjectHandler::objectsActiveStateChanged()
{
    // update label visibility
    bool visible = (getActiveObjectsCount(TYPE_POINT) > 1);
    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
    {
        if (((*it)->getTypeId() == TYPE_POINT))
            (*it)->setLabelVisibility(visible && (*it)->isActive());
    }
    // show or hide box
    ElectricFieldPlugin::plugin->setBoundingBoxVisible(getActiveObjectsCount(TYPE_POINT | TYPE_PLATE) > 0);
}

void ChargedObjectHandler::preFrame()
{

    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
        (*it)->preFrame();

    if (fieldIsDirty > 0)
    {
        --fieldIsDirty;
        if (fieldIsDirty == 0)
            calculateField();
    }
}

void ChargedObjectHandler::guiToRenderMsg(const char *msg)
{
    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
        (*it)->guiToRenderMsg(msg);
}

ChargedPoint *ChargedObjectHandler::addPoint()
{
    if (!(chargedPlates[0]->isActive() && chargedPlates[1]->isActive())) // dont allow any points if we have two plates
    {
        for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++) // search an inactive point
        {
            if (((*it)->getTypeId() == TYPE_POINT) && !((*it)->isActive()))
            {
                (*it)->setActive(true);
                dirtyField();
                return (ChargedPoint *)(*it);
            }
        }
    }
    return NULL;
}

ChargedPlate *ChargedObjectHandler::addPlate()
{
    int numberOfPoints = 0;
    int numberOfPlates = 0;
    ChargedPlate *firstInactivePlate = NULL;
    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
    {
        if (((*it)->getTypeId() == TYPE_POINT) && (*it)->isActive())
        {
            ++numberOfPoints;
        }
        if ((*it)->getTypeId() == TYPE_PLATE)
        {
            if ((*it)->isActive())
                ++numberOfPlates;
            else if (firstInactivePlate == NULL)
                firstInactivePlate = (ChargedPlate *)(*it);
        }
    }
    if ((firstInactivePlate != NULL) && ((numberOfPlates == 0) || (numberOfPoints == 0))) // dont allow more than one plate if we have points
    {
        firstInactivePlate->setActive(true);
        dirtyField();
        return firstInactivePlate;
    }
    return NULL;
}

void ChargedObjectHandler::removeAllObjects()
{
    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
        (*it)->setActive(false);
    dirtyField();
}

void ChargedObjectHandler::dirtyField()
{
    fieldIsDirty = 5;
    // don't calculate field immediately (there may be several calls to dirtyField during one user-action)
}

void ChargedObjectHandler::calculateField()
{

    // show text
    cover->getScene()->addChild(textCamera.get());
    VRViewer::instance()->redrawHUD(0.0);

    // speed up get*At by preparing a list of active objects here
    twoPlatesActive = chargedPlates[0]->isActive() && chargedPlates[1]->isActive();
    activeObjects.clear();
    for (vector<ChargedObject *>::iterator it = chargedObjects.begin(); it < chargedObjects.end(); it++)
    {
        if ((*it)->isActive())
            activeObjects.push_back(*it);
    }

    // create vector field and potential
    osg::Vec4 f;
    int pos;
    float d = (grid_max - grid_min) / (float)(grid_steps - 1);
    for (int x = 0; x < grid_steps; x++)
    {
        for (int y = 0; y < grid_steps; y++)
        {
            for (int z = 0; z < grid_steps; z++)
            {
                f = getFieldAndPotentialAt(osg::Vec3(grid_min + x * d, grid_min + y * d, grid_min + z * d));
                pos = x * grid_steps * grid_steps + y * grid_steps + z;
                field_u[pos] = f[0];
                field_v[pos] = f[1];
                field_w[pos] = f[2];
                field_potential[pos] = f[3];
            }
        }
    }

    // hide text
    cover->getScene()->removeChild(textCamera.get());

    ElectricFieldPlugin::plugin->fieldChanged();
}

void ChargedObjectHandler::setRadiusOfPlates(float radius)
{
    chargedPlates[0]->setRadius(radius, true);
    chargedPlates[1]->setRadius(radius, true);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

osg::Vec3 ChargedObjectHandler::getFieldAt(osg::Vec3 point)
{
    osg::Vec3 field = osg::Vec3(0.0, 0.0, 0.0);
    for (vector<ChargedObject *>::iterator it = activeObjects.begin(); it < activeObjects.end(); it++)
    {
        field += (*it)->getFieldAt(point);
    }
    if (twoPlatesActive)
    {
        chargedPlates[0]->correctField(point, field);
    }
    return field;
}

float ChargedObjectHandler::getPotentialAt(osg::Vec3 point)
{
    float p = 0.0;
    for (vector<ChargedObject *>::iterator it = activeObjects.begin(); it < activeObjects.end(); it++)
    {
        p += (*it)->getPotentialAt(point);
    }
    if (twoPlatesActive)
    {
        chargedPlates[0]->correctPotential(point, p);
    }
    return p;
}

osg::Vec4 ChargedObjectHandler::getFieldAndPotentialAt(osg::Vec3 point)
{
    osg::Vec4 fieldAndPotential = osg::Vec4(0.0, 0.0, 0.0, 0.0);
    for (vector<ChargedObject *>::iterator it = activeObjects.begin(); it < activeObjects.end(); it++)
    {
        fieldAndPotential += (*it)->getFieldAndPotentialAt(point);
    }
    if (twoPlatesActive)
    {
        chargedPlates[0]->correctFieldAndPotential(point, fieldAndPotential);
    }
    return fieldAndPotential;
}
