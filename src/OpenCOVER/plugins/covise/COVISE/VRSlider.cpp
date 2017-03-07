/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

// by Lars Frenzel
// 28.10.1997

#include <util/common.h>
#include "VRSlider.h"
#include <cover/VRPinboard.h>
#include <appl/RenderInterface.h>
#include "VRCoviseGeometryManager.h"
#include "VRCoviseObjectManager.h"
#include <cover/VRSceneGraph.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/RenderObject.h>
#include <cover/input/VRKeys.h>

#include <osg/Geode>
#include <osg/Geometry>
#include <osg/MatrixTransform>
#include <osgSim/SphereSegment>
#include <OpenVRUI/sginterface/vruiRendererInterface.h>

using namespace opencover;
using namespace covise;
using namespace vrui;
/*
 * make spline computation a little bit easier
 */
float Slider::getPoint(float *points, int i, int numPoints)
{
    if (i < 0)
        return points[0];

    if (i < numPoints)
        return points[i];

    return points[numPoints - 1];
}

/*
 * intersect plane (through point, normal vector norm) with vector (p2 - p1)
 */
float Slider::intersect(osg::Vec3 point, osg::Vec3 norm,
                        osg::Vec3 p1, osg::Vec3 p2)
{
    float u = (norm * (point - p1)) / (norm * (p2 - p1));
    return u;
}

Slider::Slider(const char *attrib, const char *sa, osg::Node *n)
    : coTrackerButtonInteraction(coInteraction::ButtonA, "Slider", coInteraction::Medium)
{
    int i;
    dcs = NULL;
    sphereTransform = NULL;
    line = NULL;
    button = NULL;
    node = n;
    xcoords = NULL;
    ycoords = NULL;
    zcoords = NULL;
    length = NULL;
    totalLength = 0;
    numPoints = 0;
    floatSlider = 0;
    numPoints = 0;
    oldValue = 0;
    sattrib = new char[strlen(sa) + 1];
    strcpy(sattrib, sa);
    std::string feedback;
    std::istringstream ifeedback(sa);
    ifeedback >> feedback;
    feedback_information.append(feedback).append("\n");
    ifeedback >> feedback;
    feedback_information.append(feedback).append("\n");
    ifeedback >> feedback;
    feedback_information.append(feedback).append("\n");

    std::istringstream stream(attrib);
    stream >> sliderType;
    stream >> moduleName;
    stream >> instanceName;
    stream >> hostName;
    stream >> dataType;
    stream >> parameterName;
    stream >> min;
    stream >> max;
    stream >> value;
    stream >> geometryType;

    if (dataType == "float")
        floatSlider = 1;
    else if (dataType == "int")
        floatSlider = 0;
    else
        printf("Slider must be floatSlider or intSlider\n");

    if (sliderType == 'M')
    {
        stream >> subMenu;
        updateMenu();
    }
    else
    {
        stream >> radius;
        stream >> numPoints;
        xcoords = new float[numPoints];
        ycoords = new float[numPoints];
        zcoords = new float[numPoints];
        length = new float[numPoints];
        float xd, yd, zd;
        for (i = 0; i < numPoints; i++)
        {
            stream >> xcoords[i];
            stream >> ycoords[i];
            stream >> zcoords[i];

            if (i == 0)
                length[i] = 0;
            else
            {
                xd = xcoords[i] - xcoords[i - 1];
                yd = ycoords[i] - ycoords[i - 1];
                zd = zcoords[i] - zcoords[i - 1];
                length[i] = length[i - 1] + sqrt(xd * xd + yd * yd + zd * zd);
            }
            totalLength = length[i];
        }

        if (sliderType == 'S' || sliderType == 's')
        {
            int sub = 5; // number of subdivision points per line
            int numSplinePoints = (numPoints)*sub;

            float *sxcoords = new float[numSplinePoints];
            float *sycoords = new float[numSplinePoints];
            float *szcoords = new float[numSplinePoints];
            float *slength = new float[numSplinePoints];

            int start, j;
            for (start = -3, j = 0; j < numPoints; j++, start++)
            {
                for (int i = 0; i < sub; ++i)
                {
                    float t = ((float)i) / sub;
                    float it = 1.0f - t;

                    float b0 = it * it * it / 6.0f;
                    float b1 = (3 * t * t * t - 6 * t * t + 4) / 6.0f;
                    float b2 = (-3 * t * t * t + 3 * t * t + 3 * t + 1) / 6.0f;
                    float b3 = t * t * t / 6.0f;

                    float x = b0 * getPoint(xcoords, start, numSplinePoints) + b1 * getPoint(xcoords, start + 1, numSplinePoints) + b2 * getPoint(xcoords, start + 2, numSplinePoints) + b3 * getPoint(xcoords, start + 3, numSplinePoints);

                    float y = b0 * getPoint(ycoords, start, numSplinePoints) + b1 * getPoint(ycoords, start + 1, numSplinePoints) + b2 * getPoint(ycoords, start + 2, numSplinePoints) + b3 * getPoint(ycoords, start + 3, numSplinePoints);

                    float z = b0 * getPoint(zcoords, start, numSplinePoints) + b1 * getPoint(zcoords, start + 1, numSplinePoints) + b2 * getPoint(zcoords, start + 2, numSplinePoints) + b3 * getPoint(zcoords, start + 3, numSplinePoints);
                    sxcoords[j * sub + i] = x;
                    sycoords[j * sub + i] = y;
                    szcoords[j * sub + i] = z;
                }
            }

            slength[0] = 0;

            for (i = 1; i < numSplinePoints; i++)
            {
                xd = sxcoords[i] - sxcoords[i - 1];
                yd = sycoords[i] - sycoords[i - 1];
                zd = szcoords[i] - szcoords[i - 1];
                slength[i] = slength[i - 1] + sqrt(xd * xd + yd * yd + zd * zd);
            }

            delete[] xcoords;
            delete[] ycoords;
            delete[] zcoords;
            delete[] length;

            xcoords = sxcoords;
            ycoords = sycoords;
            zcoords = szcoords;
            length = slength;

            numPoints = numSplinePoints;
            totalLength = length[numPoints - 1];

            if (sliderType == 'S')
                sliderType = 'V';
            if (sliderType == 's')
                sliderType = 'v';
        }

        if (sliderType == 'V' || sliderType == 'v')
        {
            if (sliderType == 'V')
            {
                int ll = 0;
                float r = 1.0, g = 1.0, b = 1.0;

                int *vl = new int[numPoints];
                for (i = 0; i < numPoints; i++)
                    vl[i] = i;

                line = GeometryManager::instance()->addLine(
                    (char *)moduleName.c_str(), 1, numPoints, numPoints,
                    xcoords, ycoords, zcoords, vl, &ll, 1, Bind::OverAll, 0,
                    &r, &g, &b, NULL, 0, Pack::None, NULL, NULL, NULL, 0, NULL,
                    0, 0, 0, NULL, 0, NULL, NULL, osg::Texture::CLAMP_TO_EDGE, osg::Texture::NEAREST, osg::Texture::NEAREST,
                    2.f);
                delete[] vl;
            }
            dcs = new osg::MatrixTransform();
            updatePosition();
            osg::Matrix mat;
            mat.makeScale(radius, radius, radius);
            dcs->setMatrix(mat);
            dcs->addChild(VRSceneGraph::instance()->getHandSphere().get());
            cover->getObjectsRoot()->addChild(dcs.get());

            sphereTransform = new osg::MatrixTransform();
            mat.makeScale(0.0, 0.0, 0.0);
            mat.setTrans(xcoords[0], ycoords[0], zcoords[0]);
            sphereTransform->setMatrix(mat);
            osg::Geode *geode = NULL;
            if (geometryType == "Sphere" || geometryType == "SphereSegment")
            {
                if (geometryType == "Sphere")
                {

                    osg::Sphere *sphere = new osg::Sphere(osg::Vec3(0, 0, 0), 1.0);
                    osg::ShapeDrawable *sphereDrawable = new osg::ShapeDrawable(sphere);
                    geode = new osg::Geode();
                    geode->addDrawable(sphereDrawable);
                }
                else if (geometryType == "SphereSegment")
                {
                    geode = new osgSim::SphereSegment(osg::Vec3(0, 0, 0), 1.0, 0, M_PI * 2, 0.0, 0.0, 32);
                }

                osg::StateSet *stateset = geode->getOrCreateStateSet();
                osg::Material *material = new osg::Material;
                material->setDiffuse(osg::Material::FRONT_AND_BACK, osg::Vec4(1.0, 0.0, 0.0, 1.0));
                material->setAlpha(osg::Material::FRONT_AND_BACK, 0.3);
                stateset->setAttributeAndModes(material, osg::StateAttribute::OVERRIDE);
                stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
                stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
                stateset->setNestRenderBins(false);

                sphereTransform->addChild(geode);
                cover->getObjectsRoot()->addChild(sphereTransform.get());
            }
        }
    }
}

Slider::~Slider()
{
    delete[] xcoords;
    delete[] ycoords;
    delete[] zcoords;
    delete[] length;
    if (line.get())
    {
        osg::Transform *xdcs = (osg::Transform *)line->getParent(0);
        osg::Group *parent = xdcs->getParent(0);
        xdcs->removeChild(line.get());
        parent->removeChild(xdcs);
    }
    if (dcs.get())
    {
        dcs->removeChild(VRSceneGraph::instance()->getHandSphere().get());
        osg::Group *parent = dcs->getParent(0);
        parent->removeChild(dcs.get());
    }
    if (sphereTransform.get())
    {
        sphereTransform->removeChildren(0, sphereTransform->getNumChildren());
        osg::Group *parent = sphereTransform->getParent(0);
        parent->removeChild(sphereTransform.get());
        sphereTransform = 0;
    }

    if (button)
        button->spec.calledClass = NULL;
    delete[] sattrib;
}

char Slider::getSliderType()
{

    return sliderType;
}

float Slider::getLength()
{

    return totalLength;
}

/*
 * minimum distance of slider points to line through (x, y, z) 
 * with direction (dx, dy, dz)
 */
float Slider::getMinDist(osg::Vec3 position, osg::Vec3 direction)
{

    int i;
    float minDist = -1, dist;

    for (i = 0; i < numPoints; i++)
    {
        osg::Vec3 p(xcoords[i], ycoords[i], zcoords[i]);

        float t = (direction * (p - position)) / (direction * direction);
        dist = osg::Vec3(p - (position + direction * t)).length();

        if ((minDist < 0) || (dist < minDist))
            minDist = dist;
    }
    return minDist;
}

float Slider::getMinDist(float x, float y, float z)
{
    int i;
    float minDist = -1, dist;
    float xd, yd, zd;
    for (i = 0; i < numPoints; i++)
    {
        xd = x - xcoords[i];
        yd = y - ycoords[i];
        zd = z - zcoords[i];
        dist = xd * xd + yd * yd + zd * zd;
        if ((minDist < 0) || (dist < minDist))
        {
            minDist = dist;
        }
    }
    return minDist;
}

/*
 * find interection of pointer plane (pointer base, direction, up vector)
 * with slider lines
 */
void Slider::updateValue(osg::Vec3 position,
                         osg::Vec3 direction)
{
    int i;

    for (i = 0; i < numPoints - 1; i++)
    {
        osg::Vec3 p1(xcoords[i], ycoords[i], zcoords[i]);
        osg::Vec3 p2(xcoords[i + 1], ycoords[i + 1], zcoords[i + 1]);

        osg::Vec3 up = (p2 - p1) ^ direction;

        // intersect slider with plane (pointer base, up vector)
        float u = intersect(position, up ^ direction, p1, p2);
        if (u >= 0 && u <= 1)
        {
            osg::Vec3 sect = p1 + (p2 - p1) * u;
            float len_total = osg::Vec3(p2 - p1).length();
            float len_sect = osg::Vec3(sect - p1).length();

            value = min + (max - min) * (length[i] + (length[i + 1] - length[i]) * (len_sect / len_total)) / totalLength;
        }
    }

    updatePosition();
}

void Slider::updateSpec(buttonSpecCell *spec)
{
    min = spec->sliderMin;
    max = spec->sliderMax;

    value = spec->state;
    if (spec->dragState == BUTTON_RELEASED)
    {
        if (value != oldValue)
        {
            updateParameter();
            oldValue = value;
        }
    }
}

bool Slider::updateInteraction()
{
    coTrackerButtonInteraction::update();
    if (isRunning())
        return true;

    return false;
}

void Slider::updatePosition()
{
    int i;
    float x = xcoords[numPoints - 1], y = ycoords[numPoints - 1], z = zcoords[numPoints - 1], l;
    l = (totalLength * (value - min) / (max - min));
    if (l < 0)
        l = 0;
    if (l > totalLength)
        l = totalLength;
    for (i = 1; i < numPoints; i++)
    {
        if (length[i] > l)
        {
            float f = (l - length[i - 1]) / (length[i] - length[i - 1]);
            x = xcoords[i - 1] * (1 - f);
            y = ycoords[i - 1] * (1 - f);
            z = zcoords[i - 1] * (1 - f);
            x += xcoords[i] * f;
            y += ycoords[i] * f;
            z += zcoords[i] * f;
            break;
        }
    }
    osg::Matrix mat = dcs->getMatrix();
    mat.setTrans(x, y, z);
    dcs->setMatrix(mat);

    if (sphereTransform.get())
    {
        mat.makeScale(value, value, value);
        mat.setTrans(xcoords[0], ycoords[0], zcoords[0]);
        sphereTransform->setMatrix(mat);
    }
}

void Slider::updateMenu()
{
    char buf[200];
    char buf2[200];
    char buf3[200];
    VRMenu *menu;
    sprintf(buf, "%s", moduleName.c_str());
    strcpy(buf3, buf);
    if (!(menu = VRPinboard::instance()->namedMenu(buf)))
    {
        buttonSpecCell *spec = new buttonSpecCell();
        spec->actionType = BUTTON_SUBMENU;
        strcpy(spec->subMenuName, buf);
        sprintf(buf2, "%s...", moduleName.c_str());
        strcpy(spec->name, buf2);
        spec->callback = NULL;
        spec->calledClass = (void *)this;
        spec->state = false;
        spec->dashed = false;
        spec->group = cover->createUniqueButtonGroupId();
        VRPinboard::instance()->addButtonToMainMenu(spec);
        ObjectManager::instance()->addCoviseMenu(NULL, spec);
    }

    sprintf(buf, "%s %s", moduleName.c_str(), subMenu.c_str());
    if (!(menu = VRPinboard::instance()->namedMenu(buf)))
    {
        buttonSpecCell *spec = new buttonSpecCell();
        spec->actionType = BUTTON_SUBMENU;
        strcpy(spec->subMenuName, buf);
        sprintf(buf2, "%s...", subMenu.c_str());
        strcpy(spec->name, buf2);
        spec->callback = NULL;
        spec->calledClass = (void *)this;
        spec->state = false;
        spec->dashed = false;
        spec->group = cover->createUniqueButtonGroupId();
        spec->setMenu(buf3);
        VRPinboard::instance()->addButtonToNamedMenu(spec, buf3);
    }

    if (!(menu = VRPinboard::instance()->namedMenu(buf)))
    {
        fprintf(stderr, "Pinboard Error, could not create Menu %s \n", buf);
        return;
    }
    button = menu->namedButton(parameterName.c_str());
    if (button)
    {
        button->spec.sliderMin = min;
        button->spec.sliderMax = max;
        button->spec.state = value;
        button->spec.oldState = value;
        button->setState(value);
        button->spec.calledClass = (void *)this;
    }
    else
    {
        buttonSpecCell *spec = new buttonSpecCell();
        spec->setMenu(buf);
        strcpy(spec->name, parameterName.c_str());
        spec->actionType = BUTTON_SLIDER;
        spec->callback = &Slider::menuCallback;
        spec->calledClass = (void *)this;
        spec->state = value;
        spec->oldState = value;
        spec->dashed = false;
        spec->group = -1;
        spec->sliderMin = min;
        spec->sliderMax = max;
        menu->addButton(spec);
    }
}

int Slider::isSlider(const char *n)
{
    return (!(strcmp(sattrib, n)));
}

void Slider::updateParameter()
{
    char buf[600];

    CoviseRender::set_feedback_info(feedback_information.c_str());

    if (coVRMSController::instance()->isMaster())
    {
        if (floatSlider)
        {
            sprintf(buf, "%s\nFloatSlider\n%.3f %.3f %.3f\n", parameterName.c_str(), min, max, value);
        }
        else
        {
            sprintf(buf, "%s\nIntSlider\n%d %d %d\n", parameterName.c_str(), (int)min, (int)max, (int)value);
        }
        CoviseRender::send_feedback_message("PARAM", buf);
        buf[0] = '\0';
        CoviseRender::send_feedback_message("EXEC", buf);
    }
}

void Slider::startInteraction()
{
}

void Slider::stopInteraction()
{
    updateParameter();
}

void Slider::doInteraction()
{
    // retrieve pointer coordinates
    osg::Matrix mat = cover->getPointerMat();
    osg::Vec3 position = mat.getTrans();
    osg::Vec3 direction(mat(1, 0), mat(1, 1), mat(1, 2));
    mat = cover->getInvBaseMat();
    direction = mat.transform3x3(direction, mat);

    position = mat.preMult(position);
    direction.normalize();

    updateValue(position, direction);
}

void Slider::menuCallback(void *slider, buttonSpecCell *spec)
{
    if (slider)
        ((Slider *)slider)->updateSpec(spec);
}

Slider *SliderList::find(osg::Vec3 position, osg::Vec3 direction, float *distance)
{
    Slider *nearest;
    float near_dist, cur_dist;

    // find the nearest of the Slider-Spheres
    reset();

    near_dist = -1;
    nearest = NULL;

    while (current())
    {
        if (current()->getSliderType() != 'M')
        { // this is not a Menu entry
            // check distance
            cur_dist = current()->getMinDist(position, direction);

            if (cur_dist < near_dist || near_dist == -1)
            {
                near_dist = cur_dist;
                nearest = current();
            }
        }

        next();
    }
    if (distance)
        *distance = near_dist;

    return nearest;
}

SliderList::SliderList()
{
}

SliderList::~SliderList()
{
}

void SliderList::add(RenderObject *dobj, osg::Node *n)
{
    int i = 0;
    char buf[100];
    sprintf(buf, "SLIDER%d", i);
    while (const char *attrib = dobj->getAttribute(buf))
    {
        char *sattrib = new char[strlen(attrib) + 1];
        strcpy(sattrib, attrib);
        char *tmp = strchr(sattrib, '\n');

        for (i = 0; i < 4; i++)
        {
            if (tmp)
                tmp = strchr(tmp + 1, '\n');
        }
        if (tmp)
            *tmp = '\0';

        Slider *sl = find(sattrib);
        if (sl)
            sl->node = n;
        else
        {
            sl = new Slider(attrib, sattrib, n);
            append(sl);
        }
        sprintf(buf, "SLIDER%d", ++i);
        delete[] sattrib;
    }
}

Slider *SliderList::find(const char *attrib)
{
    reset();
    while (current())
    {
        if (current()->isSlider(attrib))
            return (current());
        next();
    }

    return (NULL);
}

Slider *SliderList::find(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
            return (current());

        next();
    }

    return (NULL);
}

void SliderList::removeAll(osg::Node *n)
{
    reset();
    while (current())
    {
        if (current()->node == n)
            remove();
        else
            next();
    }
}

void SliderList::update()
{

    bool active = false;

    for (int i = 0; i < num(); i++)
    {
        Slider *s = item(i);
        if (s->isRunning())
            active = true;

        if (s->isRegistered() && !s->isRunning())
            coInteractionManager::the()->unregisterInteraction(s);
    }

    if (!active)
    {

        osg::Matrix mat = cover->getPointerMat();
        osg::Vec3 position = mat.getTrans();
        osg::Vec3 direction(mat(1, 0), mat(1, 1), mat(1, 2));
        mat = cover->getInvBaseMat();
        direction = mat.transform3x3(direction, mat);

        position = mat.preMult(position);
        direction.normalize();

        float distance = -1;
        Slider *s = find(position, direction, &distance);
        if (s)
        {
            if (distance != -1 && distance < s->getLength() / 10)
                coInteractionManager::the()->registerInteraction(s);
        }
    }
}

SliderList *SliderList::instance()
{

    static SliderList _instance;
    return &_instance;
}
