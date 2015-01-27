/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRSelectionManager.h>
#include <virvo/vvtoolshed.h>
#include "Annotation.h"
#include <cover/coVRMSController.h>
#include <cover/coOnscreenDebug.h>
#include <sstream>

Annotation::Annotation(int id, int owner, osg::Node * /*node*/, float initscale,
                       osg::Matrix orientation)
{
    //set id
    this->id = id;
    this->owner = owner;

    //create arrow
    arrowSize = 100.0;

    arrow = new coArrow(arrowSize / 3, arrowSize, true);

    //create the label
    osg::Vec4 fgcolor(0, 1, 0, 1);
    osg::Vec4 bgcolor(0.5, 0.5, 0.5, 0.5);
    float lineLen = 0.04 * cover->getSceneSize();
    float fontSize = 0.02 * cover->getSceneSize();
    label = new coVRLabel(" ", fontSize, lineLen, fgcolor, bgcolor);
    //setText("Annotation");

    setColor(0.47); // set all annotations to the same color by default (green)

    // create scale matrix
    scale = new osg::MatrixTransform;
    /*osg::Matrix matrix;
    matrix.makeScale(initscale, initscale, initscale);
    scale->setMatrix(matrix);*/
    _scaleVal = initscale;

    //create the rotation Matrix
    rot = new osg::MatrixTransform;
    rot->setMatrix(orientation);

    //create transformation matrix
    pos = new osg::MatrixTransform;

    mainGroup = new osg::Group();

    // convert int to string for the name of the annotations node in the scenegraph
    string nodeName = "Annotation";
    std::ostringstream stream;
    if (stream << id)
    {
        nodeName.append(stream.str());
        setText(stream.str().c_str());
    }
    else
    {
        setText("Annotation");
    }
    mainGroup->setName(nodeName);
    osg::StateSet *ss = mainGroup->getOrCreateStateSet();
    for (int i = 0; i < cover->getNumClipPlanes(); i++)
    {
        ss->setAttributeAndModes(cover->getClipPlane(1), osg::StateAttribute::OFF);
    }

    coVRSelectionManager::markAsHelperNode(arrow);
    coVRSelectionManager::markAsHelperNode(scale);
    coVRSelectionManager::markAsHelperNode(rot);
    coVRSelectionManager::markAsHelperNode(pos);

    osg::Group *root = cover->getObjectsRoot()->asGroup();

    coVRSelectionManager::markAsHelperNode(mainGroup);
    root->addChild(mainGroup);

    //ceate scenegraph
    scale->addChild(arrow);
    rot->addChild(scale);
    pos->addChild(rot);
    mainGroup->addChild(pos);

    mySensor = new AnnotationSensor(this, arrow);
}

void Annotation::setVisible(bool vis)
{
    if (vis)
    {
        arrow->setNodeMask(~0);
        label->show();
    }
    else
    {
        arrow->setNodeMask(0);
        label->hide();
    }
}

int Annotation::getID() const
{
    return id;
}

bool Annotation::sameID(const int id) const
{
    return this->id == id;
}

bool Annotation::changesAllowed(const int id) const
{
    return this->id == id || this->id == -1;
}

void Annotation::setColor(float hue)
{
    float r, g, b;
    _hue = hue;
    vvToolshed::HSBtoRGB(_hue, 1, 1, &r, &g, &b);
    osg::Vec4 color(r, g, b, 1.0);
    arrow->setColor(color);
    label->setFGColor(color);
}

float Annotation::getColor() const
{
    return _hue;
}

void Annotation::setAmbient(osg::Vec4 spec)
{
    arrow->setAmbient(spec);
}

/// sets color to local-locked state
void Annotation::setAmbientLocalLocked()
{
    static const osg::Vec4 LOCAL_LOCK_COLOR(1.0f, 1.0f, 1.0f, 1.0f);
    arrow->setAmbient(LOCAL_LOCK_COLOR);
}

/// sets color to remote-locked state
void Annotation::setAmbientRemoteLocked()
{
    static const osg::Vec4 REMOTE_LOCK_COLOR(0.2f, 0.2f, 0.2f, 1.0f);
    arrow->setAmbient(REMOTE_LOCK_COLOR);
}

/// sets color to unlocked state
void Annotation::setAmbientUnlocked()
{
    static const osg::Vec4 UNLOCK_COLOR(0.5f, 0.5f, 0.5f, 1.0f);
    arrow->setAmbient(UNLOCK_COLOR);
}

/////// NEED TO COMPLETE ///////
Annotation::~Annotation()
{

    //pos->getParent(0)->removeChild(pos);
    //mainGroup->getParent(0)->removeChild(mainGroup);
    label->~coVRLabel();

    delete mySensor;

    //remove scenegraph
    if (scale)
        scale->removeChild(arrow);
    if (rot)
        rot->removeChild(scale);
    if (pos)
        pos->removeChild(rot);
    if (mainGroup)
        mainGroup->removeChild(pos);

    if (mainGroup)
    {

        if (mainGroup->getNumParents() > 0)
        {
            osg::Group *parent = mainGroup->getParent(0);

            for (unsigned int i = 0; i < mainGroup->getNumChildren(); i++)
            {
                parent->addChild(mainGroup->getChild(i));
            }
            parent->removeChild(mainGroup);
        }
    }
}

void Annotation::setPos(const osg::Matrix &mat)
{
    //set the position of the arrow in local coordinates
    pos->setMatrix(mat);
}

void Annotation::updateLabelPosition()
{
    //const osg::Vec3 offset = scale->getMatrix().preMult( osg::Vec3(0.0, 0.0, - arrowSize * _scaleVal) );

    //set position of the label in global coordinates
    osg::Matrix matrix;
    matrix.makeTranslate(osg::Vec3f(0.0, 0.0, -(1.35 * arrowSize)));
    //matrix.makeTranslate( offset );

    osg::MatrixList matrixList = arrow->getWorldMatrices();
    osg::MatrixList::iterator iter;
    for (iter = matrixList.begin(); iter != matrixList.end(); iter++)
    {
        matrix.postMult(*iter);
    }

    label->setPosition(matrix.getTrans());
}

void Annotation::scaleArrowToConstantSize()
{
    osg::Matrix arrowPosMat = pos->getMatrix();
    arrowPosMat *= cover->getBaseMat();
    osg::Vec3 arrowPos = arrowPosMat.getTrans();

    //const osg::Vec3 eyePos = cover->getViewerMat().getTrans();
    //const osg::Vec3 eyeToPos = arrowPos - eyePos;

    const float interactorScale = _scaleVal
                                  * cover->getInteractorScale(arrowPos);

    osg::Matrix mat;
    mat.makeScale(interactorScale, interactorScale, interactorScale);
    scale->setMatrix(mat);
} //scaleArrowToConstantSize()

void Annotation::setScale(float scaleset)
{
    _scaleVal = scaleset;
}

float Annotation::getScale() const
{
    return _scaleVal;
}

osg::Matrix Annotation::getPos() const
{
    return pos->getMatrix();
}

void Annotation::getMat(osg::Matrix &m) const
{
    m = pos->getMatrix();
}

void Annotation::getMat(osg::Matrix::value_type m[16]) const
{
    memcpy(m, pos->getMatrix().ptr(), 16 * sizeof(osg::Matrix::value_type));
}

void Annotation::setText(const char *text)
{
    label->setString(text);
}

const osgText::String Annotation::getText() const
{
    return osgText::String("not yet implemented");
}

int Annotation::getOwnerID() const
{
    return owner;
}

void Annotation::setOwnerID(const int id)
{
    owner = id;
}

bool Annotation::sameOwnerID(const int id) const
{
    return owner == id;
}
