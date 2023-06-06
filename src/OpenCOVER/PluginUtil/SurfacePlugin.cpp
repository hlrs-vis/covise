/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */


#include "SurfacePlugin.h"
#include <config/CoviseConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <osg/Node>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/Matrix>
#include <osg/Vec3>
#include <osg/Switch>

#include <PluginUtil/PluginMessageTypes.h>

#include "SurfaceInteraction.h"

using namespace osg;

SurfacePlugin::SurfacePlugin(const char *iconname, string pluginName)
: coVRPlugin(COVER_PLUGIN_NAME)
, ModuleFeedbackPlugin()
, _show(false)
, _scale(NULL)
, _fixed_scale(NULL)
, _fixed_matrix(NULL)
, _matrix(NULL)
, _pointer(NULL)
, _inFocus(false)
, _iconName(iconname)
{
    string sectionName("COVER.Plugin.");
    sectionName += pluginName;
    string angleString(sectionName);
    angleString += ".Angle";
    float angle = coCoviseConfig::getFloat(angleString.c_str(), 0.0f);
    string displacementString(sectionName);
    displacementString += ".Displacement";
    _displacement = coCoviseConfig::getFloat(displacementString.c_str(), 0.0f);
    string scaleString(sectionName);
    scaleString += ".ScaleFactor";
    _scaleFactor = coCoviseConfig::getFloat(scaleString.c_str(), 70.0f);
    _angle = angle;
    angle *= M_PI / 180.0;
    _cos_angle = cos(angle);
    _sin_angle = sin(angle);
    _rootNode = cover->getObjectsRoot();
}

void
SurfacePlugin::GetPointerParams(float &angle, float &scaleFactor, float &displacement)
{
    angle = _angle;
    scaleFactor = _scaleFactor;
    displacement = _displacement;
}

void
SurfacePlugin::SubstitutePointer(const char *iconname)
{
    _iconName = iconname;

    if (!_scale)
    {
        return;
    }

    if (_pointer)
    {
        _scale->removeChild(_pointer);
    }

    _pointer = readPointer(iconname);
    if (_pointer == NULL)
    {
        cerr << "SurfacePlugin::SubstitutePointer: Could not create pointer" << endl;
        return;
    }
    else
    {
        _pointer->ref();
    }
    _scale->addChild(_pointer);
    if (_show && !cover->getPointer()->containsNode(_scale))
    {
        cover->getPointer()->addChild(_scale);
    }
}

void
SurfacePlugin::VerwaltePointer(bool show)
{
    if (show)
    {
        if (_pointer == NULL)
        {
            _pointer = readPointer(_iconName.c_str());
            if (_pointer == NULL)
            {
                cerr << "SurfacePlugin::VerwaltePointer: Could not create pointer" << endl;
                return;
            }
            else
            {
                _pointer->ref();
            }
        }
        if (_scale == NULL)
        {
            osg::Matrix ScaleMat;
            ScaleMat.makeScale(_scaleFactor, _scaleFactor, _scaleFactor);

            osg::Matrix tmat;
            tmat.makeTranslate(0.0, _displacement, 0.0);

            osg::Matrix TransMat = tmat * ScaleMat;

            osg::Matrix rmat;
            rmat.makeRotate(_angle, Vec3(1.0, 0.0, 0.0));

            osg::Matrix RotMat = rmat * TransMat;

            _matrix = new Matrix;
            *_matrix = RotMat;

            _scale = new MatrixTransform;
            _scale->ref();
            _scale->setMatrix(*_matrix);

            // attach _pointer to _scale
            _scale->addChild(_pointer);
        }
        if (!_show)
        {
            if (!cover->getPointer()->containsNode(_scale))
            {
                cover->getPointer()->addChild(_scale);
            }
            // XXX: why? adding alone should be sufficient
            //_scale->setNodeMask(_scale->getNodeMask() | Isect::Intersection);
            _show = true;
        }
    }
    else if (_scale != NULL)
    {
        if (_show)
        {
            if (cover->getPointer()->containsNode(_scale))
            {
                cover->getPointer()->removeChild(_scale);
            }
            // XXX: why? removal alone should be sufficient
            //_scale->setNodeMask(_scale->getNodeMask() & (~Isect::Intersection));
            _show = false;
        }
    }
}

void
SurfacePlugin::GetPoint(osg::Vec3 &vect) const
{
    osg::Matrix pointerMatrix, invBaseMatrix;
    // get the pointer's matrix (position and direction)
    pointerMatrix = cover->getPointerMat();
    // get inverse base matrix from scenegraph
    invBaseMatrix = cover->getInvBaseMat();
    // transform pointer matrix into model space
    pointerMatrix.mult(pointerMatrix, invBaseMatrix);
    osg::Vec3 vect1(pointerMatrix(1, 0), pointerMatrix(1, 1), pointerMatrix(1, 2));
    osg::Vec3 vect2(pointerMatrix(2, 0), pointerMatrix(2, 1), pointerMatrix(2, 2));
    vect = Vec3(pointerMatrix(3, 0), pointerMatrix(3, 1), pointerMatrix(3, 2));
    //vect += _scaleFactor*(_cos_angle*vect1+_sin_angle*vect2);
    vect += (vect1 * _cos_angle + vect2 * _sin_angle) * _scaleFactor;
    //vect += _scaleFactor*_displacement*vect1;
    vect += vect1 * _scaleFactor * _displacement;
    //vect += _displacement*_scaleFactor*(_cos_angle*vect1+_sin_angle*vect2);
}

void
SurfacePlugin::GetNormal(osg::Vec3 &vect) const
{
    osg::Matrix pointerMatrix, invBaseMatrix;
    // get the pointer's matrix (position and direction)
    pointerMatrix = cover->getPointerMat();
    // get inverse base matrix from scenegraph
    invBaseMatrix = cover->getInvBaseMat();
    // transform hand coordinates into model coordinates
    pointerMatrix.mult(pointerMatrix, invBaseMatrix);
    osg::Vec3 vect1(pointerMatrix(1, 0), pointerMatrix(1, 1), pointerMatrix(1, 2));
    osg::Vec3 vect2(pointerMatrix(2, 0), pointerMatrix(2, 1), pointerMatrix(2, 2));
    vect = (vect1 * _cos_angle + vect2 * _sin_angle) * _scaleFactor;
}

SurfacePlugin::~SurfacePlugin()
{
    if (_fixed_scale)
    {
        if (_scale)
        {
            _fixed_scale->removeChild(_scale);
        }
        _rootNode->removeChild(_fixed_scale);
        delete _fixed_matrix;
    }
    if (_scale)
    {
        _scale->removeChild(_pointer);
        if (_show)
        {
            _rootNode->removeChild(_scale);
        }
        _scale->unref();
        if (_matrix)
        {
            delete _matrix;
        }
    }

    if (_pointer)
    {
        _pointer->unref();
    }
}

void
SurfacePlugin::focusEvent(bool /*focus*/, coMenu * /*menu*/)
{
    //VRPinboard::mainPinboard->lockInteraction = focus;
}

void
SurfacePlugin::AddObject(const char *objName, RenderObject *colorOrText)
{
    _findColor.insert(pair<string, RenderObject *>(objName, colorOrText));
}

void
SurfacePlugin::RemoveObject(const char *contName)
{
    map<string, string>::iterator p = _findObject.find(contName);
    if (p != _findObject.end())
    {
        string objName = p->second;
        _findColor.erase(objName);
        _findObjectSym.erase(objName);
    }
    _findObject.erase(contName);
}

void
SurfacePlugin::RemoveNode(osg::Node *node)
{
    map<osg::Node *, string>::iterator p = _findNodeSym.find(node);
    if (p != _findNodeSym.end())
    {
        _findNode.erase(p->second);
        _findNodeSym.erase(node);
    }
}

RenderObject *
SurfacePlugin::GetColor(const char *objName)
{
    map<string, RenderObject *>::iterator p = _findColor.find(objName);
    if (p != _findColor.end())
    {
        return p->second;
    }
    return NULL;
}

osg::Node *
SurfacePlugin::GetNode(const char *objName)
{
    map<string, osg::Node *>::iterator p = _findNode.find(objName);
    if (p != _findNode.end())
    {
        return p->second;
    }
    return NULL;
}

void
SurfacePlugin::AddNode(const char *objName, osg::Node *node)
{
    if (objName)
    {
        _findNode.insert(pair<string, osg::Node *>(objName, node));
        _findNodeSym.insert(pair<osg::Node *, string>(node, objName));
    }
    else
    {
        fprintf(stderr, "no objName for node named %s\n", node->getName().c_str());
    }
}

void
SurfacePlugin::AddContainer(const char *contName, const char *objName)
{
    _findObject.insert(pair<string, string>(contName, objName));
    _findObjectSym.insert(pair<string, string>(objName, contName));
}

void
SurfacePlugin::ToggleVisibility(string objName)
{
    // find a range of all affected objects
    map<string, osg::Node *>::iterator it = _findNode.find(objName);
    if (it != _findNode.end())
    {
        osg::Node *node = it->second;

        map<osg::Node *, osg::Group *>::iterator p = _parentNode.find(node);
        if (p != _parentNode.end() && p->second)
        {
            p->second->addChild(node);
            node->unref();
            _parentNode.erase(p);
        }
        else
        {
            osg::Group *parent = dynamic_cast<osg::Group *>(node->getParent(0));
            if (parent)
            {
                node->ref();
                parent->removeChild(node);
                _parentNode[node] = parent;
            }
        }
    }
}

void
SurfacePlugin::SuppressOther3DTex(ModuleFeedbackManager *)
{
}

void
SurfacePlugin::DeleteInteractor(coInteractor *)
{
}

void
SurfacePlugin::AddFixedIcon()
{
    osg::Group *rootNode = cover->getObjectsRoot();
    if (_fixed_scale)
    {
        if (_scale && _fixed_scale->getNumChildren() > 0)
        {
            _fixed_scale->removeChild(_scale);
        }
        rootNode->removeChild(_fixed_scale);
        //pfDelete(_fixed_scale);
    }
    if (_fixed_matrix)
    {
        delete _fixed_matrix;
        _fixed_matrix = NULL;
    }
    _fixed_matrix = new osg::Matrix;
    osg::Matrix pointerMatrix, invBaseMatrix;
    // get the pointer's matrix (position and direction)
    pointerMatrix = cover->getPointerMat();
    // get inverse base matrix from scenegraph
    invBaseMatrix = cover->getInvBaseMat();
    // transform pointer matrix into model space
    _fixed_matrix->mult(pointerMatrix, invBaseMatrix);

    _fixed_scale = new MatrixTransform;
    _fixed_scale->setMatrix(*_fixed_matrix);
    if (_scale)
    {
        _fixed_scale->addChild(_scale);
    }
    rootNode->addChild(_fixed_scale);
}

void
SurfacePlugin::RemoveFixedIcon()
{
    osg::Group *rootNode = cover->getObjectsRoot();
    if (_fixed_scale)
    {
        if (_scale && _fixed_scale->getNumChildren() > 0)
        {
            _fixed_scale->removeChild(_scale);
        }
        rootNode->removeChild(_fixed_scale);
        _fixed_scale = NULL;
    }
    if (_fixed_matrix)
    {
        delete _fixed_matrix;
        _fixed_matrix = NULL;
    }
}

osg::Switch *SurfacePlugin::readPointer(const char *basename)
{

    osg::Switch *switchNode = new osg::Switch();

    string wholename = basename;
    osg::Node *normalPointer = coVRFileManager::instance()->loadIcon(wholename.c_str());
    if (!normalPointer)
    {
        cerr << "SurfacePlugin::readPointer: Could not create pointer from "
             << wholename << endl;
        return NULL;
    }

    wholename = basename;
    wholename += "-selected";
    osg::Node *highlightPointer = coVRFileManager::instance()->loadIcon(wholename.c_str());
    if (!highlightPointer)
    {
        highlightPointer = normalPointer;

        cerr << "SurfacePlugin::readPointer: Could not find highlighted pointer for "
             << basename << endl;
    }
    switchNode->addChild(normalPointer);
    switchNode->addChild(highlightPointer);
    switchNode->setSingleChildOn(0);

    return switchNode;
}

void
SurfacePlugin::setActive(bool isActive)
{
    if (_pointer)
    {
        if (isActive)
        {
            _pointer->setSingleChildOn(1);
        }
        else
        {
            _pointer->setSingleChildOn(0);
        }
    }

    cover->sendMessage(this, "AKToolbar",
                       isActive ? PluginMessageTypes::AKToolbarActive : PluginMessageTypes::AKToolbarInactive,
                       0, NULL);
}
