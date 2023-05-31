/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\ 
 **                                                            (C)2007/8 HLRS **
 **                                                                           **
 ** Description: Scenegraph Browser											            **
 **																		                     **
 **                                                                           **
 ** Author: Mario Baalcke	                                                   **
 **                                                                           **
 ** History:  								                                          **
 ** Jun-07   v1	    				       		                                 **
 ** April-08 v2                                                               **
 **                                                                           **
\****************************************************************************/
#include "SGBrowser.h"
#include <cover/coVRPluginSupport.h>
#include <cover/RenderObject.h>
#include <cover/coVRTui.h>
#include <osg/StateAttribute>
#include <osg/Image>
#include <osg/Material>
#include <osg/Geode>
#include <osg/TexGen>
#include <osg/TexEnv>
#include <osg/Switch>
#include <osg/Group>
#include <osg/MatrixTransform>
#include <osg/BlendFunc>
#include <osg/PolygonMode>
#include <util/coTabletUIMessages.h>
#include <osg/Texture2D>
#include <cover/VRSceneGraph.h>
#include <cover/OpenCOVER.h>
#include <net/tokenbuffer.h>

#include <PluginUtil/PluginMessageTypes.h>

#include "LoadFileVisitor.h"
#include "SimVisitor.h"
using namespace osg;

class MyNodeVisitor;
class SGBrowser;

SGBrowser *SGBrowser::plugin = NULL;

bool SGBrowser::pickedObjChanged()
{
    std::string path = "";
    osg::Node *node = NULL;
    std::list<osg::ref_ptr<osg::Node> > selList = selectionManager->getSelectionList();
    std::list<osg::ref_ptr<osg::Node> >::iterator iter = selList.begin();
    while (iter != selList.end())
    {
        node = (*iter).get();
        path = selectionManager->generatePath(node);
        for (auto t: tuis)
            t.tab->sendCurrentNode(node, path);
        iter++;
    }
    return true;
}

bool SGBrowser::processTexture(coTUISGBrowserTab *sGBrowserTab, TexVisitor *texvis, osg::StateSet *ss)
{
    bool found = false;
    if (ss)
    {
        for (int textureNumber = 0; textureNumber < 21; textureNumber++)
        {
            int texGenMode = TEX_GEN_NONE;
            int texEnvMode = TEX_ENV_MODULATE;
            int index = -1;
            StateAttribute *stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXTURE);
            if (stateAttrib)
            {

                Texture2D *texture = dynamic_cast<Texture2D *>(stateAttrib);
                if (texture)
                {
                    Image *image = texture->getImage();
                    if (image)
                    {
                        index = texvis->findImage(image);
                        if (index < 0)
                        {

                            LItem newItem;
                            newItem.image = image;
                            newItem.index = (texvis->getMax() + 1);
                            index = newItem.index;
                            texvis->insertImage(newItem);

                            sGBrowserTab->setTexture(image->t(),
                                                     image->s(),
                                                     image->getPixelSizeInBits(),
                                                     index,
                                                     image->getImageSizeInBytes(),
                                                     reinterpret_cast<char *>(image->data()));

                            found = true;
                        }
                    }
                }

                stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXGEN);
                if (stateAttrib)
                {
                    TexGen *texGen = dynamic_cast<TexGen *>(stateAttrib);
                    if (texGen)
                    {
                        switch (texGen->getMode())
                        {
                        case TexGen::OBJECT_LINEAR:
                            texGenMode = TEX_GEN_OBJECT_LINEAR;
                            break;
                        case TexGen::EYE_LINEAR:
                            texGenMode = TEX_GEN_EYE_LINEAR;
                            break;
                        case TexGen::SPHERE_MAP:
                            texGenMode = TEX_GEN_SPHERE_MAP;
                            break;
                        case TexGen::NORMAL_MAP:
                            texGenMode = TEX_GEN_NORMAL_MAP;
                            break;
                        case TexGen::REFLECTION_MAP:
                            texGenMode = TEX_GEN_REFLECTION_MAP;
                            break;
                        }
                    }
                }
                stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXENV);
                if (stateAttrib)
                {
                    TexEnv *texEnv = dynamic_cast<TexEnv *>(stateAttrib);
                    if (texEnv)
                    {
                        switch (texEnv->getMode())
                        {
                        case TexEnv::DECAL:
                            texEnvMode = TEX_ENV_DECAL;
                            break;
                        case TexEnv::MODULATE:
                            texEnvMode = TEX_ENV_MODULATE;
                            break;
                        case TexEnv::BLEND:
                            texEnvMode = TEX_ENV_BLEND;
                            break;
                        case TexEnv::REPLACE:
                            texEnvMode = TEX_ENV_REPLACE;
                            break;
                        case TexEnv::ADD:
                            texEnvMode = TEX_ENV_ADD;
                            break;
                        default:
                            texEnvMode = TEX_ENV_MODULATE;
                        }
                    }
                }
                sGBrowserTab->setTexture(textureNumber, texEnvMode, texGenMode, index);
            }
        }
    }
    return found;
}

bool SGBrowser::selectionChanged()
{
    bool finished = false;
    StateSet *ss = NULL;

    std::list<osg::ref_ptr<osg::Node> > selectedNodeList = selectionManager->getSelectionList();
    std::list<osg::ref_ptr<osg::Node> >::iterator nodeIter = selectedNodeList.end();
    if (selectedNodeList.size() == 0)
    {
        pickedObject = NULL;
    }
    else
    {
        nodeIter--;

        pickedObject = (*nodeIter).get();

        if (pickedObject.get())
        {
            std::string path = selectionManager->generatePath(pickedObject.get());
            ss = pickedObject.get()->getStateSet();

            finished = true;

            for (auto t: tuis)
            {
                bool found = false;
                t.tab->setCurrentPath(path);
                if (processTexture(t.tab, t.tex, ss))
                    found = true;

                Geode *geode = dynamic_cast<Geode *>(pickedObject.get());
                if (geode)
                {
                    for (unsigned int numdraw = 0; numdraw < geode->getNumDrawables(); numdraw++)
                    {
                        Drawable *drawable = geode->getDrawable(numdraw);

                        if (drawable)
                        {
                            ss = drawable->getStateSet();
                            if (processTexture(t.tab, t.tex, ss))
                                found = true;
                        }
                    }
                }

                if (found)
                    t.tab->finishedNode();
                else
                    t.tab->noTexture();
            }
        }
    }

    if (!finished)
    {
        for (auto t: tuis)
            t.tab->noTexture();
    }

    return true;
}

SGBrowser::SGBrowser()
: coVRPlugin(COVER_PLUGIN_NAME)
, idata(NULL)
, myMes(false)
, reconnect(false)
, selectionManager(NULL)
, restraint(NULL)
, shaderList(NULL)
, linked(false)
{
    assert(plugin == NULL);
    plugin = this;
}

bool SGBrowser::init()
{
    std::cerr << "SGBrowser: #tuis = " <<  OpenCOVER::instance()->numTuis() << std::endl;
    for (size_t i=0; i<OpenCOVER::instance()->numTuis(); ++i)
    {
        auto tui = OpenCOVER::instance()->tui(i);

        auto sgt = new coTUISGBrowserTab(tui, "Scenegraph", OpenCOVER::instance()->tuiTab(i)->getID());
        sgt->setPos(0, 0);
        sgt->setEventListener(this);

        tuis.emplace_back(sgt,
                          new MyNodeVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN, sgt),
                          new TexVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN, sgt));
    }

    selectionManager = coVRSelectionManager::instance();
    selectionManager->addListener(this);

    restraint = new coRestraint();

    shaderList = coVRShaderList::instance();

    myMes = false;
    reconnect = false;

    return true;
}

// this is called if the plugin is removed at runtime
SGBrowser::~SGBrowser()
{

    if (selectionManager)
        selectionManager->removeListener(this);

    for (auto t: tuis)
    {
        delete t.tex;
        delete t.vis;
        delete t.tab;
    }

    delete restraint;

    plugin = NULL;
}

void SGBrowser::tabletReleaseEvent(coTUIElement *tUIItem)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            if (t.tab->getImageMode() == SEND_IMAGES)
            {
                t.tex->setTexFound(false);
                t.tex->apply(*cover->getObjectsRoot());
                if (t.tex->getTexFound())
                    t.tab->finishedTraversing();
                else
                    t.tab->noTexture();
            }
            if (t.tab->getImageMode() == SEND_LIST)
            {
                t.tex->sendImageList();
                t.tab->finishedTraversing();
            }

            break;
        }
    }
}

void SGBrowser::tabletPressEvent(coTUIElement *tUIItem)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            t.vis->updateMyParent(*cover->getObjectsRoot());
            t.tab->sendEnd();
            break;
        }
    }
}

void SGBrowser::tabletSelectEvent(coTUIElement *tUIItem)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            t.vis->addMyNode();
            break;
        }
    }
}

void SGBrowser::tabletChangeModeEvent(coTUIElement *tUIItem)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            int mode = t.tab->getVisMode();

            if (mode == UPDATE_COLOR)
            {
                selectionManager->setSelectionColor(t.tab->getR(), t.tab->getG(), t.tab->getB());
            }
            else if (mode == UPDATE_WIRE)
            {
                selectionManager->setSelectionWire(t.tab->getPolyMode());
            }
            else if (mode == UPDATE_SEL)
            {
                selectionManager->showhideSelection(t.tab->getSelMode());
            }

            break;
        }
    }
}

void SGBrowser::tabletFindEvent(coTUIElement *tUIItem)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            t.vis->myInit();
            t.vis->apply(*cover->getObjectsRoot());
            t.vis->traverseFindList();
            t.vis->sendMyFindList();
            t.tab->sendEnd();
            break;
        }
    }
}
void SGBrowser::tabletCurrentEvent(coTUIElement *tUIItem)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            t.vis->myInit();
            t.vis->traverseFindList();
            t.vis->sendMyFindList();
            break;
        }
    }
}

void SGBrowser::tabletLoadFilesEvent(char *nodeName)
{
    osg::Node *node = selectionManager->validPath(nodeName);
    if (node)
    {
        LoadFileVisitor visitor;
        node->traverse(visitor);
        //LoadFile FileObj;
        // FileObj.load(node->asGroup());
    }
}

/*void SGBrowser::switchSim( char *nodeName )
{
   cout<<"SGBrowser::switchSim aufgerufen"<<nodeName<<endl;
}*/

/*void SGBrowser::hideNode ( )
{
   TokenBuffer tb;
   tb << sGBrowserTab->getShowHidePath().c_str();
   tb << sGBrowserTab->getShowHideParentPath().c_str();

   //if(currentMode == HIDE_NODE)
     // cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, 0, tb.get_length(),tb.get_data());
   //else
      cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, 1, tb.get_length(),tb.get_data());
   
}*/

void SGBrowser::tabletDataEvent(coTUIElement *tUIItem, TokenBuffer &tb)
{
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            auto sGBrowserTab = t.tab;

            int mode;
            const char *path, *pPath;

            TokenBuffer _tb;

            tb >> mode;
            if (mode == CENTER_OBJECT)
            {
                VRSceneGraph::instance()->viewAll();
            }
            if (mode == SET_PROPERTIES)
            {
                int depthOnly, all, remove, j, trans;
                const char *children;
                tb >> path;
                tb >> pPath;
                tb >> depthOnly;
                tb >> children;
                tb >> all;
                tb >> remove;
                tb >> trans;
                for (j = 0; j < 4; j++)
                    tb >> sGBrowserTab->diffuse[j];
                for (j = 0; j < 4; j++)
                    tb >> sGBrowserTab->specular[j];
                for (j = 0; j < 4; j++)
                    tb >> sGBrowserTab->ambient[j];
                for (j = 0; j < 4; j++)
                    tb >> sGBrowserTab->emissive[j];
                for (j = 0; j < 16; ++j)
                {
                    tb >> sGBrowserTab->matrix[j];
                    ;
                }

                _tb << path;
                _tb << pPath;
                _tb << depthOnly;
                _tb << children;
                _tb << all;
                _tb << remove;
                _tb << trans;
                for (j = 0; j < 4; j++)
                    _tb << sGBrowserTab->diffuse[j];
                for (j = 0; j < 4; j++)
                    _tb << sGBrowserTab->specular[j];
                for (j = 0; j < 4; j++)
                    _tb << sGBrowserTab->ambient[j];
                for (j = 0; j < 4; j++)
                    _tb << sGBrowserTab->emissive[j];
                for (j = 0; j < 16; ++j)
                {
                    _tb << sGBrowserTab->matrix[j];
                    ;
                }

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetProperties,
                                   _tb.getData().length(), _tb.getData().data());
            }
            if (mode == GET_PROPERTIES)
            {
                tb >> path;
                tb >> pPath;

                _tb << path;
                _tb << pPath;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserGetProperties,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == REMOVE_TEXTURE)
            {
                int texNumber;
                tb >> path;
                tb >> texNumber;

                _tb << path;
                _tb << texNumber;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserRemoveTexture,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == GET_SHADER)
            {
                std::list<coVRShader *>::iterator iter;

                for (iter = shaderList->begin(); iter != shaderList->end(); iter++)
                {
                    std::string name = (*iter)->getName();
                    sGBrowserTab->sendShader(name);
                }
            }
            if (mode == GET_UNIFORMS)
            {

                const char *name;
                tb >> name;
                std::string shaderName = std::string(name);
                coVRShader *_shader = shaderList->get(shaderName);
                std::list<coVRUniform *> uniformList = _shader->getUniforms();
                if (_shader->getFragmentShader().get() != NULL && _shader->getVertexShader().get() != NULL)
                {
                    std::string fragmentSource = _shader->getFragmentShader().get()->getShaderSource();
                    std::string vertexSource = _shader->getVertexShader().get()->getShaderSource();

                    std::string tessControlSource, tessEvalSource;
                    if (_shader->getTessControlShader().get() != NULL && _shader->getTessEvalShader().get() != NULL)
                    {
                        tessControlSource = _shader->getTessControlShader()->getShaderSource();
                        tessEvalSource = _shader->getTessEvalShader()->getShaderSource();
                    }

                    std::string geometrySource;
                    if (_shader->getGeometryShader().get())
                    {
                        geometrySource = _shader->getGeometryShader().get()->getShaderSource();
                        sGBrowserTab->updateShaderNumVertices(shaderName, _shader->getNumVertices());
                        sGBrowserTab->updateShaderInputType(shaderName, _shader->getInputType());
                        sGBrowserTab->updateShaderOutputType(shaderName, _shader->getOutputType());
                    }

                    sGBrowserTab->sendShaderSource(vertexSource, fragmentSource, geometrySource, tessControlSource, tessEvalSource);

                    std::list<coVRUniform *>::iterator iter;
                    for (iter = uniformList.begin(); iter != uniformList.end(); iter++)
                    {
                        std::string name = (*iter)->getName();
                        std::string texFile = (*iter)->getTextureFileName();
                        std::string type = (*iter)->getType();
                        std::string value = (*iter)->getValue();
                        std::string min = (*iter)->getMin();
                        std::string max = (*iter)->getMax();
                        sGBrowserTab->sendUniform(name, type, value, min, max, texFile);
                    }
                }
                else
                {
                    if (_shader->getFragmentShader().get() == NULL)
                        cerr << "missing fragment shader" << endl;
                    if (_shader->getVertexShader().get() == NULL)
                        cerr << "missing vertex shader" << endl;
                }
            }
            if (mode == REMOVE_SHADER)
            {
                tb >> path;

                _tb << path;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserRemoveShader,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == STORE_SHADER)
            {
                const char *shaderName;
                tb >> shaderName;
                coVRShader *_shader = shaderList->get(shaderName);
                if (_shader)
                {
                    _shader->storeMaterial();
                }
            }
            if (mode == SET_SHADER)
            {
                const char *name;
                tb >> path;
                tb >> name;

                _tb << path;
                _tb << name;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetShader,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == SET_UNIFORM)
            {
                const char *Sname, *Uname, *Uvalue, *StexFile;
                tb >> Sname;
                tb >> Uname;
                tb >> Uvalue;
                tb >> StexFile;

                _tb << Sname;
                _tb << Uname;
                _tb << Uvalue;
                _tb << StexFile;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetUniform,
                    _tb.getData().length(), _tb.getData().data());
            }

            if (mode == SET_INPUT_TYPE)
            {
                const char *Sname;
                int value;
                tb >> Sname;
                tb >> value;

                _tb << Sname;
                _tb << value;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetInputType,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == SET_OUTPUT_TYPE)
            {
                const char *Sname;
                int value;
                tb >> Sname;
                tb >> value;

                _tb << Sname;
                _tb << value;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetOutputType,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == SET_NUM_VERT)
            {
                const char *Sname;
                int value;
                tb >> Sname;
                tb >> value;

                _tb << Sname;
                _tb << value;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetNumVertex,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == SET_VERTEX)
            {
                const char *Sname, *Vvalue;
                tb >> Sname;
                tb >> Vvalue;

                _tb << Sname;
                _tb << Vvalue;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetVertex,
                    _tb.getData().length(), _tb.getData().data());
            }

            if (mode == SET_TESSCONTROL)
            {
                const char *Sname, *Vvalue;
                tb >> Sname;
                tb >> Vvalue;

                _tb << Sname;
                _tb << Vvalue;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetTessControl,
                    _tb.getData().length(), _tb.getData().data());
            }

            if (mode == SET_TESSEVAL)
            {
                const char *Sname, *Vvalue;
                tb >> Sname;
                tb >> Vvalue;

                _tb << Sname;
                _tb << Vvalue;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetTessEval,
                    _tb.getData().length(), _tb.getData().data());
            }

            if (mode == SET_FRAGMENT)
            {
                const char *Sname, *Fvalue;
                tb >> Sname;
                tb >> Fvalue;

                _tb << Sname;
                _tb << Fvalue;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetFragment,
                    _tb.getData().length(), _tb.getData().data());
            }
            if (mode == SET_GEOMETRY)
            {
                const char *Sname, *Gvalue;
                tb >> Sname;
                tb >> Gvalue;

                _tb << Sname;
                _tb << Gvalue;

                myMes = true;

                cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetGeometry,
                    _tb.getData().length(), _tb.getData().data());
            }

            break;
        }
    }
}

void SGBrowser::tabletEvent(coTUIElement *tUIItem)
{
    // add Texture
    for (auto t: tuis)
    {
        if (tUIItem == t.tab)
        {
            auto sGBrowserTab = t.tab;
            TokenBuffer _tb;
            _tb << sGBrowserTab->getChangedPath();
            _tb << sGBrowserTab->getWidth();
            _tb << sGBrowserTab->getHeight();
            _tb << sGBrowserTab->getDepth();
            _tb << sGBrowserTab->getIndex();

            _tb << sGBrowserTab->getTextureNumber();
            _tb << sGBrowserTab->getTextureMode();
            _tb << sGBrowserTab->getTextureTexGenMode();
            _tb << sGBrowserTab->hasAlpha();
            int dataLength = (int)sGBrowserTab->getDataLength();
            _tb << dataLength;

            _tb.addBinary(sGBrowserTab->getData(), dataLength);

            cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserSetTexture,
                _tb.getData().length(), _tb.getData().data());

            break;
        }
    }
}
void SGBrowser::addNode(osg::Node *node, const RenderObject *obj)
{
    (void)obj;

    for (auto t: tuis)
    {
        t.vis->myUpdate(node);
        osg::Group *my = node->asGroup();
        if (!my)
            return;

        for (unsigned int i = 0; my && i < my->getNumChildren(); i++)
        {
            t.vis->updateMyChild(my->getChild(i));
        }
    }

#if 0
    // takes too much time ...
    for (auto sgt: sGBrowserTab)
    {
        TexVisitor *texvis = new TexVisitor(NodeVisitor::TRAVERSE_ALL_CHILDREN, sgt);
        texvis->apply(*cover->getObjectsRoot());
        if(texvis->getTexFound())
            sgt->finishedTraversing();
        else
            sgt->noTexture();
    }
#endif
}

void SGBrowser::removeNode(Node *node, bool /*isGroup*/, Node * /*realNode*/)
{
    int numParents = node->getNumParents();
    std::string path = selectionManager->generatePath(node);
    std::string parentPath = "";
    for (int i = 0; i < numParents; i++)
    {
        parentPath = selectionManager->generatePath(node->getParent(i));
        for (auto t: tuis)
            t.tab->sendRemoveNode(path, parentPath);
    }
}

void SGBrowser::preFrame()
{
}
void SGBrowser::message(int toWhom, int type, int len, const void *buf)
{
    TokenBuffer tb((const char *)buf, len);
    //gottlieb<

    //conflictif (type == 14)//message from PLMXML -> set simpair

    if (type == PluginMessageTypes::PLMXMLSetSimPair) //message from PLMXML -> show Node
    {
        const char *nodePath, *simPath, *simName;
        tb >> nodePath;
        tb >> simPath;
        tb >> simName;
        for (auto t: tuis)
            t.tab->setSimPair(nodePath, simPath, simName);
    }

    if (type == PluginMessageTypes::PLMXMLShowNode) //message from PLMXML -> show Node
    {
        const char *nodePath;
        const char *simPath;
        tb >> nodePath;
        tb >> simPath;
        for (auto t: tuis)
            t.tab->hideSimNode(false, nodePath, simPath);
    }
    if (type == PluginMessageTypes::PLMXMLHideNode) //message from PLMXML -> hide Node
    {
        const char *nodePath;
        const char *simPath;
        tb >> nodePath;
        tb >> simPath;
        for (auto t: tuis)
            t.tab->hideSimNode(true, nodePath, simPath);
    }
    if (type == PluginMessageTypes::PLMXMLLoadFiles) //message from PLMXML ->create Load-Files Button
    {
        const char *Sname;
        tb >> Sname;
        for (auto t: tuis)
            t.tab->loadFilesFlag(true); //calls the loadFilesFlag-methode in /src/renderer/OpenCOVER/cover/coTabletUI.cpp
    } //>gottlieb
    if (type == PluginMessageTypes::SGBrowserSetFragment) // Set Fragment
    {
        const char *Sname, *Fvalue;
        tb >> Sname;
        tb >> Fvalue;
        std::string SName = std::string(Sname);
        std::string FValue = std::string(Fvalue);

        coVRShader *_shader = shaderList->get(SName);

        _shader->getFragmentShader().get()->setShaderSource(FValue);
        _shader->getFragmentShader()->dirtyShader();

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderSourceF(Sname, Fvalue);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetGeometry) // Set Fragment
    {
        const char *Sname, *Gvalue;
        tb >> Sname;
        tb >> Gvalue;
        std::string SName = std::string(Sname);
        std::string GValue = std::string(Gvalue);

        coVRShader *_shader = shaderList->get(SName);

        if (_shader->getGeometryShader().get())
        {
            _shader->getGeometryShader().get()->setShaderSource(GValue);
            _shader->getGeometryShader()->dirtyShader();
        }
        else
        {
            _shader->getGeometryShader() = new osg::Shader(osg::Shader::GEOMETRY, GValue);

            _shader->getProgram()->addShader(_shader->getGeometryShader().get());
            _shader->getProgram()->setParameter(GL_GEOMETRY_VERTICES_OUT_EXT, _shader->getNumVertices());
            _shader->getProgram()->setParameter(GL_GEOMETRY_INPUT_TYPE_EXT, _shader->getInputType());
            _shader->getProgram()->setParameter(GL_GEOMETRY_OUTPUT_TYPE_EXT, _shader->getOutputType());
        }

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderSourceG(Sname, Gvalue);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetNumVertex) // Set Vertex
    {
        const char *Sname;
        int value;
        tb >> Sname;
        tb >> value;
        std::string SName = std::string(Sname);

        coVRShader *_shader = shaderList->get(SName);
        _shader->setNumVertices(value);

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderNumVertices(Sname, value);
        }
    }

    if (type == PluginMessageTypes::SGBrowserSetOutputType)
    {
        const char *Sname;
        int value;
        tb >> Sname;
        tb >> value;
        std::string SName = std::string(Sname);

        coVRShader *_shader = shaderList->get(SName);
        _shader->setOutputType(value);

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderOutputType(Sname, value);
        }
    }

    if (type == PluginMessageTypes::SGBrowserSetInputType)
    {
        const char *Sname;
        int value;
        tb >> Sname;
        tb >> value;
        std::string SName = std::string(Sname);

        coVRShader *_shader = shaderList->get(SName);

        _shader->setInputType(value);

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderInputType(Sname, value);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetVertex) // Set Vertex
    {
        const char *Sname, *Vvalue;
        tb >> Sname;
        tb >> Vvalue;
        std::string SName = std::string(Sname);
        std::string VValue = std::string(Vvalue);

        coVRShader *_shader = shaderList->get(SName);

        _shader->getVertexShader().get()->setShaderSource(VValue);
        _shader->getVertexShader()->dirtyShader();

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderSourceV(Sname, Vvalue);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetTessControl) // Set TessControl
    {
        const char *Sname, *Vvalue;
        tb >> Sname;
        tb >> Vvalue;
        std::string SName = std::string(Sname);
        std::string VValue = std::string(Vvalue);

        coVRShader *_shader = shaderList->get(SName);

        _shader->getTessControlShader().get()->setShaderSource(VValue);
        _shader->getTessControlShader()->dirtyShader();

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderSourceTC(Sname, Vvalue);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetTessEval) // Set TessEval
    {
        const char *Sname, *Vvalue;
        tb >> Sname;
        tb >> Vvalue;
        std::string SName = std::string(Sname);
        std::string VValue = std::string(Vvalue);

        coVRShader *_shader = shaderList->get(SName);

        _shader->getTessEvalShader().get()->setShaderSource(VValue);
        _shader->getTessEvalShader()->dirtyShader();

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateShaderSourceTE(Sname, Vvalue);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetUniform) // Set Uniform
    {
        const char *Sname, *Uname, *Uvalue, *StexFile;
        tb >> Sname;
        tb >> Uname;
        tb >> Uvalue;
        tb >> StexFile;
        std::string SName = std::string(Sname);
        std::string UName = std::string(Uname);
        std::string UValue = std::string(Uvalue);
        std::string STexFile = std::string(StexFile);

        coVRShader *_shader = shaderList->get(SName);
        std::list<coVRUniform *> uniformList = _shader->getUniforms();
        std::list<coVRUniform *>::iterator iter;
        for (iter = uniformList.begin(); iter != uniformList.end(); iter++)
        {
            if (UName == (*iter)->getName())
            {
                (*iter)->setValue(UValue.c_str());
                (*iter)->setTexture(STexFile.c_str());
            }
        }

        if (myMes)
        {
            myMes = false;
        }
        else
        {
            for (auto t: tuis)
                t.tab->updateUniform(SName, Uname, Uvalue, StexFile);
        }
    }

    if (type == PluginMessageTypes::SGBrowserSetShader) // Set Shader
    {
        const char *path, *name;
        tb >> path;
        tb >> name;
        std::string nodePath = std::string(path);
        std::string shaderName = std::string(name);
        osg::Node *node = selectionManager->validPath(nodePath);
        if (node)
        {
            coVRShader *_shader = shaderList->get(shaderName);
            _shader->apply(node);
        }
    }
    if (type == PluginMessageTypes::SGBrowserRemoveShader) //Remove Shader
    {
        const char *path;
        tb >> path;

        std::string nodePath = std::string(path);

        osg::Node *node = selectionManager->validPath(nodePath);
        if (node)
        {
            shaderList->remove(node);
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetTexture) // add Texture
    {
        std::string path;
        int width, height, depth, texIndex;
        int texNumber, texMode, texGenMode, Alpha, dataLength;

        tb >> path;
        tb >> width;
        tb >> height;
        tb >> depth;
        tb >> texIndex;

        tb >> texNumber;
        tb >> texMode;
        tb >> texGenMode;
        tb >> Alpha;
        tb >> dataLength;
        const char *imagedata = tb.getBinary(dataLength);
        idata = new char[dataLength];
        memcpy(idata, imagedata, dataLength);

        osg::Node *validNode = selectionManager->validPath(path);
        if (validNode)
        {
            StateSet *ss = NULL;
            Geode *geode = dynamic_cast<Geode *>(validNode);
            if (geode)
            {
                Drawable *drawable = geode->getDrawable(0);
                if (drawable)
                {
                    ss = drawable->getOrCreateStateSet();
                }
                else
                {
                    ss = geode->getOrCreateStateSet();
                }
            }
            else
            {
                ss = validNode->getOrCreateStateSet();
            }

            int intTexFormat = GL_RGBA8;
            int GLtype = GL_UNSIGNED_BYTE;
            int pixelFormat = GL_RGB;

            if (depth == 24)
            {
                pixelFormat = GL_RGB;
                intTexFormat = GL_RGB8;
            }
            else if (depth == 32)
            {
                pixelFormat = GL_RGBA;
                intTexFormat = GL_RGBA8;
            }
            else if (depth == 16)
            {
                pixelFormat = GL_LUMINANCE_ALPHA;
                intTexFormat = GL_RGBA8;
            }
            else if (depth == 8)
            {
                pixelFormat = GL_LUMINANCE;
                intTexFormat = GL_RGB8;
            }

            if (ss)
            {
                Texture2D *texture;
                Image *image = NULL;
                TexGen *texGen = NULL;
                TexEnv *texEnv = NULL;

                texture = dynamic_cast<Texture2D *>(ss->getTextureAttribute(texNumber, StateAttribute::TEXTURE));

                StateAttribute *stateAttrib = ss->getTextureAttribute(texNumber, StateAttribute::TEXENV);
                if (stateAttrib)
                {
                    texEnv = dynamic_cast<TexEnv *>(stateAttrib);
                }
                stateAttrib = ss->getTextureAttribute(texNumber, StateAttribute::TEXGEN);
                if (stateAttrib)
                {
                    texGen = dynamic_cast<TexGen *>(stateAttrib);
                }
                if (texEnv == NULL)
                    texEnv = new TexEnv;
                if (texGen == NULL)
                    texGen = new TexGen;

                if (!texture)
                {
                    texture = new Texture2D;

                    texture->setDataVariance(Object::DYNAMIC);
                    texture->setFilter(Texture::MIN_FILTER, Texture::LINEAR_MIPMAP_LINEAR);
                    texture->setFilter(Texture::MAG_FILTER, Texture::LINEAR);
                    texture->setWrap(Texture::WRAP_S, Texture::REPEAT);
                    texture->setWrap(Texture::WRAP_T, Texture::REPEAT);
                    ss->setTextureAttributeAndModes(texNumber, texture, StateAttribute::ON);
                    texGenMode = TEX_GEN_OBJECT_LINEAR;
                }
                switch (texGenMode)
                {
                case TEX_GEN_OBJECT_LINEAR:
                    texGen->setMode(TexGen::OBJECT_LINEAR);
                    break;
                case TEX_GEN_EYE_LINEAR:
                    texGen->setMode(TexGen::EYE_LINEAR);
                    break;
                case TEX_GEN_SPHERE_MAP:
                    texGen->setMode(TexGen::SPHERE_MAP);
                    break;
                case TEX_GEN_NORMAL_MAP:
                    texGen->setMode(TexGen::NORMAL_MAP);
                    break;
                case TEX_GEN_REFLECTION_MAP:
                    texGen->setMode(TexGen::REFLECTION_MAP);
                    break;
                }
                switch (texMode)
                {
                case TEX_ENV_DECAL:
                    texEnv->setMode(TexEnv::DECAL);
                    break;
                case TEX_ENV_MODULATE:
                    texEnv->setMode(TexEnv::MODULATE);
                    break;
                case TEX_ENV_BLEND:
                    texEnv->setMode(TexEnv::BLEND);
                    break;
                case TEX_ENV_REPLACE:
                    texEnv->setMode(TexEnv::REPLACE);
                    break;
                case TEX_ENV_ADD:
                    texEnv->setMode(TexEnv::ADD);
                    break;
                default:
                    texEnv->setMode(TexEnv::DECAL);
                }
                if (Alpha)
                {
                    ss->setRenderingHint(StateSet::TRANSPARENT_BIN);
                    ss->setNestRenderBins(false);
                    ss->setMode(GL_BLEND /*StateAttribute::BLENDFUNC*/, StateAttribute::ON);
                    BlendFunc *blendFunc = new BlendFunc();
                    blendFunc->setFunction(BlendFunc::SRC_ALPHA, BlendFunc::ONE_MINUS_SRC_ALPHA);
                    ss->setAttributeAndModes(blendFunc, StateAttribute::ON);
                }
                if (texGenMode != TEX_GEN_NONE)
                    ss->setTextureAttributeAndModes(texNumber, texGen, StateAttribute::ON);
                else
                    ss->setTextureAttributeAndModes(texNumber, texGen, StateAttribute::OFF);
                ss->setTextureAttributeAndModes(texNumber, texEnv, StateAttribute::ON);

                osg::Image *foundImage = nullptr;
                if (!tuis.empty())
                {
                    auto texvis = tuis.front().tex;

                    foundImage = texvis->getImage(texIndex);
                    if (foundImage)
                    {
                        image = foundImage;
                    }

                    else
                    {
                        image = new Image;
                        image->setImage(width,
                                        height,
                                        depth,
                                        intTexFormat,
                                        pixelFormat,
                                        GLtype,
                                        (unsigned char *)(idata),
                                        Image::USE_NEW_DELETE);

                        Texture2D *helpTexture = new Texture2D;
                        helpTexture->setImage(image);

                        LItem newItem;
                        newItem.image = image;
                        newItem.index = texIndex;
                        texvis->insertImage(newItem);
                    }
                    texture->setImage(image);
                }
            }
        }
    }
    if (type == PluginMessageTypes::SGBrowserRemoveTexture) // Remove Texture
    {
        int texNumber;
        const char *path;
        osg::Node *propNode = NULL;

        tb >> path;
        tb >> texNumber;

        std::string _path = std::string(path);
        propNode = selectionManager->validPath(_path);
        StateSet *ss = NULL;
        if (propNode)
        {
            ss = propNode->getStateSet();
            if (ss)
            {
                ss->removeTextureAttribute(texNumber, StateAttribute::TEXTURE);
            }
            Geode *geode = dynamic_cast<Geode *>(propNode);
            if (geode)
            {
                for (unsigned int numdraw = 0; numdraw < geode->getNumDrawables(); numdraw++)
                {
                    Drawable *drawable = geode->getDrawable(numdraw);
                    if (drawable)
                    {
                        ss = drawable->getStateSet();
                        if (ss)
                        {
                            ss->removeTextureAttribute(texNumber, StateAttribute::TEXTURE);
                        }
                    }
                }
            }
        }
    }
    if (type == PluginMessageTypes::SGBrowserSetProperties) //SET
    {
        int all, remove, transparent;
        int j, depthOnly;
        const char *path, *pPath, *children;
        osg::Node *propNode = NULL;

        tb >> path;
        tb >> pPath;
        tb >> depthOnly;
        tb >> children;
        tb >> all;
        tb >> remove;
        tb >> transparent;
        float diffuse[4], specular[4], ambient[4], emissive[4];
        for (j = 0; j < 4; j++)
            tb >> diffuse[j];
        for (j = 0; j < 4; j++)
            tb >> specular[j];
        for (j = 0; j < 4; j++)
            tb >> ambient[j];
        for (j = 0; j < 4; j++)
            tb >> emissive[j];

        std::string _path = std::string(path);
        std::string _pPath = std::string(pPath);
        std::string _children = std::string(children);

        propNode = selectionManager->validPath(_path);

        float matrix[16];
        osg::MatrixTransform *osgTMatrix = dynamic_cast<osg::MatrixTransform *>(propNode);
        if (osgTMatrix != NULL)
        {
            for (int i = 0; i < 16; ++i)
            {
                tb >> matrix[i];
            }

            osgTMatrix->setMatrix(osg::Matrix(matrix));
        }

        Geode *geode = dynamic_cast<Geode *>(propNode);
        if (propNode)
        {
            //setProperties(propNode);
            StateSet *stateset = NULL;
            StateAttribute *stateAttrib = NULL;
            Material *mat = NULL;
            if (geode && geode->getNumDrawables() > 0)
            {
                Drawable* drawable = geode->getDrawable(0);
                if (drawable)
                {
                    stateset = drawable->getStateSet();
                }
            }
            if (stateset == nullptr)
            {
                stateset = propNode->getOrCreateStateSet();
            }
            stateAttrib = stateset->getAttribute(StateAttribute::MATERIAL);
            mat = dynamic_cast<Material *>(stateAttrib);
            if (remove && mat)
            {
                stateset->removeAttribute(mat);
            }
            if (!remove && diffuse[0] >= 0)
            {
                if (!mat)
                {
                    mat = new osg::Material();
                }
                mat->setDiffuse(Material::FRONT_AND_BACK, osg::Vec4(diffuse[0], diffuse[1], diffuse[2], diffuse[3]));
                if (specular[0] >= 0)
                    mat->setSpecular(Material::FRONT_AND_BACK, osg::Vec4(specular[0], specular[1], specular[2], specular[3]));
                if (ambient[0] >= 0)
                    mat->setAmbient(Material::FRONT_AND_BACK, osg::Vec4(ambient[0], ambient[1], ambient[2], ambient[3]));
                if (emissive[0] >= 0)
                    mat->setEmission(Material::FRONT_AND_BACK, osg::Vec4(emissive[0], emissive[1], emissive[2], emissive[3]));

                if (diffuse[3] < 1.0)
                {
                    transparent = 1;
                }

                if (!geode)
                    stateset->setAttribute(mat, osg::StateAttribute::OVERRIDE);
            }
            if (transparent)
            {
                stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
                stateset->setRenderingHint(StateSet::TRANSPARENT_BIN);
                stateset->setNestRenderBins(false);
            }
            else
            {
                stateset->setMode(GL_BLEND, osg::StateAttribute::OFF);
                stateset->setRenderingHint(StateSet::OPAQUE_BIN);
                stateset->setNestRenderBins(false);
            }

            if (depthOnly)
            {

                stateset->setRenderBinDetails(-1, "RenderBin");
                stateset->setNestRenderBins(false);
                stateset->setAttributeAndModes(cover->getNoFrameBuffer().get(), StateAttribute::ON);
            }
            else
            {
                stateset->removeAttribute(cover->getNoFrameBuffer().get());
            }

            osg::Switch *switchNode = dynamic_cast<osg::Switch *>(propNode);
            if (switchNode)
            {

                if (all)
                {
                    switchNode->setAllChildrenOn();
                }
                else if (_children != "NOCHANGE")
                {
                    restraint->clear();
                    restraint->add(_children.c_str());
                    coRestraint res = *restraint;
                    for (unsigned i = 0; i < switchNode->getNumChildren(); i++)
                    {
                        if (res(i + 1))
                        {
                            switchNode->setChildValue(switchNode->getChild(i), true);
                        }
                        else
                        {
                            switchNode->setChildValue(switchNode->getChild(i), false);
                        }
                    }
                }
            }
        }
    }
    if (type == PluginMessageTypes::SGBrowserGetProperties) //GET
    {
#if 0
        for (auto t: tuis)
        {
            for (int j=0; j<4; ++j)
            {
                t.tab->diffuse[j] = diffuse[j];
                t.tab->specular[j] = specular[j];
                t.tab->ambient[j] = ambient[j];
                t.tab->emissive[j] = emissive[j];
            }
            for (int i=0; i<16; ++i)
            {
                t.tab->matrix[i] = matrix[i];
            }
        }
#endif

        int i;
        const char *path, *pPath;
        osg::Node *propNode = NULL;

        tb >> path;
        tb >> pPath;
        std::string _path = std::string(path);
        std::string _pPath = std::string(pPath);
        int depthMode, transparent;
        float matrix[16];

        propNode = selectionManager->validPath(_path);
        if (propNode)
        {
            //getProperties(propNode);
            Geode *geode = dynamic_cast<Geode *>(propNode);
            StateSet *stateset = NULL;
            if (geode && geode->getNumDrawables()>0)
            {
                Drawable *drawable = geode->getDrawable(0);
                if (drawable)
                {
                    stateset = drawable->getStateSet();
                }
            }
            if(stateset == nullptr)
            {
                stateset = propNode->getOrCreateStateSet();
            }

            if (stateset)
            {
                StateAttribute *stateAttrib = stateset->getAttribute(StateAttribute::MATERIAL);
                for (auto t: tuis)
                {
                    auto sGBrowserTab = t.tab;
                    Material *mat = dynamic_cast<Material *>(stateAttrib);
                    if (mat)
                    {
                        osg::Vec4 currentDiffuseColor = mat->getDiffuse(Material::FRONT);
                        osg::Vec4 currentSpecularColor = mat->getSpecular(Material::FRONT);
                        osg::Vec4 currentAmbientColor = mat->getAmbient(Material::FRONT);
                        osg::Vec4 currentEmissiveColor = mat->getEmission(Material::FRONT);

                        for (i = 0; i < 4; i++)
                        {
                            sGBrowserTab->diffuse[i] = currentDiffuseColor[i];
                            sGBrowserTab->specular[i] = currentSpecularColor[i];
                            sGBrowserTab->ambient[i] = currentAmbientColor[i];
                            sGBrowserTab->emissive[i] = currentEmissiveColor[i];
                        }
                    }
                    else
                    {
                        for (i = 0; i < 4; i++)
                        {
                            sGBrowserTab->diffuse[i] = -1.0f;
                            sGBrowserTab->specular[i] = -1.0f;
                            sGBrowserTab->ambient[i] = -1.0f;
                            sGBrowserTab->emissive[i] = -1.0f;
                        }
                    }
                }

                if (stateset->getAttribute(cover->getNoFrameBuffer().get()->getType()))
                {
                    depthMode = 1;
                }
                else
                {
                    depthMode = 0;
                }
                if (stateset->getMode(GL_BLEND) == osg::StateAttribute::ON)
                {
                    transparent = 1;
                }
                else
                {
                    transparent = 0;
                }

                osg::MatrixTransform *osgTMatrix = dynamic_cast<osg::MatrixTransform *>(propNode);
                if (osgTMatrix != NULL)
                {
                    osg::Matrix osgMat = osgTMatrix->getMatrix();
                    for (int i = 0; i < 4; ++i)
                        for (int j = 0; j < 4; ++j)
                        {
                            matrix[i * 4 + j] = osgMat(i, j);
                        }
                }
            }
            for (auto t: tuis)
                t.tab->sendProperties(_path, _pPath, depthMode, transparent, matrix);
        }
    }
    if ((type == PluginMessageTypes::SGBrowserHideNode) || (type == PluginMessageTypes::SGBrowserShowNode))
    {

        osg::Node *myNode = NULL;
        osg::Node *shParent = NULL;
        std::string path, pPath;
        tb >> path;
        tb >> pPath;
        myNode = selectionManager->validPath(path);
        shParent = selectionManager->validPath(pPath);
        osg::Group *parent = NULL;
        osg::Group *helpC = NULL;
        osg::Node *child = NULL;

        //std::cerr << "SGBrowser::message info: " << (type == PluginMessageTypes::SGBrowserHideNode ? "hide " : "show ") << path << std::endl;

        if (shParent)
            parent = shParent->asGroup();

        if (myNode && parent)
        {
            std::string className = parent->className();
            if (className == "Switch")
            {
                int number = -1;
                for (unsigned int c = 0; c < parent->getNumChildren(); c++)
                {

                    child = parent->getChild(c);
                    helpC = child->asGroup();

                    while (helpC && selectionManager->isHelperNode(helpC))
                    {
                        if (helpC->getNumChildren() == 0)
                            break;
                        child = helpC->getChild(0);
                        helpC = child->asGroup();
                    }

                    if (child == myNode)
                    {
                        number = c;
                        break;
                    }
                }
                osg::Switch *mySwitch = (osg::Switch *)parent;
                osg::Node *S_child = NULL;
                if (number >= 0)
                    S_child = mySwitch->getChild(number);

                if (S_child)
                {
                    if (type == PluginMessageTypes::SGBrowserShowNode)
                    {
                        mySwitch->setAllChildrenOff();
                        mySwitch->setChildValue(S_child, true);
                    }
                    else

                        mySwitch->setChildValue(S_child, false);
                }
            }
            else
            {
                osg::Group *switchGroup = NULL;
                osg::Switch *mySwitch = NULL;
                switchGroup = selectionManager->getHelperNode(parent, myNode, selectionManager->SHOWHIDE);

                if (!switchGroup)
                {
                    mySwitch = new osg::Switch();
                    mySwitch->setName("SGBrowser:Switch");
                    if (type == PluginMessageTypes::SGBrowserShowNode)
                        selectionManager->insertHelperNode(parent, myNode, mySwitch, selectionManager->SHOWHIDE, true);
                    else
                        selectionManager->insertHelperNode(parent, myNode, mySwitch, selectionManager->SHOWHIDE, false);
                }
                else
                {
                    mySwitch = (osg::Switch *)switchGroup;
                    if (mySwitch->getNumChildren() > 0)
                    {
                        osg::Node *child = mySwitch->getChild(0);
                        if (child)
                        {

                            if (type == PluginMessageTypes::SGBrowserShowNode)
                                mySwitch->setChildValue(child, true);
                            else
                                mySwitch->setChildValue(child, false);
                        }
                    }
                }
            }
        }
    }
}
void TexVisitor::sendImageList()
{
    int max = getMax();
    if (max < 0)
    {
        return;
    }
    else
    {
        for (int index = 0; index < (max + 1); index++)
        {
            osg::Image *image = getImage(index);
            if (image)
            {
                texTab->setTexture(image->t(),
                                   image->s(),
                                   image->getPixelSizeInBits(),
                                   index,
                                   image->getImageSizeInBytes(),
                                   reinterpret_cast<char *>(image->data()));
            }
        }
    }
}
osg::Image *TexVisitor::getImage(int index)
{
    std::vector<LItem>::iterator iter;
    for (iter = imageListe.begin(); iter < imageListe.end(); iter++)
    {
        if ((*iter).index == index)
        {
            return (*iter).image;
        }
    }
    return NULL;
}
int TexVisitor::getMax()
{
    int max = -1;
    std::vector<LItem>::iterator iter;
    for (iter = imageListe.begin(); iter < imageListe.end(); iter++)
    {
        if ((*iter).index > max)
        {
            max = (*iter).index;
        }
    }
    return max;
}
LItem *TexVisitor::getLItem(osg::Image *image)
{
    std::vector<LItem>::iterator iter;
    for (iter = imageListe.begin(); iter < imageListe.end(); iter++)
    {
        if ((*iter).image == image)
        {
            return &(*iter);
        }
    }
    return NULL;
}
int TexVisitor::findImage(osg::Image *image)
{

    std::vector<LItem>::iterator iter;
    for (iter = imageListe.begin(); iter < imageListe.end(); iter++)
    {
        if ((*iter).image == image)
        {
            return (*iter).index;
        }
    }
    return -1;
}
TexVisitor::TexVisitor(TraversalMode tm, coTUISGBrowserTab *textureTab)
    : NodeVisitor(tm)
{
    texTab = textureTab;
    texFound = false;
}

TexVisitor::~TexVisitor()
{
    clearImageList();
}

void TexVisitor::processStateSet(osg::StateSet *ss)
{
    if (ss)
    {
        for (int textureNumber = 0; textureNumber < 21; textureNumber++)
        {
            StateAttribute *stateAttrib = ss->getTextureAttribute(textureNumber, StateAttribute::TEXTURE);
            if (stateAttrib)
            {
                Texture2D *texture = dynamic_cast<Texture2D *>(stateAttrib);
                if (texture)
                {
                    Image *image = texture->getImage();
                    if (image)
                    {
                        if (findImage(image) < 0)
                        {
                            LItem newItem;
                            newItem.image = image;
                            newItem.index = (getMax() + 1);
                            int index = newItem.index;
                            imageListe.push_back(newItem);
                            texTab->setTexture(image->t(),
                                               image->s(),
                                               image->getPixelSizeInBits(),
                                               index,
                                               image->getImageSizeInBytes(),
                                               reinterpret_cast<char *>(image->data()));
                            texFound = true;
                        }
                    }
                }
            }
        }
    }
}
void TexVisitor::apply(Node &node)
{

    StateSet *ss = NULL;

    ss = node.getStateSet();
    processStateSet(ss);

    Geode *geode = dynamic_cast<Geode *>(&node);
    if (geode)
    {
        for (unsigned int numdraw = 0; numdraw < geode->getNumDrawables(); numdraw++)
        {
            Drawable *drawable = geode->getDrawable(numdraw);
            if (drawable)
            {
                ss = drawable->getStateSet();
                processStateSet(ss);
            }
        }
    }

    traverse(node);
}

MyNodeVisitor::MyNodeVisitor(TraversalMode tm, coTUISGBrowserTab *sGBrowserTab)
    : NodeVisitor(tm)
{
    this->sGBrowserTab = sGBrowserTab;
    selectionManager = coVRSelectionManager::instance();

    selectNode = NULL;
    selectParentNode = NULL;
    mySelGroupNode = NULL;

    selNodeList.clear();
    findNodeList.clear();
    sendFindList.clear();

    ROOT = NULL;
}
void MyNodeVisitor::myInit()
{
    int currentMode = sGBrowserTab->getVisMode();
    if (currentMode == FIND_NODE)
    {
        findNodeList.clear();
        sendFindList.clear();
    }
    else if (currentMode == CURRENT_NODE)
    {
        sendCurrentList.clear();
    }
}
void MyNodeVisitor::addMyNode()
{
    int currentMode = sGBrowserTab->getVisMode();

    osg::Node *selParent = NULL;

    if (currentMode == HIDE_NODE || currentMode == SHOW_NODE)
    {

        TokenBuffer tb;
        tb << sGBrowserTab->getShowHidePath().c_str();
        tb << sGBrowserTab->getShowHideParentPath().c_str();

        if (currentMode == HIDE_NODE)
            cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserHideNode,
                tb.getData().length(), tb.getData().data());
        else
            cover->sendMessage(SGBrowser::plugin, coVRPluginSupport::TO_SAME, PluginMessageTypes::SGBrowserShowNode,
                tb.getData().length(), tb.getData().data());
    }
    else if (currentMode == CLEAR_SELECTION)
    {
        selectionManager->clearSelection();
        sGBrowserTab->setCurrentNode(NULL);
    }
    else if (currentMode == SET_SELECTION)
    {

        selectNode = selectionManager->validPath(sGBrowserTab->getSelectPath());
        selParent = selectionManager->validPath(sGBrowserTab->getSelectParentPath());

        if (selectNode)
        {
            sGBrowserTab->setCurrentNode(selectNode);
            selectionManager->addSelection(selParent->asGroup(), selectNode);
        }
    }
}

void MyNodeVisitor::updateMyParent(Node &node)
{
    int currentMode = sGBrowserTab->getVisMode();

    if (currentMode == UPDATE_NODES)
    {

        myUpdate(&node);
        osg::Group *my = node.asGroup();
        for (unsigned int i = 0; my && i < my->getNumChildren(); i++)
        {
            updateMyChild(my->getChild(i));
        }
    }
    else if (currentMode == UPDATE_EXPAND)
    {
        osg::Node *expandNode = selectionManager->validPath(sGBrowserTab->getExpandPath());

        if (expandNode)
        {
            myUpdate(expandNode);

            osg::Group *my = expandNode->asGroup();
            for (unsigned int i = 0; my && i < my->getNumChildren(); i++)
            {
                osg::Node *child = my->getChild(i);
                osg::Group *markGroup = child->asGroup();
                osg::Node *help = child;
                if (markGroup)
                {
                    while (selectionManager->isHelperNode(markGroup))
                    {
                        if (markGroup->getNumChildren())
                        {
                            help = markGroup->getChild(0);
                            if (help)
                            {
                                markGroup = help->asGroup();
                            }
                            else
                            {
                                break;
                            }
                        }
                        else
                        {
                            break;
                        }
                    }
                    updateMyChild(help);
                }
                else
                    updateMyChild(child);
            }
        }
    }
}

void MyNodeVisitor::updateMyChild(Node *node)
{
    myUpdate(node);

    osg::Group *my = node->asGroup();
    if (my && my->getNumChildren())
    {
        if (my->getChild(0))
        {
            osg::Node *child = my->getChild(0);
            osg::Group *markGroup = child->asGroup();
            osg::Node *help = child;
            if (markGroup)
            {
                while (selectionManager->isHelperNode(markGroup) && (markGroup->getNumChildren() > 0))
                {
                    help = markGroup->getChild(0);
                    if (help)
                    {
                        markGroup = help->asGroup();
                    }
                    else
                    {
                        break;
                    }
                }
                if(dynamic_cast<osg::Drawable *>(help) == NULL)
                    myUpdate(help);
            }
            else
            {
                if(dynamic_cast<osg::Drawable *>(child) == NULL)
                    myUpdate(child);
            }
        }
    }
}

void MyNodeVisitor::myUpdate(Node *node)
{

    int numberOfParents = node->getNumParents();
    std::string name = node->getName();
    int mode = 0;
    bool onOff;
    std::string path = "";
    std::string parentPath = "";

    if (!selectionManager->isHelperNode(node))
    {
        path = selectionManager->generatePath(node);

        if (name == "OBJECTS_ROOT")
            sGBrowserTab->sendType(SG_CLIP_NODE, node->className(), name.c_str(), path.c_str(), path.c_str(), 0);
        else
        {
            for (int j = 0; j < numberOfParents; j++)
            {
                std::string nodeType = node->className();
                std::string parentType = node->getParent(j)->className();

                if (parentType == "Switch")
                {
                    osg::Switch *parentSwitch = (osg::Switch *)node->getParent(j);
                    onOff = parentSwitch->getChildValue(node);
                    if (onOff)
                        mode = 1;
                    else
                        mode = 2;
                }
                else
                    mode = 0;

                osg::Node *parentNode = node->getParent(j);
                while (selectionManager->isHelperNode(parentNode))
                {
                    parentNode = parentNode->getParent(0);
                }
                parentPath = selectionManager->generatePath(parentNode);

                if (nodeType == "Node")
                    sGBrowserTab->sendType(SG_NODE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "Geode")
                    sGBrowserTab->sendType(SG_GEODE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "Group")
                    sGBrowserTab->sendType(SG_GROUP, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "ClipNode")
                    sGBrowserTab->sendType(SG_CLIP_NODE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "ClearNode")
                    sGBrowserTab->sendType(SG_CLEAR_NODE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "LightSource")
                    sGBrowserTab->sendType(SG_LIGHT_SOURCE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "TexGenNode")
                    sGBrowserTab->sendType(SG_TEX_GEN_NODE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "Transform")
                    sGBrowserTab->sendType(SG_TRANSFORM, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "MatrixTransform")
                    sGBrowserTab->sendType(SG_MATRIX_TRANSFORM, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "AutoTransform")
                    sGBrowserTab->sendType(SG_AUTO_TRANSFORM, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else if (nodeType == "Switch")
                    sGBrowserTab->sendType(SG_SWITCH, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode, node->asGroup()->getNumChildren());
                else if (nodeType == "LOD")
                    sGBrowserTab->sendType(SG_LOD, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
                else
                    sGBrowserTab->sendType(SG_NODE, node->className(), name.c_str(), path.c_str(), parentPath.c_str(), mode);
            }
        }
    }
}

void MyNodeVisitor::apply(Node &node)
{

    int currentMode = sGBrowserTab->getVisMode();

    if (currentMode == UPDATE_NODES)
    {
        myUpdate(&node);
    }
    else if (currentMode == FIND_NODE)
    {
        std::string searchName = sGBrowserTab->getFindName();
        if (strstr(node.getName().c_str(), searchName.c_str()))
        {
            findNodeList.push_back(&node);
        }
    }

    traverse(node);
}

void MyNodeVisitor::traverseFindList()
{
    int currentMode = sGBrowserTab->getVisMode();

    if (currentMode == FIND_NODE)
    {
        while (!findNodeList.empty())
        {
            traverseMyParents(findNodeList.front());
            findNodeList.pop_front();
        }
    }
    else if (currentMode == CURRENT_NODE)
    {
        Node *currentNode = sGBrowserTab->getCurrentNode();
        traverseMyParents(currentNode);
    }
}
void MyNodeVisitor::traverseMyParents(Node *node)
{
    int currentMode = sGBrowserTab->getVisMode();

    if (currentMode == FIND_NODE)
    {
        sendFindList.push_front(node);

        for (unsigned int i = 0; i < node->getNumParents(); i++)
        {
            traverseMyParents(node->getParent(i));
        }
    }
    else if (currentMode == CURRENT_NODE)
    {
        sendCurrentList.push_front(node);

        for (unsigned int i = 0; i < node->getNumParents(); i++)
        {
            traverseMyParents(node->getParent(i));
        }
    }
}
void MyNodeVisitor::sendMyFindList()
{
    int currentMode = sGBrowserTab->getVisMode();

    if (currentMode == FIND_NODE)
    {
        while (!sendFindList.empty())
        {
            myUpdate(sendFindList.front());
            sendFindList.pop_front();
        }
    }
    else if (currentMode == CURRENT_NODE)
    {
        while (!sendCurrentList.empty())
        {
            myUpdate(sendCurrentList.front());
            sendCurrentList.pop_front();
        }
    }
}

COVERPLUGIN(SGBrowser)
