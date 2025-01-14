#include "vvTabletUI.h"
#include "vvPluginSupport.h"

#include <util/vvTabletUIMessages.h>
#include <util/threadname.h>
#include <net/tokenbuffer.h>
#include <net/covise_host.h>
#include <net/covise_connect.h>
#include <net/message.h>
#include <net/message_types.h>
#include <iostream>
#ifndef WIN32
#include <sys/socket.h>
#endif
#include <util/unixcompat.h>

namespace vive {

using covise::TokenBuffer;
using covise::Message;
using covise::Connection;
using covise::ClientConnection;
using covise::ServerConnection;


class SGTextureThread : public OpenThreads::Thread
{

public:
    SGTextureThread(vvTUISGBrowserTab *tab)
        : OpenThreads::Thread()
    {
        this->tab = tab;
        running = true;
        textureListCount = 0;
        sentTextures = 0;
    }
    virtual void run();
    void setType(int type)
    {
        this->type = type;
    }
    void terminateTextureThread()
    {
        running = false;
    }
    void incTextureListCount()
    {
        textureListCount++;
    }
    void traversingFinished(bool state)
    {
        finishedTraversing = state;
    }
    void nodeFinished(bool state)
    {
        finishedNode = state;
    }
    void noTexturesFound(bool state)
    {
        noTextures = state;
    }
    void msleep(int msec);

private:
    vvTUISGBrowserTab *tab;
    int type=0;
    int textureListCount=0;
    int sentTextures=0;
    bool running=false;
    bool finishedTraversing=false;
    bool finishedNode=false;
    bool noTextures=true;
};

vvTUISGBrowserTab::vvTUISGBrowserTab(vvTabletUI *tui, const char *n, int pID)
    : vvTUIElement(tui, n, pID, TABLET_BROWSER_TAB)
{
    currentNode = 0;
    changedNode = 0;
    texturesToChange = 0;

    texturePort = 0;
    // next port is for Texture communication

    currentPath = "";
    loadFile = false; //gottlieb

    thread = new SGTextureThread(this);
    thread->setType(THREAD_NOTHING_TO_DO);
    thread->traversingFinished(false);
    thread->nodeFinished(false);
    thread->noTexturesFound(false);
    thread->start();
}

vvTUISGBrowserTab::~vvTUISGBrowserTab()
{
    if (thread)
    {
        thread->terminateTextureThread();

        while (thread->isRunning())
        {
            usleep(10000);
        }
        delete thread;
    }
}

void vvTUISGBrowserTab::noTexture()
{
    thread->noTexturesFound(true);
}

void vvTUISGBrowserTab::finishedNode()
{
    thread->nodeFinished(true);
}

void vvTUISGBrowserTab::sendNodeTextures()
{
    std::cerr << "vvTUISGBrowserTab::sendNodeTextures" << std::endl;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_NODE_TEXTURES;
    t << ID;
    this->send(t);
}
//gottlieb<
void vvTUISGBrowserTab::loadFilesFlag(bool state)
{
    loadFile = state;
    if (loadFile)
        setVal(TABLET_LOADFILE_SATE, 1);
    else
        setVal(TABLET_LOADFILE_SATE, 0);
}

void vvTUISGBrowserTab::hideSimNode(bool state, const char *nodePath, const char *simPath)
{
    if (state)
    {
        setVal(TABLET_SIM_SHOW_HIDE, 0, nodePath, simPath);
    }
    else
    {
        setVal(TABLET_SIM_SHOW_HIDE, 1, nodePath, simPath);
    }
}
void vvTUISGBrowserTab::setSimPair(const char *nodePath, const char *simPath, const char *simName)
{
    setVal(TABLET_SIM_SETSIMPAIR, nodePath, simPath, simName);
}

size_t vvTUISGBrowserTab::getDataLength() const
{
    return data.size();
}

char *vvTUISGBrowserTab::getData()
{
    return data.data();
}

Connection *vvTUISGBrowserTab::getConnection()
{
    if (tui()->connectedHost)
        return tui()->sgConn.get();

    return nullptr;
}

//>gottlieb
void vvTUISGBrowserTab::sendNoTextures()
{
    std::cerr << "vvTUISGBrowserTab::sendNoTextures" << std::endl;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_NO_TEXTURES;
    t << ID;
    this->send(t);
}

void vvTUISGBrowserTab::finishedTraversing()
{
    thread->traversingFinished(true);
}

void vvTUISGBrowserTab::incTextureListCount()
{
    thread->incTextureListCount();
}

void vvTUISGBrowserTab::sendTraversedTextures()
{
    std::cerr << "vvTUISGBrowserTab::sendTraversedTextures" << std::endl;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_TRAVERSED_TEXTURES;
    t << ID;
    this->send(t);
}

void vvTUISGBrowserTab::send(TokenBuffer &tb)
{
    if (!getConnection())
    {
        std::cerr << "vvTUISGBrowserTab::send: not connected" << std::endl;
        return;
    }
    Message m(tb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    getConnection()->sendMessage(&m);
}

void vvTUISGBrowserTab::parseTextureMessage(TokenBuffer &tb)
{
    int type;
    tb >> type;
    if (type == TABLET_TEX_CHANGE)
    {
        const char *path = nullptr;
        int dataLength = 0;
        tb >> textureNumber;
        tb >> textureMode;
        tb >> textureTexGenMode;
        tb >> alpha;
        tb >> height;
        tb >> width;
        tb >> depth;
        tb >> dataLength;
        tb >> path;
        tb >> index;

        //changedNode = (vsg::Node *)(uintptr_t)node;
        changedPath = std::string(path);
        if (currentNode)
        {
            changedNode = currentNode;
            data.resize(dataLength);

            for (int k = 0; k < dataLength; k++)
            {
                if ((k % 4) == 3)
                    tb >> data[k];
                else if ((k % 4) == 2)
                    tb >> data[k - 2];
                else if ((k % 4) == 1)
                    tb >> data[k];
                else if ((k % 4) == 0)
                    tb >> data[k + 2];
            }
            if (listener)
            {
                listener->tabletEvent(this);
            }
        }
    }
}

void vvTUISGBrowserTab::sendTexture()
{
    mutex.lock();
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX;
    tb << ID;
    tb << _heightList.front();
    tb << _widthList.front();
    tb << _depthList.front();
    tb << _indexList.front();
    tb << _lengthList.front();

    int length = _heightList.front() * _widthList.front() * (_depthList.front() / 8);
    tb.addBinary(_dataList.front(), length);
    this->send(tb);
    _heightList.pop();
    _widthList.pop();
    _depthList.pop();
    _indexList.pop();
    _lengthList.pop();
    _dataList.pop();
    mutex.unlock();
}

void vvTUISGBrowserTab::setTexture(int height, int width, int depth, int texIndex, int dataLength, const char *data)
{
    mutex.lock();
    _heightList.push(height);
    _widthList.push(width);
    _depthList.push(depth);
    _indexList.push(texIndex);
    _lengthList.push(dataLength);
    _dataList.push(data);
    thread->incTextureListCount();
    //cout << " added texture : \n";
    mutex.unlock();
}

void vvTUISGBrowserTab::setTexture(int texNumber, int mode, int texGenMode, int texIndex)
{
    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_TEX_MODE;
    tb << ID;
    tb << texNumber;
    tb << mode;
    tb << texGenMode;
    tb << texIndex;

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendType(int type, const char *nodeType, const char *name, const char *path, const char *pPath, int mode, int numChildren)
{

    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_NODE;
    tb << ID;
    tb << type;
    tb << name;
    tb << nodeType;
    tb << mode;
    tb << path;
    tb << pPath;
    tb << numChildren;

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendEnd()
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_END;
    tb << ID;
    tui()->send(tb);
}

void vvTUISGBrowserTab::sendProperties(std::string path, std::string pPath, int mode, int transparent)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    int i;
    const char *currentPath = path.c_str();
    const char *currentpPath = pPath.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << ID;
    tb << currentPath;
    tb << currentpPath;
    tb << mode;
    tb << transparent;
    for (i = 0; i < 4; i++)
        tb << diffuse[i];
    for (i = 0; i < 4; i++)
        tb << specular[i];
    for (i = 0; i < 4; i++)
        tb << ambient[i];
    for (i = 0; i < 4; i++)
        tb << emissive[i];

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendProperties(std::string path, std::string pPath, int mode, int transparent, float mat[])
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    int i;
    const char *currentPath = path.c_str();
    const char *currentpPath = pPath.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_PROPERTIES;
    tb << ID;
    tb << currentPath;
    tb << currentpPath;
    tb << mode;
    tb << transparent;
    for (i = 0; i < 4; i++)
        tb << diffuse[i];
    for (i = 0; i < 4; i++)
        tb << specular[i];
    for (i = 0; i < 4; i++)
        tb << ambient[i];
    for (i = 0; i < 4; i++)
        tb << emissive[i];

    for (i = 0; i < 16; ++i)
    {
        tb << mat[i];
    }

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendCurrentNode(vsg::Node *node, std::string path)
{
    currentNode = node;
    visitorMode = CURRENT_NODE;

    if (listener)
        listener->tabletCurrentEvent(this);

    if (!tui()->isConnected())
        return;

    TokenBuffer tb;
    const char *currentPath = path.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_CURRENT_NODE;
    tb << ID;
    tb << currentPath;

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendRemoveNode(std::string path, std::string parentPath)
{

    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    const char *removePath = path.c_str();
    const char *removePPath = parentPath.c_str();
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_REMOVE_NODE;
    tb << ID;
    tb << removePath;
    tb << removePPath;

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendShader(std::string name)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    const char *shaderName = name.c_str();

    tb << TABLET_SET_VALUE;
    tb << GET_SHADER;
    tb << ID;
    tb << shaderName;

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendUniform(std::string name, std::string type, std::string value, std::string min, std::string max, std::string textureFile)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << GET_UNIFORMS;
    tb << ID;
    tb << name.c_str();
    tb << type.c_str();
    tb << value.c_str();
    tb << min.c_str();
    tb << max.c_str();
    tb << textureFile.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::sendShaderSource(std::string vertex, std::string fragment, std::string geometry, std::string tessControl, std::string tessEval)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << GET_SOURCE;
    tb << ID;
    tb << vertex.c_str();
    tb << fragment.c_str();
    tb << geometry.c_str();
    tb << tessControl.c_str();
    tb << tessEval.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateUniform(std::string shader, std::string name, std::string value, std::string textureFile)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_UNIFORM;
    tb << ID;
    tb << shader.c_str();
    tb << name.c_str();
    tb << value.c_str();
    tb << textureFile.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderSourceV(std::string shader, std::string vertex)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_VERTEX;
    tb << ID;
    tb << shader.c_str();
    tb << vertex.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderSourceTC(std::string shader, std::string tessControl)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_TESSCONTROL;
    tb << ID;
    tb << shader.c_str();
    tb << tessControl.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderSourceTE(std::string shader, std::string tessEval)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_TESSEVAL;
    tb << ID;
    tb << shader.c_str();
    tb << tessEval.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderSourceF(std::string shader, std::string fragment)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_FRAGMENT;
    tb << ID;
    tb << shader.c_str();
    tb << fragment.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderSourceG(std::string shader, std::string geometry)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << UPDATE_GEOMETRY;
    tb << ID;
    tb << shader.c_str();
    tb << geometry.c_str();

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderNumVertices(std::string shader, int value)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << SET_NUM_VERT;
    tb << ID;
    tb << shader.c_str();
    tb << value;

    tui()->send(tb);
}

void vvTUISGBrowserTab::updateShaderOutputType(std::string shader, int value)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << SET_OUTPUT_TYPE;
    tb << ID;
    tb << shader.c_str();
    tb << value;

    tui()->send(tb);
}
void vvTUISGBrowserTab::updateShaderInputType(std::string shader, int value)
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << SET_INPUT_TYPE;
    tb << ID;
    tb << shader.c_str();
    tb << value;

    tui()->send(tb);
}

std::vector<std::string> vvTUISGBrowserTab::parsePathString(std::string path)
{
    std::vector<std::string> path_indices = { };

    size_t pos = 0;

    while ((pos = path.find(";")) != std::string::npos)
    {
        path_indices.push_back(path.substr(0, pos));
        path.erase(0, pos + 1);
    }

    path_indices.push_back(path);

    return path_indices;
}

vsg::Node* vvTUISGBrowserTab::getNode(std::string path)
{
    vsg::ref_ptr<vsg::Node> objectsRoot_node = vv->getObjectsRoot()->asNode();
    vsg::ref_ptr<vsg::Node> vrmlRoot_node = objectsRoot_node->asGroup()->getChild(1);

    // vsg::ref_ptr<vsg::Node> vrmlRoot_node = vv->getScene()->asNode();
    // vsg::ref_ptr<vsg::Node> objectsRoot_node = vv->getScene()->asNode();
    // vsg::ref_ptr<vsg::Node> vrmlRoot_node = objectsRoot_node->asGroup()->getChild(0);

    // vsg::ref_ptr<vsg::Node> vrmlRoot_node = vv->getScene()->asNode();

    std::vector<std::string> path_str_vector = vvTUISGBrowserTab::parsePathString(path);

    vsg::ref_ptr<vsg::Node> path_cursorNode = nullptr;

    for (int i = 0; i < path_str_vector.size(); i++)
    {
        int path_index = -1;
        std::string path_str_index;

        try
        {
            path_index = std::stoi(path_str_vector[i]);
        }
        catch(const std::exception& /*e*/)
        {
            // std::cerr << e.what() << '\n';

            path_str_index = path_str_vector[i];
        }
        
        if (path_index != -1)
        {
            path_cursorNode = path_cursorNode->asGroup()->getChild(path_index);
        }
        else if (!path_str_index.empty())
        {
            if (path_str_index == "ROOT")
            {
                path_cursorNode = objectsRoot_node;
            }
            else if (path_str_index == "VRMLRoot")
            {
                path_cursorNode = vrmlRoot_node;
            }
        }
    }

    return path_cursorNode;
}

void vvTUISGBrowserTab::addNode(const char* nodePath, int nodeType)
{
    std::string nodePath_str = std::string(nodePath);
    vsg::ref_ptr<vsg::Node> currentNode = vvTUISGBrowserTab::getNode(nodePath_str)->asGroup()->getChild(0);

    switch (nodeType)
    {
        case SG_GROUP:
        {
            currentNode->asGroup()->addChild(new vsg::Group);
            
            std::cout << "Added new GROUP to: " << nodePath_str << std::endl;

            break;
        }

        case SG_MATRIX_TRANSFORM:
        {
            currentNode->asGroup()->addChild(vsg::MatrixTransform::create());

            std::cout << "Added new MATRIX TRANSFORM to: " << nodePath_str << std::endl;

            break;
        }

        default:
        {
            break;
        }
    }
}

void vvTUISGBrowserTab::removeNode(const char* nodePath, const char* parent_nodePath)
{
    std::string nodePath_str = std::string(nodePath);
    std::string parentPath_str = std::string(parent_nodePath);

    vsg::ref_ptr<vsg::Node> node_remove = vvTUISGBrowserTab::getNode(nodePath_str);
    vsg::ref_ptr<vsg::Node> parentNode = vvTUISGBrowserTab::getNode(parentPath_str);

    vv->removeNode(node_remove);

    /*
    if (parentNode->asGroup()->containsNode(node_remove))
    {
        parentNode->asGroup()->removeChild(node_remove);
        // vv->removeNode(node_remove);
    }
    */
}

void vvTUISGBrowserTab::moveNode(const char* nodePath, const char* oldParent_nodePath, const char* newParent_nodePath, unsigned int dropIndex)
{
    std::string oldPath_str = std::string(nodePath);
    std::string oldParent_str = std::string(oldParent_nodePath);
    std::string newParent_str = std::string(newParent_nodePath);

    vsg::ref_ptr<vsg::Node> oldPath_node = vvTUISGBrowserTab::getNode(oldPath_str);
    vsg::ref_ptr<vsg::Node> oldParent_node = vvTUISGBrowserTab::getNode(oldParent_str);
    vsg::ref_ptr<vsg::Node> newParent_node = vvTUISGBrowserTab::getNode(newParent_str);

    if (oldParent_node->asGroup()->containsNode(oldPath_node))
    {
        oldParent_node->asGroup()->removeChild(oldPath_node);

        if ((dropIndex <= -1) || (dropIndex >= newParent_node->asGroup()->children.size()))
        {
            // oldParent_node->asGroup()->removeChild(oldPath_node);
            newParent_node->asGroup()->addChild(oldPath_node);
        }
        else
        {
            // oldParent_node->asGroup()->removeChild(oldPath_node);
            newParent_node->asGroup()->insertChild(dropIndex, oldPath_node);
        }
    }
}

void vvTUISGBrowserTab::renameNode(const char* nodePath, const char* nodeNewName)
{
    std::string nodePath_str = std::string(nodePath);
    // std::string nodeNewName_str = std::string(nodeNewName);

    std::cout << nodePath_str << std::endl;

    vsg::ref_ptr<vsg::Node> path_node = vvTUISGBrowserTab::getNode(nodePath_str);
    vsg::ref_ptr<vsg::Node> outlinedObject_node = path_node->asGroup()->getChild(0);

    // vsg::ref_ptr<vsg::Group> parent_group = path_node->getParent(0);
    
    // path_node = parent_group->getChild(parent_group->getChildIndex(path_node));

    outlinedObject_node->setName(nodeNewName);
}

void vvTUISGBrowserTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;

    switch (i)
    {
        case TABLET_INIT_SECOND_CONNECTION:
        {
            std::cerr << "vvTUISGBrowserTab: recreating 2nd connection" << std::endl;

            if (thread)
            {
                thread->terminateTextureThread();

                while (thread->isRunning())
                {
                    sleep(1);
                }
                delete thread;
                thread = NULL;
            }

            thread = new SGTextureThread(this);
            thread->setType(THREAD_NOTHING_TO_DO);
            thread->traversingFinished(false);
            thread->nodeFinished(false);
            thread->noTexturesFound(false);
            thread->start();

            break;
        }

        case TABLET_BROWSER_PROPERTIES:
        {
            if (listener)
            {
                listener->tabletDataEvent(this, tb);
            }

            break;
        }

        case TABLET_BROWSER_UPDATE:
        {
            visitorMode = UPDATE_NODES;
        
            if (listener)
                listener->tabletPressEvent(this);
            
            break;
        }

        case TABLET_BROWSER_EXPAND_UPDATE:
        {
            visitorMode = UPDATE_EXPAND;
            const char *path;
            tb >> path;
            expandPath = std::string(path);

            if (listener)
                listener->tabletPressEvent(this);

            break;
        }

        case TABLET_BROWSER_CURRENT_NODE:
        {
            const char* path;

            tb >> path;
            std::string path_str = std::string(path);
            vsg::Node* path_node = vvTUISGBrowserTab::getNode(path_str);

            vvTUISGBrowserTab::sendCurrentNode(path_node, path_str);

            break;
        }

        case TABLET_BROWSER_SELECTED_NODE:
        {
            visitorMode = SET_SELECTION;
            const char *path, *pPath;
            tb >> path;
            selectPath = std::string(path);
            tb >> pPath;
            selectParentPath = std::string(pPath);

            if (listener)
                listener->tabletSelectEvent(this);
        
            break;
        }
    
        case TABLET_BROWSER_CLEAR_SELECTION:
        {
            visitorMode = CLEAR_SELECTION;

            if (listener)
                listener->tabletSelectEvent(this);

            break;
        }
    
        case TABLET_BROWSER_SHOW_NODE:
        {
            visitorMode = SHOW_NODE;
            const char *path, *pPath;
            tb >> path;
            showhidePath = std::string(path);
            tb >> pPath;
            showhideParentPath = std::string(pPath);

            if (listener)
                listener->tabletSelectEvent(this);
        
            break;
        }
        
        case TABLET_BROWSER_HIDE_NODE:
        {
            visitorMode = HIDE_NODE;
            const char *path, *pPath;
            tb >> path;
            showhidePath = std::string(path);
            tb >> pPath;
            showhideParentPath = std::string(pPath);

            if (listener)
                listener->tabletSelectEvent(this);
        
            break;
        }

        case TABLET_BROWSER_COLOR:
        {
            visitorMode = UPDATE_COLOR;
            tb >> ColorR;
            tb >> ColorG;
            tb >> ColorB;

            if (listener)
                listener->tabletChangeModeEvent(this);
        
            break;
        }
    
        case TABLET_BROWSER_WIRE:
        {
            visitorMode = UPDATE_WIRE;
            tb >> polyMode;

            if (listener)
                listener->tabletChangeModeEvent(this);
        
            break;
        }
    
        case TABLET_BROWSER_SEL_ONOFF:
        {
            visitorMode = UPDATE_SEL;
            tb >> selOnOff;

            if (listener)
                listener->tabletChangeModeEvent(this);
        
            break;
        }
    
        case TABLET_BROWSER_FIND:
        {
            const char *fname;
            visitorMode = FIND_NODE;
            tb >> fname;
            findName = std::string(fname);

            if (listener)
                listener->tabletFindEvent(this);

            break;
        }

        case TABLET_BROWSER_LOAD_FILES:
        {
            const char *fname;
            tb >> fname;

            if (listener)
                listener->tabletLoadFilesEvent((char*)fname);

            break;
        }

        case TABLET_BROWSER_ADDED_NODE:
        {
            const char* path;
            int node_type;

            tb >> path;
            tb >> node_type;

            vvTUISGBrowserTab::addNode(path, node_type);

            break;
        }

        case TABLET_BROWSER_REMOVED_NODE:
        {
            // std::cout << "TABLET_BROWSER_REMOVE_NODE" << std::endl;
            const char* path;
            const char* parentPath;

            tb >> path;
            tb >> parentPath;

            vvTUISGBrowserTab::removeNode(path, parentPath);

            break;
        }

        case TABLET_BROWSER_MOVED_NODE:
        {
            const char* old_path;
            const char* oldParent_path;
            const char* newParent_path;
            int drop_index;

            tb >> old_path;
            tb >> oldParent_path;
            tb >> newParent_path;
            tb >> drop_index;

            vvTUISGBrowserTab::moveNode(old_path, oldParent_path, newParent_path, drop_index);

            break;
        }

        case TABLET_BROWSER_RENAMED_NODE:
        {
            const char* path;
            const char* newName;

            tb >> path;
            tb >> newName;

            vvTUISGBrowserTab::renameNode(path, newName);
            break;
        }

        case TABLET_TEX_UPDATE:
        {
            thread->setType(TABLET_TEX_UPDATE);
            sendImageMode = SEND_IMAGES;

            if (listener)
            {
                listener->tabletEvent(this);
                listener->tabletReleaseEvent(this);
            }
            
            break;
        }

        case TABLET_TEX_CHANGE:
        {
            //cout << " currentNode : " << currentNode << "\n";
            if (currentNode)
            {
                // let the tabletui know that it can send texture data now
                TokenBuffer t;
                int buttonNumber;
                const char *path = currentPath.c_str();
                tb >> buttonNumber;
                t << TABLET_SET_VALUE;
                t << TABLET_TEX_CHANGE;
                t << ID;
                t << buttonNumber;
                t << path;
                tui()->send(t);
                //thread->setType(TABLET_TEX_CHANGE);
                texturesToChange++;
            }

            break;
        }
        
        default:
        {
            std::cerr << "unknown event " << i << std::endl;
        }
    }
}

void vvTUISGBrowserTab::resend(bool create)
{
    std::cerr << "vvTUISGBrowserTab::resend(create=" << create << ")" << std::endl;

#if 0
    if (thread)
    {
        thread->terminateTextureThread();

        while (thread->isRunning())
        {
            sleep(1);
        }
        delete thread;
    }

    thread = new SGTextureThread(this);
    thread->setType(THREAD_NOTHING_TO_DO);
    thread->traversingFinished(false);
    thread->nodeFinished(false);
    thread->noTexturesFound(false);
    thread->start();
#else
#if 0
    if (!thread)
    {
        thread = new SGTextureThread(this);
        thread->setType(THREAD_NOTHING_TO_DO);
        thread->traversingFinished(false);
        thread->nodeFinished(false);
        thread->noTexturesFound(false);
        thread->start();
    }
#endif
#endif

    vvTUIElement::resend(create);

    sendImageMode = SEND_LIST;
    if (listener)
    {
        listener->tabletEvent(this);
        listener->tabletReleaseEvent(this);
    }
    //gottlieb<
    if (loadFile)
        setVal(TABLET_LOADFILE_SATE, 1);
    else
        setVal(TABLET_LOADFILE_SATE, 0); //>gottlieb
}

//----------------------------------------------------------
//------n----------------------------------------------------

void SGTextureThread::msleep(int msec)
{
    usleep(msec * 1000);
}

void SGTextureThread::run()
{
    covise::setThreadName("cover:SG:tex");
    while (running)
    {
        bool work = false;
        if (!tab->queueIsEmpty())
        {
            tab->sendTexture();
            sentTextures++;

            //cout << " sended textures : " << sendedTextures << "\n";
            if (finishedTraversing && textureListCount == sentTextures)
            {
                //cout << " finished sending with: " << textureListCount << "  textures \n";
                tab->sendTraversedTextures();
                //type = THREAD_NOTHING_TO_DO;
                finishedTraversing = false;
                textureListCount = 0;
                sentTextures = 0;
            }
            if (finishedNode && textureListCount == sentTextures)
            {
                //cout << " finished sending with: " << textureListCount << "  textures \n";
                tab->sendNodeTextures();

                finishedNode = false;
                textureListCount = 0;
                sentTextures = 0;
            }

            work = true;
        }

        while (tab->getTexturesToChange()) //still Textures in queue
        {
            // waiting for incoming data
            if (!tab->getConnection())
                break;

            // message arrived
            if (!tab->getConnection()->check_for_input())
                break;

            Message m;
            if (tab->getConnection()->recv_msg(&m) == 0)
                break;

            work = true;

            if (m.type == covise::COVISE_MESSAGE_TABLET_UI)
            {
                TokenBuffer tokenbuffer(&m);
                int ID;
                tokenbuffer >> ID;
                tab->parseTextureMessage(tokenbuffer);
                tab->decTexturesToChange();
                //type = THREAD_NOTHING_TO_DO;
            }
        }

        if (!work)
            msleep(50);
    }
}

}
