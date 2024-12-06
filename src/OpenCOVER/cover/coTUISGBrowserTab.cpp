#include "coTabletUI.h"
#include "coVRPluginSupport.h"

#include <util/coTabletUIMessages.h>
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

namespace opencover {

using covise::TokenBuffer;
using covise::Message;
using covise::Connection;
using covise::ClientConnection;
using covise::ServerConnection;


class SGTextureThread : public OpenThreads::Thread
{

public:
    SGTextureThread(coTUISGBrowserTab *tab)
        : OpenThreads::Thread()
    {
        this->tab = tab;
        running = true;
        textureListCount = 0;
        sendedTextures = 0;
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
    coTUISGBrowserTab *tab;
    int type;
    int textureListCount;
    int sendedTextures;
    bool running;
    bool finishedTraversing;
    bool finishedNode;
    bool noTextures;
};

coTUISGBrowserTab::coTUISGBrowserTab(coTabletUI *tui, const char *n, int pID)
    : coTUIElement(tui, n, pID, TABLET_BROWSER_TAB)
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

coTUISGBrowserTab::~coTUISGBrowserTab()
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

void coTUISGBrowserTab::noTexture()
{
    thread->noTexturesFound(true);
}

void coTUISGBrowserTab::finishedNode()
{
    thread->nodeFinished(true);
}

void coTUISGBrowserTab::sendNodeTextures()
{
    std::cerr << "coTUISGBrowserTab::sendNodeTextures" << std::endl;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_NODE_TEXTURES;
    t << ID;
    this->send(t);
}
//gottlieb<
void coTUISGBrowserTab::loadFilesFlag(bool state)
{
    loadFile = state;
    if (loadFile)
        setVal(TABLET_LOADFILE_SATE, 1);
    else
        setVal(TABLET_LOADFILE_SATE, 0);
}

void coTUISGBrowserTab::hideSimNode(bool state, const char *nodePath, const char *simPath)
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
void coTUISGBrowserTab::setSimPair(const char *nodePath, const char *simPath, const char *simName)
{
    setVal(TABLET_SIM_SETSIMPAIR, nodePath, simPath, simName);
}

size_t coTUISGBrowserTab::getDataLength() const
{
    return data.size();
}

char *coTUISGBrowserTab::getData()
{
    return data.data();
}

Connection *coTUISGBrowserTab::getConnection()
{
    if (tui()->connectedHost)
        return tui()->sgConn.get();

    return nullptr;
}

//>gottlieb
void coTUISGBrowserTab::sendNoTextures()
{
    std::cerr << "coTUISGBrowserTab::sendNoTextures" << std::endl;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_NO_TEXTURES;
    t << ID;
    this->send(t);
}

void coTUISGBrowserTab::finishedTraversing()
{
    thread->traversingFinished(true);
}

void coTUISGBrowserTab::incTextureListCount()
{
    thread->incTextureListCount();
}

void coTUISGBrowserTab::sendTraversedTextures()
{
    std::cerr << "coTUISGBrowserTab::sendTraversedTextures" << std::endl;
    TokenBuffer t;
    t << TABLET_SET_VALUE;
    t << TABLET_TRAVERSED_TEXTURES;
    t << ID;
    this->send(t);
}

void coTUISGBrowserTab::send(TokenBuffer &tb)
{
    if (!getConnection())
    {
        std::cerr << "coTUISGBrowserTab::send: not connected" << std::endl;
        return;
    }
    Message m(tb);
    m.type = covise::COVISE_MESSAGE_TABLET_UI;
    getConnection()->sendMessage(&m);
}

void coTUISGBrowserTab::parseTextureMessage(TokenBuffer &tb)
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

        //changedNode = (osg::Node *)(uintptr_t)node;
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

void coTUISGBrowserTab::sendTexture()
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

void coTUISGBrowserTab::setTexture(int height, int width, int depth, int texIndex, int dataLength, const char *data)
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

void coTUISGBrowserTab::setTexture(int texNumber, int mode, int texGenMode, int texIndex)
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

void coTUISGBrowserTab::sendType(int type, const char *nodeType, const char *name, const char *path, const char *pPath, int mode, int numChildren)
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

void coTUISGBrowserTab::sendEnd()
{
    if (!tui()->isConnected())
        return;
    TokenBuffer tb;
    tb << TABLET_SET_VALUE;
    tb << TABLET_BROWSER_END;
    tb << ID;
    tui()->send(tb);
}

void coTUISGBrowserTab::sendProperties(std::string path, std::string pPath, int mode, int transparent)
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

void coTUISGBrowserTab::sendProperties(std::string path, std::string pPath, int mode, int transparent, float mat[])
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

void coTUISGBrowserTab::sendCurrentNode(osg::Node *node, std::string path)
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

void coTUISGBrowserTab::sendRemoveNode(std::string path, std::string parentPath)
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

void coTUISGBrowserTab::sendShader(std::string name)
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

void coTUISGBrowserTab::sendUniform(std::string name, std::string type, std::string value, std::string min, std::string max, std::string textureFile)
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

void coTUISGBrowserTab::sendShaderSource(std::string vertex, std::string fragment, std::string geometry, std::string tessControl, std::string tessEval)
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

void coTUISGBrowserTab::updateUniform(std::string shader, std::string name, std::string value, std::string textureFile)
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

void coTUISGBrowserTab::updateShaderSourceV(std::string shader, std::string vertex)
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

void coTUISGBrowserTab::updateShaderSourceTC(std::string shader, std::string tessControl)
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

void coTUISGBrowserTab::updateShaderSourceTE(std::string shader, std::string tessEval)
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

void coTUISGBrowserTab::updateShaderSourceF(std::string shader, std::string fragment)
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

void coTUISGBrowserTab::updateShaderSourceG(std::string shader, std::string geometry)
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

void coTUISGBrowserTab::updateShaderNumVertices(std::string shader, int value)
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

void coTUISGBrowserTab::updateShaderOutputType(std::string shader, int value)
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
void coTUISGBrowserTab::updateShaderInputType(std::string shader, int value)
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

std::vector<std::string> coTUISGBrowserTab::parsePathString(std::string path)
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

osg::Node* coTUISGBrowserTab::getNode(std::string path)
{
    osg::ref_ptr<osg::Node> objectsRoot_node = cover->getObjectsRoot()->asNode();
    osg::ref_ptr<osg::Node> vrmlRoot_node = objectsRoot_node->asGroup()->getChild(1);

    // osg::ref_ptr<osg::Node> vrmlRoot_node = cover->getScene()->asNode();
    // osg::ref_ptr<osg::Node> objectsRoot_node = cover->getScene()->asNode();
    // osg::ref_ptr<osg::Node> vrmlRoot_node = objectsRoot_node->asGroup()->getChild(0);

    // osg::ref_ptr<osg::Node> vrmlRoot_node = cover->getScene()->asNode();

    std::vector<std::string> path_str_vector = coTUISGBrowserTab::parsePathString(path);

    osg::ref_ptr<osg::Node> path_cursorNode = nullptr;

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

void coTUISGBrowserTab::addNode(const char* nodePath, int nodeType)
{
    std::string nodePath_str = std::string(nodePath);
    osg::ref_ptr<osg::Node> currentNode = coTUISGBrowserTab::getNode(nodePath_str)->asGroup()->getChild(0);

    switch (nodeType)
    {
        case SG_GROUP:
        {
            currentNode->asGroup()->addChild(new osg::Group);
            
            std::cout << "Added new GROUP to: " << nodePath_str << std::endl;

            break;
        }

        case SG_MATRIX_TRANSFORM:
        {
            currentNode->asGroup()->addChild(new osg::MatrixTransform);

            std::cout << "Added new MATRIX TRANSFORM to: " << nodePath_str << std::endl;

            break;
        }

        default:
        {
            break;
        }
    }
}

void coTUISGBrowserTab::removeNode(const char* nodePath, const char* parent_nodePath)
{
    std::string nodePath_str = std::string(nodePath);
    std::string parentPath_str = std::string(parent_nodePath);

    osg::ref_ptr<osg::Node> node_remove = coTUISGBrowserTab::getNode(nodePath_str);
    osg::ref_ptr<osg::Node> parentNode = coTUISGBrowserTab::getNode(parentPath_str);

    cover->removeNode(node_remove);

    /*
    if (parentNode->asGroup()->containsNode(node_remove))
    {
        parentNode->asGroup()->removeChild(node_remove);
        // cover->removeNode(node_remove);
    }
    */
}

void coTUISGBrowserTab::moveNode(const char* nodePath, const char* oldParent_nodePath, const char* newParent_nodePath, int dropIndex)
{
    std::string oldPath_str = std::string(nodePath);
    std::string oldParent_str = std::string(oldParent_nodePath);
    std::string newParent_str = std::string(newParent_nodePath);

    osg::ref_ptr<osg::Node> oldPath_node = coTUISGBrowserTab::getNode(oldPath_str);
    osg::ref_ptr<osg::Node> oldParent_node = coTUISGBrowserTab::getNode(oldParent_str);
    osg::ref_ptr<osg::Node> newParent_node = coTUISGBrowserTab::getNode(newParent_str);

    if (oldParent_node->asGroup()->containsNode(oldPath_node))
    {
        oldParent_node->asGroup()->removeChild(oldPath_node);

        if ((dropIndex <= -1) || (dropIndex >= newParent_node->asGroup()->getNumChildren()))
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

void coTUISGBrowserTab::renameNode(const char* nodePath, const char* nodeNewName)
{
    std::string nodePath_str = std::string(nodePath);
    // std::string nodeNewName_str = std::string(nodeNewName);

    std::cout << nodePath_str << std::endl;

    osg::ref_ptr<osg::Node> path_node = coTUISGBrowserTab::getNode(nodePath_str);
    osg::ref_ptr<osg::Node> outlinedObject_node = path_node->asGroup()->getChild(0);

    // osg::ref_ptr<osg::Group> parent_group = path_node->getParent(0);
    
    // path_node = parent_group->getChild(parent_group->getChildIndex(path_node));

    outlinedObject_node->setName(nodeNewName);
}

void coTUISGBrowserTab::parseMessage(TokenBuffer &tb)
{
    int i;
    tb >> i;

    switch (i)
    {
        case TABLET_INIT_SECOND_CONNECTION:
        {
            std::cerr << "coTUISGBrowserTab: recreating 2nd connection" << std::endl;

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
            osg::Node* path_node = coTUISGBrowserTab::getNode(path_str);

            coTUISGBrowserTab::sendCurrentNode(path_node, path_str);

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

            coTUISGBrowserTab::addNode(path, node_type);

            break;
        }

        case TABLET_BROWSER_REMOVED_NODE:
        {
            // std::cout << "TABLET_BROWSER_REMOVE_NODE" << std::endl;
            const char* path;
            const char* parentPath;

            tb >> path;
            tb >> parentPath;

            coTUISGBrowserTab::removeNode(path, parentPath);

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

            coTUISGBrowserTab::moveNode(old_path, oldParent_path, newParent_path, drop_index);

            break;
        }

        case TABLET_BROWSER_RENAMED_NODE:
        {
            const char* path;
            const char* newName;

            tb >> path;
            tb >> newName;

            coTUISGBrowserTab::renameNode(path, newName);
        }

        // break;
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

void coTUISGBrowserTab::resend(bool create)
{
    std::cerr << "coTUISGBrowserTab::resend(create=" << create << ")" << std::endl;

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

    coTUIElement::resend(create);

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
            sendedTextures++;

            //cout << " sended textures : " << sendedTextures << "\n";
            if (finishedTraversing && textureListCount == sendedTextures)
            {
                //cout << " finished sending with: " << textureListCount << "  textures \n";
                tab->sendTraversedTextures();
                //type = THREAD_NOTHING_TO_DO;
                finishedTraversing = false;
                textureListCount = 0;
                sendedTextures = 0;
            }
            if (finishedNode && textureListCount == sendedTextures)
            {
                //cout << " finished sending with: " << textureListCount << "  textures \n";
                tab->sendNodeTextures();

                finishedNode = false;
                textureListCount = 0;
                sendedTextures = 0;
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
