/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVRDistributionManager.h"

#include <cover/coVRPluginList.h>
#include <cover/coVRPluginSupport.h>
#include "coVRParallelRenderPlugin.h"
#include <cover/coVRMSController.h>

#include <config/coConfig.h>
#include <config/coConfigConstants.h>
#include <cover/coVRConfig.h>
#include <cover/VRViewer.h>

#include <do/coDistributedObject.h>
#include <do/coDoPoints.h>
#include <do/coDoLines.h>
#include <do/coDoPolygons.h>
#include <do/coDoTriangleStrips.h>
#include <do/coDoGeometry.h>

#include <iostream>
#include <cassert>
#include <algorithm>
#include <sstream>

#include <net/tokenbuffer.h>
#include <net/message.h>

#define LOG_CERR(x)            \
    {                          \
        if (std::cerr.bad())   \
            std::cerr.clear(); \
        std::cerr << x;        \
    }

static inline std::list<std::string> splitHostlist(const std::string &hostlistString)
{
    std::list<std::string> hostlist;
    std::istringstream hl(hostlistString);
    do
    {
        std::string host;
        hl >> host;
        if (host != "")
            hostlist.push_back(host);
    } while (hl);
    return hostlist;
}

opencover::coVRDistributionManager::coVRDistributionManager()
    : active(false)
    , display(false)
{

    if (!coVRMSController::instance()->isCluster())
        return;

    // Some sanity checks before activating the manager
    if (covise::coConfig::getInstance()->getValue("COVER.Parallel.Type") == "Distributed")
    {
        covise::coConfigString method = covise::coConfig::getInstance()->getString("COVER.Parallel.Method");
        if (method.hasValidValue())
        {

            this->active = true;

            std::string algo = ((QString)covise::coConfig::getInstance()->getString("value", "COVER.Parallel.DistributionAlgorithm", "RoundRobin")).toStdString();

            if (DistributionAlgorithm::create(algo) == 0)
            {
                LOG_CERR("coVRDistributionManager::<init> err: unknown distribution algorithm " << algo << std::endl);
                this->active = false;
                return;
            }
            else
            {
                LOG_CERR("coVRDistributionManager::<init> info: using distribution " << algo << std::endl);
            }
        }
    }
}

opencover::coVRDistributionManager::~coVRDistributionManager()
{
}

bool opencover::coVRDistributionManager::isActive() const
{
    return this->active;
}

bool opencover::coVRDistributionManager::init()
{
    if (!this->active)
        return false;

    covise::coConfigString method = covise::coConfig::getInstance()->getString("COVER.Parallel.Method");
    if (method.hasValidValue())
    {

        opencover::coVRMSController *msControl = opencover::coVRMSController::instance();

        opencover::coVRParallelRenderPlugin *implementation = dynamic_cast<opencover::coVRParallelRenderPlugin *>(opencover::coVRPluginList::instance()->addPlugin(((QString)method).toLatin1().data()));

        if (implementation == 0)
        {
            this->active = false;
            return false;
        }

        // Get the unqique host ID for parallel rendering for this node
        std::string hostid = implementation->getHostID();

        this->self = RenderNode(hostid);

        if (this->self.isDisplay())
        {
            // Displays are parallel rendering masters
            if (!implementation->initialiseAsMaster())
            {
                this->active = false;
                return false;
            }
        }
        else
        {
            // all others are slaves
            if (!implementation->initialiseAsSlave())
            {
                this->active = false;
                return false;
            }
        }

        // Assign the slaves to the masters on head node
        if (msControl->isMaster())
        {

            std::string dNodes;
            std::string rNodes;
            RenderNode node;

            if (this->self.isDisplay())
            {
                this->displayNodes.push_back(this->self);
                dNodes += " " + std::string(this->self.getNodename());
            }
            else
            {
                this->renderNodes.push_back(this->self);
                rNodes += " " + std::string(this->self.getNodename());
            }

            int maxRenderNodeSize = 0;
            opencover::coVRMSController::SlaveData renderNodeSize(sizeof(int));
            msControl->readSlaves(&renderNodeSize);

            for (size_t ctr = 0; ctr < renderNodeSize.data.size(); ++ctr)
            {
                maxRenderNodeSize = std::max(maxRenderNodeSize, *reinterpret_cast<int *>(renderNodeSize.data[ctr]));
                assert(maxRenderNodeSize == *reinterpret_cast<int *>(renderNodeSize.data[ctr]));
            }

            LOG_CERR("coVRDistributionManager::init info: maxRenderNodeSize is " << maxRenderNodeSize << "b" << std::endl);

            // Get the render node information from all slaves
            opencover::coVRMSController::SlaveData renderNodeInfo(maxRenderNodeSize);
            msControl->readSlaves(&renderNodeInfo);

            for (size_t ctr = 0; ctr < renderNodeInfo.data.size(); ++ctr)
            {

                covise::TokenBuffer tb{covise:: DataHandle{static_cast<char*>(renderNodeInfo.data[ctr]), *reinterpret_cast<int*>(renderNodeSize.data[ctr]), false} };
                tb >> node;

                if (node.isDisplay())
                {
                    this->displayNodes.push_back(node);
                    dNodes += " " + std::string(node.getNodename());
                }
                else
                {
                    this->renderNodes.push_back(node);
                    rNodes += " " + std::string(node.getNodename());
                }
            }

            if (this->displayNodes.empty())
            {
                LOG_CERR("coVRDistributionManager::init err: no displays found in config" << std::endl);
                this->active = false;
                return false;
            }
            else
            {
                LOG_CERR("coVRDistributionManager::init info: displays =" << dNodes << std::endl);
            }

            if (renderNodes.empty())
            {
                LOG_CERR("coVRDistributionManager::init err: no render nodes defined in config" << std::endl);
                this->active = false;
                return false;
            }
            else
            {
                LOG_CERR("coVRDistributionManager::init info: render nodes =" << rNodes << std::endl);
            }

            // Distribute evenly at the beginning
            int ratio = this->renderNodes.size() / this->displayNodes.size();
            LOG_CERR("coVRDistributionManager::init info: ratio = " << ratio << std::endl);

            RenderGroup masterRenderGroup;

            // Create render groups (i.e display with assigned slaves)
            for (size_t displayCtr = 0; displayCtr < this->displayNodes.size(); ++displayCtr)
            {
                RenderGroup r;
                r.setMaster(this->displayNodes[displayCtr]);
                for (int slaveCtr = 0; slaveCtr < ratio; ++slaveCtr)
                {
                    r.addSlave(this->renderNodes[displayCtr * ratio + slaveCtr]);
                    if (this->self == renderNodes[displayCtr * ratio + slaveCtr])
                        masterRenderGroup = r;
                }
                this->renderGroups.push_back(r);

                if (this->self == this->displayNodes[displayCtr])
                    masterRenderGroup = r;
            }

            std::vector<covise::TokenBuffer> tb(msControl->getNumSlaves());
            int maxTbSize = 0;

            for (std::vector<RenderGroup>::iterator rg = this->renderGroups.begin();
                 rg != this->renderGroups.end(); ++rg)
            {
                const RenderNode &master = rg->getMaster();

                if (master.slaveIndex > 0)
                {
                    tb[master.getSlaveIndex() - 1] << rg->getHostlist();
                    maxTbSize = std::max(tb[master.getSlaveIndex() - 1].getData().length(), maxTbSize);
                    LOG_CERR(master.getSlaveIndex() << "m : " << rg->getHostlist() << std::endl);
                }

                const std::vector<RenderNode> &slaves = rg->getSlaves();

                for (std::vector<RenderNode>::const_iterator slave = slaves.begin(); slave != slaves.end(); ++slave)
                {
                    if (slave->getSlaveIndex() > 0)
                    {
                        tb[slave->getSlaveIndex() - 1] << rg->getHostlist();
                        tb[slave->getSlaveIndex() - 1] << master;
                        maxTbSize = std::max(tb[slave->getSlaveIndex() - 1].getData().length(), maxTbSize);
                        LOG_CERR(slave->getSlaveIndex() << "s : " << rg->getHostlist() << std::endl);
                    }
                }
            }

            opencover::coVRMSController::SlaveData d(maxTbSize);

            for (size_t ctr = 0; ctr < tb.size(); ++ctr)
            {
                LOG_CERR("coVRDistributionManager::init info: sending " << maxTbSize << "b to slave " << ctr << " (payload " << tb[ctr].getData().length() << "b)" << std::endl);
                memcpy(d.data[ctr], tb[ctr].getData().data(), tb[ctr].getData().length());
            }

            // Send group hostlist to slaves
            msControl->sendSlaves(&maxTbSize, sizeof(int));
            msControl->sendSlaves(d);

            LOG_CERR("coVRDistributionManager::init info: creating context with " << masterRenderGroup.getHostlist() << std::endl);
            implementation->createContext(splitHostlist(masterRenderGroup.getHostlist()), 0);
            LOG_CERR("coVRDistributionManager::init info: done creating master context" << std::endl);

            msControl->sendSlaves(&maxTbSize, sizeof(int)); // sync
        }
        else // Not the master
        {

            // Send render info to head node
            {
                covise::TokenBuffer tb;
                tb << this->self;
                int renderNodeSize = tb.getData().length();

                opencover::coVRMSController::instance()->sendMaster(&renderNodeSize, sizeof(int));
                LOG_CERR("coVRDistributionManager::init info: renderNodeSize is " << renderNodeSize << "b" << std::endl);

                opencover::coVRMSController::instance()->sendMaster(tb.getData().data(), renderNodeSize);
            }

            // Get render group information from head node
            int maxTbSize = 0;
            msControl->readMaster(&maxTbSize, sizeof(int));

            LOG_CERR("coVRDistributionManager::init info: reading " << maxTbSize << "b from master... -> ");

            char *tbContent = new char[maxTbSize];
            LOG_CERR(msControl->readMaster(tbContent, maxTbSize));

            LOG_CERR(" <- done" << std::endl);

            covise::TokenBuffer tb{ covise::DataHandle{tbContent, maxTbSize} };

            std::string hostlist;
            RenderNode master;

            tb >> hostlist;

            LOG_CERR("coVRDistributionManager::init info: creating context with "
                     << hostlist << std::endl);

            if (!this->self.isDisplay())
            {
                tb >> master;
                master.applyScreens();
            }

            implementation->createContext(splitHostlist(hostlist), 0);
            LOG_CERR("coVRDistributionManager::init info: done creating slave context" << std::endl);

            msControl->readMaster(&maxTbSize, sizeof(int)); //sync
            LOG_CERR(maxTbSize << std::endl);
        }
    }
    else
    {
        this->active = false;
        return false;
    }

    return true;
}

std::vector<int> opencover::coVRDistributionManager::assign(const covise::coDistributedObject *object)
{
    opencover::coVRMSController *msControl = opencover::coVRMSController::instance();
    std::vector<int> assignments;

    if (msControl->isMaster())
    {
        int numberOfGroups = this->renderGroups.size();
        int *assignmentsArray = new int[numberOfGroups];

        msControl->sendSlaves(&numberOfGroups, sizeof(int));
        //LOG_CERR("coVRDistributionManager::assign info: sent (" << numberOfGroups << ")");

        for (int ctr = 0; ctr < numberOfGroups; ++ctr)
        {
            assignmentsArray[ctr] = this->renderGroups[ctr].assignObject(object);
            assignments.push_back(assignmentsArray[ctr]);
            LOG_CERR(" " << assignmentsArray[ctr]);
        }
        LOG_CERR(std::endl);

        msControl->sendSlaves(assignmentsArray, sizeof(int) * numberOfGroups);

        delete[] assignmentsArray;
    }
    else
    {
        int numberOfGroups;
        msControl->readMaster(&numberOfGroups, sizeof(int));
        //LOG_CERR("coVRDistributionManager::assign info: received (" << numberOfGroups << ")");
        int *assignmentsArray = new int[numberOfGroups];
        msControl->readMaster(assignmentsArray, sizeof(int) * numberOfGroups);

        for (int ctr = 0; ctr < numberOfGroups; ++ctr)
        {
            assignments.push_back(assignmentsArray[ctr]);
            LOG_CERR(" " << assignmentsArray[ctr]);
        }
        LOG_CERR(std::endl);

        delete[] assignmentsArray;
    }

    return assignments;
}

// ================== Render Node ==================

std::vector<bool> opencover::coVRDistributionManager::RenderNode::idle;
std::vector<bool> opencover::coVRDistributionManager::RenderNode::enabled;

opencover::coVRDistributionManager::RenderNode::RenderNode(const std::string &name)
    : nodename(name)
    , display(false)
    , slaveIndex(-1)

{

    if (coVRMSController::instance()->clusterSize() > ssize_t(idle.size()))
    {
        idle.resize(coVRMSController::instance()->clusterSize());
        enabled.resize(coVRMSController::instance()->clusterSize());
    }

    this->hostname = covise::coConfigConstants::getHostname().toStdString();

    covise::coConfigString displaysConfig = covise::coConfig::getInstance()->getString("COVER.Parallel.Displays");

    QStringList displays = ((QString)displaysConfig).split(",");

    if (displays.contains(hostname.c_str()))
        this->display = true;

    this->slaveIndex = coVRMSController::instance()->getID();
    enabled[this->slaveIndex] = true;
    idle[this->slaveIndex] = true;
}

opencover::coVRDistributionManager::RenderNode::RenderNode(const opencover::coVRDistributionManager::RenderNode &node)
    : nodename(node.nodename)
    , hostname(node.hostname)
    , display(node.display)
    , slaveIndex(node.slaveIndex)
    , screens(node.screens)
{
}

void opencover::coVRDistributionManager::RenderNode::setNodename(const std::string &name)
{
    this->nodename = name;
}

const std::string &opencover::coVRDistributionManager::RenderNode::getNodename() const
{
    return this->nodename;
}

const std::string &opencover::coVRDistributionManager::RenderNode::getHostname() const
{
    return this->hostname;
}

bool opencover::coVRDistributionManager::RenderNode::isIdle() const
{
    if (this->slaveIndex < 0)
        return false;
    else
        return idle[this->slaveIndex];
}

bool opencover::coVRDistributionManager::RenderNode::isDisplay() const
{
    return this->display;
}

bool opencover::coVRDistributionManager::RenderNode::isEnabled() const
{
    if (this->slaveIndex < 0)
        return false;
    else
        return enabled[this->slaveIndex];
}

int opencover::coVRDistributionManager::RenderNode::getSlaveIndex() const
{
    return this->slaveIndex;
}

bool opencover::coVRDistributionManager::RenderNode::operator==(const RenderNode &other)
{
    return this->nodename == other.nodename;
}

/**
  * Apply incoming screen setup to coVRConfig and update VRViewer
  */
void opencover::coVRDistributionManager::RenderNode::applyScreens()
{

    //TODO Adjust window size

    assert(ssize_t(this->screens.size()) <= opencover::coVRConfig::instance()->numScreens());
    for (size_t ctr = 0; ctr < this->screens.size(); ++ctr)
    {
        screenStruct &target = opencover::coVRConfig::instance()->screens[ctr];
        channelStruct &targetChannel = opencover::coVRConfig::instance()->channels[ctr];
        screenStruct &source = this->screens[ctr];
        channelStruct &sourceChannel = this->channels[ctr];

        target.hsize = source.hsize;
        target.vsize = source.vsize;
        target.xyz = source.xyz;
        target.hpr = source.hpr;

        targetChannel.stereoMode = sourceChannel.stereoMode;
        targetChannel.ds->setStereoMode((osg::DisplaySettings::StereoMode)sourceChannel.stereoMode);
        targetChannel.camera->setDisplaySettings(targetChannel.ds);

     /*  
     TODO set viewport config
        target.viewportXMin = source.viewportXMin;
        target.viewportYMin = source.viewportYMin;
        target.viewportXMax = source.viewportXMax;
        target.viewportYMax = source.viewportYMax;

        target.leftView = source.leftView;
        target.rightView = source.rightView;

        target.leftProj = source.leftProj;
        target.rightProj = source.rightProj;*/
        
    }

    //opencover::VRViewer::instance()->setChannelConfig();
    for (size_t ctr = 0; ctr < this->screens.size(); ++ctr)
    {
        opencover::VRViewer::instance()->setFrustumAndView(ctr);
        opencover::VRViewer::instance()->clearWindow = true;
    }
}

covise::TokenBuffer &opencover::operator<<(covise::TokenBuffer &tb, const opencover::coVRDistributionManager::RenderNode &node)
{

    int numScreens = node.screens.size();
    if (numScreens == 0)
        numScreens = opencover::coVRConfig::instance()->numScreens();

    tb << node.nodename;
    tb << node.hostname;
    tb << (char)node.display;
    tb << node.slaveIndex;

    tb << numScreens;

    for (int ctr = 0; ctr < numScreens; ++ctr)
    {
        opencover::screenStruct screen;
        opencover::channelStruct channel;
        opencover::viewportStruct viewport;

        if (node.screens.size() == 0)
        {
            screen = opencover::coVRConfig::instance()->screens[ctr];
            channel = opencover::coVRConfig::instance()->channels[ctr];
            viewport = opencover::coVRConfig::instance()->viewports[ctr];
        }
        else
        {
            screen = node.screens[ctr];
            channel = node.channels[ctr];
            viewport = node.viewports[ctr];
        }

        tb << screen.hsize;
        tb << screen.vsize;
        tb << screen.xyz;
        tb << screen.hpr;

        tb << viewport.window;
        tb << channel.stereoMode;

        tb << viewport.viewportXMin;
        tb << viewport.viewportYMin;
        tb << viewport.viewportXMax;
        tb << viewport.viewportYMax;

        tb << channel.leftView;
        tb << channel.rightView;

        tb << channel.leftProj;
        tb << channel.rightProj;
    }

    return tb;
}

covise::TokenBuffer &opencover::operator>>(covise::TokenBuffer &tb, opencover::coVRDistributionManager::RenderNode &node)
{
    char display = 0;
    int numScreens = 0;

    tb >> node.nodename;
    tb >> node.hostname;
    tb >> display;
    tb >> node.slaveIndex;

    tb >> numScreens;

    node.screens.clear();

    for (int ctr = 0; ctr < numScreens; ++ctr)
    {
        opencover::screenStruct screen;
        opencover::channelStruct channel;
        opencover::viewportStruct viewport;

        tb >> screen.hsize;
        tb >> screen.vsize;
        tb >> screen.xyz;
        tb >> screen.hpr;

        tb >> viewport.window;
        tb >> channel.stereoMode;

        tb >> viewport.viewportXMin;
        tb >> viewport.viewportYMin;
        tb >> viewport.viewportXMax;
        tb >> viewport.viewportYMax;

        tb >> channel.leftView;
        tb >> channel.rightView;

        tb >> channel.leftProj;
        tb >> channel.rightProj;
        
        node.screens.push_back(screen);
        node.channels.push_back(channel);
        node.viewports.push_back(viewport);
    }

    node.display = (bool)display;

    return tb;
}

// ================== Render Group ==================

opencover::coVRDistributionManager::RenderGroup::RenderGroup()
    : id(-1)
{
    std::string algo = ((QString)covise::coConfig::getInstance()->getString("value", "COVER.Parallel.DistributionAlgorithm", "RoundRobin")).toStdString();

    this->distributionAlgorithm = DistributionAlgorithm::create(algo);

    assert(this->distributionAlgorithm);
}

void opencover::coVRDistributionManager::RenderGroup::setGroupID(int id)
{
    this->id = id;
}

void opencover::coVRDistributionManager::RenderGroup::setMaster(const opencover::coVRDistributionManager::RenderNode &master)
{
    this->distributionAlgorithm->removeSlave(this->master);
    this->master = master;
    this->distributionAlgorithm->addSlave(this->master);

    this->hostlist = this->master.getNodename();
    for (size_t ctr = 0; ctr < this->slaves.size(); ++ctr)
        this->hostlist += " " + std::string(this->slaves[ctr].getNodename());
}

void opencover::coVRDistributionManager::RenderGroup::addSlave(const opencover::coVRDistributionManager::RenderNode &slave)
{
    if (std::find(this->slaves.begin(), this->slaves.end(), slave) == this->slaves.end())
    {
        this->slaves.push_back(slave);
        this->hostlist.append(" ").append(slave.getNodename());
        this->distributionAlgorithm->addSlave(slave);
    }
}

void opencover::coVRDistributionManager::RenderGroup::removeSlave(const opencover::coVRDistributionManager::RenderNode &slave)
{

    this->distributionAlgorithm->removeSlave(slave);

    this->slaves.erase(std::find(this->slaves.begin(), this->slaves.end(), slave));

    this->hostlist = this->master.getNodename();
    for (size_t ctr = 0; ctr < this->slaves.size(); ++ctr)
        this->hostlist.append(" ").append(std::string(this->slaves[ctr].getNodename()));
}

const std::string &opencover::coVRDistributionManager::RenderGroup::getHostlist() const
{
    return this->hostlist;
}

int opencover::coVRDistributionManager::RenderGroup::assignObject(const covise::coDistributedObject *object)
{
#ifndef COVISEPLUGIN
    if (object && object->getAttribute("MODEL_FILE"))
    {
        this->distributionAlgorithm->assignedObjectToAll(object);
        return -1;
    }
#endif

    return this->distributionAlgorithm->assignObject(object);
}

// ================== Render Distribution Algorithm ==================

opencover::coVRDistributionManager::DistributionAlgorithm::DistributionAlgorithm(const std::string &algoName)
    : name(algoName)
{
}

void opencover::coVRDistributionManager::DistributionAlgorithm::addSlave(const opencover::coVRDistributionManager::RenderNode &slave)
{
    if (this->slaves.find(slave.nodename) == this->slaves.end())
    {
        this->slaves[slave.nodename] = slave;
    }
}

void opencover::coVRDistributionManager::DistributionAlgorithm::removeSlave(const opencover::coVRDistributionManager::RenderNode &slave)
{
    this->slaves.erase(slave.nodename);
}

std::map<std::string, opencover::coVRDistributionManager::DACreate> &
opencover::coVRDistributionManager::DistributionAlgorithm::registerAlgorithm(const std::string &name,
                                                                             opencover::coVRDistributionManager::DACreate creator)
{
    static std::map<std::string, DACreate> algorithms;

    if (name == "")
        return algorithms;

    assert(creator);

    if (algorithms.find(name) == algorithms.end())
    {
        algorithms[name] = creator;
    }

    return algorithms;
}

opencover::coVRDistributionManager::DistributionAlgorithm *opencover::coVRDistributionManager::DistributionAlgorithm::create(const std::string &name)
{
    std::map<std::string, DACreate> &algorithms = registerAlgorithm();
    if (algorithms.find(name) != algorithms.end())
    {
        return algorithms[name]();
    }
    else
    {
        return 0;
    }
}

// ================== Round Robin Render Distribution Algorithm ==================

static opencover::coVRDistributionManager::DARegistration rrReg("RoundRobin", opencover::coVRDistributionManager::RoundRobinDA::create);

opencover::coVRDistributionManager::RoundRobinDA::RoundRobinDA()
    : DistributionAlgorithm("RoundRobin")
{
    LOG_CERR("RoundRobinDA::<init> info: creating..." << std::endl);
}

int opencover::coVRDistributionManager::RoundRobinDA::assignObject(const covise::coDistributedObject *)
{
    if (this->currentNodename == "")
        this->currentNodename = (this->slaves.begin())->first;

    int id = this->slaves[this->currentNodename].getSlaveIndex();

    std::map<std::string, RenderNode>::iterator node = this->slaves.find(this->currentNodename);

    ++node;

    if (node == this->slaves.end())
        node = (this->slaves.begin());

    this->currentNodename = node->first;

    return id;
}

// ================== Element Count Render Distribution Algorithm ==================

static opencover::coVRDistributionManager::DARegistration ecReg("ElementCount", opencover::coVRDistributionManager::ElementCountDA::create);

opencover::coVRDistributionManager::ElementCountDA::ElementCountDA()
    : DistributionAlgorithm("ElementCount")
{
    LOG_CERR("ElementCountDA::<init> info: creating, also creating RoundRobinDA..." << std::endl);
    this->defaultDA = new RoundRobinDA();
}

opencover::coVRDistributionManager::ElementCountDA::~ElementCountDA()
{
    delete this->defaultDA;
}

int opencover::coVRDistributionManager::ElementCountDA::assignObject(const covise::coDistributedObject *object)
{

    LOG_CERR("X");

    int elementCount = 0;
    int minElementIndex = 0;
    int minElementCount = this->elementCounts[0];

#ifndef COVISEPLUGIN
    const covise::coDistributedObject *geometry = dynamic_cast<const covise::coDoGeometry *>(object);

    if (geometry != 0)
    {
        const covise::coDoGeometry *geo = dynamic_cast<const covise::coDoGeometry *>(geometry);
        while (dynamic_cast<const covise::coDoGeometry *>(geo->getGeometry()))
        {
            geo = dynamic_cast<const covise::coDoGeometry *>(geo->getGeometry());
        }
        geometry = geo;
    }

    const covise::coDoPoints *points = dynamic_cast<const covise::coDoPoints *>(geometry);
    const covise::coDoLines *lines = dynamic_cast<const covise::coDoLines *>(geometry);
    const covise::coDoPolygons *polygons = dynamic_cast<const covise::coDoPolygons *>(geometry);
    const covise::coDoTriangleStrips *tstrips = dynamic_cast<const covise::coDoTriangleStrips *>(geometry);

    if (points != 0)
        elementCount = points->getNumPoints();
    else if (lines != 0)
        elementCount = lines->getNumLines();
    else if (polygons != 0)
        elementCount = polygons->getNumPolygons();
    else if (tstrips != 0)
        elementCount = tstrips->getNumStrips();
    else
    {
        LOG_CERR("d");
        return this->defaultDA->assignObject(object);
    }
#endif

    LOG_CERR("e");

    for (size_t ctr = 1; ctr < this->elementCounts.size(); ++ctr)
    {
        minElementCount = std::min(minElementCount, this->elementCounts[ctr]);
        if (minElementCount == this->elementCounts[ctr])
            minElementIndex = ctr;
    }

    this->elementCounts[minElementIndex] += elementCount;

    LOG_CERR("ElementCountDA::assignObject info: assigning to slave " << this->slaves[minElementIndex].getSlaveIndex() << ", growing element count " << minElementCount << " -> " << this->elementCounts[minElementIndex]);

    return this->slaves[minElementIndex].getSlaveIndex();
}

void opencover::coVRDistributionManager::ElementCountDA::addSlave(const opencover::coVRDistributionManager::RenderNode &slave)
{
    if (std::find(this->slaves.begin(), this->slaves.end(), slave) == this->slaves.end())
    {
        this->slaves.push_back(slave);
        this->elementCounts.push_back(0);
        this->defaultDA->addSlave(slave);
    }
}

void opencover::coVRDistributionManager::ElementCountDA::removeSlave(const opencover::coVRDistributionManager::RenderNode &slave)
{
    for (size_t ctr = 0; ctr < this->slaves.size(); ++ctr)
    {
        if (this->slaves[ctr] == slave)
        {
            this->slaves.erase(this->slaves.begin() + ctr);
            this->elementCounts.erase(this->elementCounts.begin() + ctr);
            this->defaultDA->removeSlave(slave);
        }
    }
}
