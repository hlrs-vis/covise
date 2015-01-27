/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRDISTRIBUTIONMANAGER_H
#define COVRDISTRIBUTIONMANAGER_H

#include <vector>
#include <string>
#include <list>
#include <map>
#include <cassert>

#include <util/coExport.h>
#include <cover/coVRConfig.h>

namespace covise
{
class coDistributedObject;
class TokenBuffer;
}

namespace opencover
{

class RenderObject;

class coVRDistributionManager
{
public:
    class DistributionAlgorithm;
    class DARegistration;

    class RenderNode
    {

        friend class coVRDistributionManager;
        friend class DistributionAlgorithm;

    public:
        RenderNode(const std::string &nodename = "");
        RenderNode(const RenderNode &);
        const std::string &getNodename() const;
        const std::string &getHostname() const;
        bool isIdle() const;
        bool isDisplay() const;
        bool isEnabled() const;
        int getSlaveIndex() const;

        void applyScreens();

        bool operator==(const RenderNode &);

    private:
        void setNodename(const std::string &name);

        std::string nodename;
        std::string hostname;
        bool display;
        int slaveIndex;

        std::vector<screenStruct> screens;

        static std::vector<bool> idle;
        static std::vector<bool> enabled;

        friend covise::TokenBuffer &operator<<(covise::TokenBuffer &, const opencover::coVRDistributionManager::RenderNode &);

        friend covise::TokenBuffer &operator>>(covise::TokenBuffer &, opencover::coVRDistributionManager::RenderNode &);
    };

    class RenderGroup
    {

        friend class coVRDistributionManager;

    public:
        RenderGroup();

        const RenderNode &getMaster() const
        {
            return this->master;
        }

        const std::vector<RenderNode> &getSlaves() const
        {
            return this->slaves;
        }

        const std::string &getHostlist() const;

        int assignObject(const covise::coDistributedObject *object);

    private:
        void setGroupID(int id);
        void setMaster(const RenderNode &master);
        void addSlave(const RenderNode &slave);
        void removeSlave(const RenderNode &slave);

        int id;
        RenderNode master;
        std::vector<RenderNode> slaves;
        std::string hostlist;

        std::map<std::string, int> assignedObjects;
        DistributionAlgorithm *distributionAlgorithm;
    };

    typedef DistributionAlgorithm *(*DACreate)();

    class DistributionAlgorithm
    {
    public:
        DistributionAlgorithm(const std::string &name);
        virtual ~DistributionAlgorithm()
        {
        }
        virtual void addSlave(const RenderNode &slave);
        virtual void removeSlave(const RenderNode &slave);

        virtual int assignObject(const covise::coDistributedObject *object) = 0;

        /// Called when an object is assigned to all nodes, can be used for internal purposes (polygon counting, etc.)
        virtual void assignedObjectToAll(const covise::coDistributedObject *object)
        {
            (void)object;
        }

        static DistributionAlgorithm *create(const std::string &name);

    protected:
        std::map<std::string, RenderNode> slaves;

    private:
        friend class DARegistration;
        static std::map<std::string, DACreate> &registerAlgorithm(const std::string &name = "", DACreate creator = 0);
        std::string name;
    };

    class DARegistration
    {
    public:
        DARegistration(const std::string &name, DACreate creator)
        {
            DistributionAlgorithm::registerAlgorithm(name, creator);
        }
    };

    class RoundRobinDA : public DistributionAlgorithm
    {
    public:
        RoundRobinDA();
        int assignObject(const covise::coDistributedObject *object);

        static DistributionAlgorithm *create()
        {
            return new RoundRobinDA();
        }

    private:
        std::string currentNodename;
    };

    class ElementCountDA : public DistributionAlgorithm
    {
    public:
        ElementCountDA();
        virtual ~ElementCountDA();
        int assignObject(const covise::coDistributedObject *object);

        static DistributionAlgorithm *create()
        {
            return new ElementCountDA();
        }

        virtual void addSlave(const RenderNode &slave);
        virtual void removeSlave(const RenderNode &slave);

    private:
        std::vector<int> elementCounts;
        std::vector<RenderNode> slaves;
        RoundRobinDA *defaultDA;
    };

    bool isActive() const;

    bool isDisplay() const;

    bool init();

    std::vector<int> assign(const covise::coDistributedObject *object);

    static coVRDistributionManager &instance()
    {
        static coVRDistributionManager instance;
        return instance;
    }

private:
    coVRDistributionManager();
    virtual ~coVRDistributionManager();

    coVRDistributionManager &operator=(const coVRDistributionManager &)
    {
        assert(0);
        return *this;
    }
    coVRDistributionManager(const coVRDistributionManager &)
    {
        assert(0);
    }

    bool active;

    bool display;

    std::vector<RenderNode> renderNodes;
    std::vector<RenderNode> displayNodes;

    RenderNode self;

    std::vector<RenderGroup> renderGroups;

    DistributionAlgorithm *distributionAlgorithm;
};

covise::TokenBuffer &operator<<(covise::TokenBuffer &, const coVRDistributionManager::RenderNode &);
covise::TokenBuffer &operator>>(covise::TokenBuffer &, coVRDistributionManager::RenderNode &);
}
#endif // COVRDISTRIBUTIONMANAGER_H
