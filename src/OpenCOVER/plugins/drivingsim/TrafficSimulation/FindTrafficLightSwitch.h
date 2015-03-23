/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FindTrafficLightSwitch_h
#define FindTrafficLightSwitch_h

#include <osg/NodeVisitor>
#include <osgSim/MultiSwitch>

class FindTrafficLightSwitch : public osg::NodeVisitor
{
public:
    enum SearchModeType
    {
        SEARCH_GROUP,
        SEARCH_SWITCH
    };

    FindTrafficLightSwitch(const std::string &setName, const osg::NodeVisitor::TraversalMode travMode = TRAVERSE_ALL_CHILDREN)
        : osg::NodeVisitor(travMode)
        , name(setName)
        , multiSwitchGreen(NULL)
        , multiSwitchYellow(NULL)
        , multiSwitchRed(NULL)
        , switchNameGreen("_SW_green")
        , switchNameYellow("_SW_yellow")
        , switchNameRed("_SW_red")
        , searchMode(SEARCH_GROUP)
    {
    }

    ~FindTrafficLightSwitch()
    {
    }

    void apply(osg::Node &node)
    {
        switch (searchMode)
        {
        case SEARCH_GROUP:
            if (node.getName() == name)
            {
                searchMode = SEARCH_SWITCH;
                traverse(node);
            }
            else
            {
                traverse(node);
            }
            break;

        case SEARCH_SWITCH:
            if (node.getName().find(switchNameGreen) != -1)
            {
                multiSwitchGreen = dynamic_cast<osgSim::MultiSwitch *>(&node);
            }
            else if (node.getName().find(switchNameYellow) != -1)
            {
                multiSwitchYellow = dynamic_cast<osgSim::MultiSwitch *>(&node);
            }
            else if (node.getName().find(switchNameRed) != -1)
            {
                multiSwitchRed = dynamic_cast<osgSim::MultiSwitch *>(&node);
            }

            if (!(multiSwitchGreen && multiSwitchYellow && multiSwitchRed))
            {
                traverse(node);
            }
            break;
        }
    }

    osgSim::MultiSwitch *getMultiSwitchGreen()
    {
        return multiSwitchGreen;
    }
    osgSim::MultiSwitch *getMultiSwitchYellow()
    {
        return multiSwitchYellow;
    }
    osgSim::MultiSwitch *getMultiSwitchRed()
    {
        return multiSwitchRed;
    }

protected:
    std::string name;
    osgSim::MultiSwitch *multiSwitchGreen;
    osgSim::MultiSwitch *multiSwitchYellow;
    osgSim::MultiSwitch *multiSwitchRed;
    std::string switchNameGreen;
    std::string switchNameYellow;
    std::string switchNameRed;
    SearchModeType searchMode;
};
#endif
