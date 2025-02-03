/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.
   */

#pragma once

#include <util/coExport.h>

#include <string>
#include <iostream>
namespace vsg
{
    class Node;
}
namespace vive
{

    class VVCORE_EXPORT vvRegisterSceneGraph
    {

    private:
        static vvRegisterSceneGraph* s_instance;
        int registerId;
        bool blocked;
        bool active;
        std::string whole_message;
        std::string transparency_message;
        std::string sceneGraphAppendixIdString;

    protected:
        vvRegisterSceneGraph();

        void createRegisterMessage(vsg::Node* node, std::string parent);
        void createUnregisterMessage(vsg::Node* node, std::string parent);
        void createTransparencyMessage(vsg::Node* node);
        void sendRegisterMessage();
        void sendUnregisterMessage();
        void sendTransparencyMessage();

        std::string createName(std::string);

    public:
        virtual ~vvRegisterSceneGraph();
        static vvRegisterSceneGraph* instance();

        void setRegisterStartIndex(int startIndex);

        void registerNode(vsg::Node* node, std::string parent);
        void unregisterNode(vsg::Node* node, std::string parent);

        void block()
        {
            blocked = true;
        };
        void activate()
        {
            active = true;
        };
        void unblock()
        {
            blocked = false;
        };
        bool isBlocked()
        {
            return blocked;
        };
    };
}
