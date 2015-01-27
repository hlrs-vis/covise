/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */



#include "handler/cyMirrorListHandler.h"

//======================================================================

cyMirrorListHandler::cyMirrorListHandler()
    : QObject()
{
}

cyMirrorListHandler *cyMirrorListHandler::instance()
{
    static cyMirrorListHandler *singleton = 0;
    if (singleton == 0)
        singleton = new cyMirrorListHandler();

    return singleton;
}

//======================================================================
cyMirrorListHandler::~cyMirrorListHandler()
//======================================================================
{
    qDeleteAll(mirrorList);
    mirrorList.clear();
}

//======================================================================
// set mirror (CSCW)
// set hosts for mirroring
//======================================================================
void cyMirrorListHandler::setMirror()
{
    /*   if(hostList.count() == 1)
   {
      printMessage("You are working only locally. \nIt makes no sense to mirror modules. Please add a host or partner");
      return;
   }

   if(mirrorMode >= 2)
   {
      printMessage("You can't reset the mirrors, if you have already started to mirror nodes");
      return;
   }

   if(mirrorMode == 0)
   {
      if(!mirrorBox)
         makeMirrorBox();
      mirrorMode = 1;
   }

   if(mirrorBox)
      mirrorBox->show();*/
}

//======================================================================
// start mirroring (CSCW)
// copy nodes to the mirror hosts
// if a host has no mirror copy the module to the same host
//======================================================================
void cyMirrorListHandler::addMirrorNodes()
{
    /*
   // check if it is possible to do some reflection
   if(hostList.count() == 1)
   {
      printMessage("You are working only locally. \nIt makes no sense to mirror modules. Please add a host or partner");
      return;
   }

   // check current mirror state
   if(mirrorMode >= 2)
   {
      printMessage("You have already mirrored all local nodes");
      return;
   }

   // check if user has already set his mirrors
   if(mirrorMode == 0)
   {
      if(!mirrorBox)
         makeMirrorBox();
      mirrorMode = 1;
      mirrorBox->show();
      printMessage("You have to set your mirrors before");
      return;
   }

   mirrorNodes();*/
}

//======================================================================
// copy all current nodes to a mirror
//======================================================================
void cyMirrorListHandler::mirrorNodes()
{
    /*
   // first make a selected_node_list automatically
   // this is necessary to get the right connections
   if(copyList.isEmpty() )
   {
      foreach ( cyNode *node, nodeList)
         copyList << node;
   }


   // get max number of mirrors
   int ln, nb_mirrors = 0;
   foreach ( cyHost *host, hostList )
   {
      ln = host->mirrorNames.count();
      if( ln > nb_mirrors )
         nb_mirrors = ln;
   }


   // send  Mirror_State on all UI's
   QString tmp = "MIRROR_STATE\n" + QString::number(nb_mirrors + 1);
   sendMessage(COVISE_MESSAGE_UI, tmp);


   // first loop goes over the no. og mirror operations
   for(int cnt = 0; cnt < nb_mirrors; cnt++)
   {

      // loop over all nodes
      foreach ( cyNode *node, copyList)
      {
         // search for the host & get its current mirror
         // copy nodes to the same host if no mirror exist
         cyHost *host   = node->getHost();
         cyHost *mirror = host->mirrorNames.at(cnt);
         if(mirror == NULL)
            tmp = host->getIPAddress();
         else
            tmp = mirror->getIPAddress();

         requestNode(node->getName(), tmp, node->x()+(cnt+1)*200, node->y(), node, cyMirrorLinkList::SYNC);
      }
   }

   // send the message to deactivate selection
   tmp = "CLEAR_COPY_LIST\n";
   sendMessage(COVISE_MESSAGE_UI, tmp);*/
}

//======================================================================
// delete all synced nodes in the canvas (CSCW)
//======================================================================
void cyMirrorListHandler::delMirrorNodes()
{
    /*   QStringList buffer;

   foreach ( cyNode *node, mirrorList)
   {
      buffer << "DEL_SYNC" << node->getName() << node->getNumber() << node->getIPAddress();
      QString data = buffer.join("\n");
      sendMessage(COVISE_MESSAGE_UI, data);
      buffer.clear();
      mapWasChanged("DEL_SYNC");
   }

   // reset mirror state
   // send  Mirror_State on all UI's
   QString tmp = "MIRROR_STATE\n" + QString::number(1);
   sendMessage(COVISE_MESSAGE_UI, tmp);*/
}

//------------------------------------------------------------------------
// mirror state changed
// change module popup menu
// permit deletion of hosts
//------------------------------------------------------------------------
void cyMirrorListHandler::mirrorStateChanged(int newMode)
{
    /*   int oldMode    = mirrorMode;
   mirrorMode = newMode;


   // enable move/copy widget for every node; def. in cyMirror.cpp
   if( mirrorMode >= 2 && oldMode < 2 )
   {
      canvasArea->disableNodePopupItems(false);
      //delListBox->setEnabled(false);
   }

   else if( oldMode >=2 && mirrorMode < 2 )
      canvasArea->disableNodePopupItems(true);
*/
}
