/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief  interface class to COVISE

 \author Daniela Rainer
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   20.08.1997
 \date   10.07.1998 (Performer c++ interface)
 */
#ifndef VR_COVISE_CONN_H
#define VR_COVISE_CONN_H

#include <util/coExport.h>
#include <osg/Node>

namespace opencover
{
class buttonSpecCell;

class ObjectManager;
class VRCoviseConnection
{

private:
    static void quitInfoCallback(void *userData, void *callbackData);
    static void addObjectCallback(void *userData, void *callbackData);
    static void coviseErrorCallback(void *userData, void *callbackData);
    static void deleteObjectCallback(void *userData, void *callbackData);
    static void renderCallback(void *userData, void *callbackData);
    static void masterSwitchCallback(void *userData, void *callbackData);
    static void paramCallback(bool inMapLoading, void *userData, void *callbackData);

    void quitInfo(void *callbackData);
    void addObject(void *callbackData);
    void coviseError(void *callbackData);
    void deleteObject(void *callbackData);
    void masterSwitch(void *callbackData);
    void render(void *callbackData);
    void localParam(bool inMapLoading, void *callbackData);
    void receiveRenderMessage();
    int exitFlag;

    void hideObject(const char *objName, bool hide);
    void transformSGItem(const char *objName, float *row0, float *row1, float *row2, float *row3);
    void setColor(osg::Node *node, int *color);
    void setColor(const char *objName, int *color);
    void setMaterial(osg::Node *node, int *ambient, int *diffuse, int *specular, float shininess, float transparency);
    void setMaterial(const char *objectName, int *ambient, int *diffuse, int *specular, float shininess, float transparency);
    void setTransparency(osg::Node *node, float transparency);
    void setTransparency(const char *objectName, float transparency);
    void setShader(const char *objectName, const char *shaderName);

public:
    static VRCoviseConnection *covconn;

    VRCoviseConnection();

    ~VRCoviseConnection();

    void update(bool handleOneMessageOnly = false);
    void sendQuit();

    static void executeCallback(void *sceneGraph, buttonSpecCell *);

    void processRenderMessage(char *key, char *tmp);
};
}
#endif
