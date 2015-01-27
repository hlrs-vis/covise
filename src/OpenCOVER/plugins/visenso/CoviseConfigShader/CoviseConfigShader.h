/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/****************************************************************************\
 **                                                       (C)2013 Visenso  **
 **                                                                        **
 ** Description: CoviseConfigShader - add shader via config entries        **
 **                                                                        **
 ** Author: C. Spenrath                                                    **
 **                                                                        **
\****************************************************************************/

#ifndef _COVISE_CONFIG_SHADER_H
#define _COVISE_CONFIG_SHADER_H

#include <cover/coVRPlugin.h>

#include <QRegExp>

using namespace opencover;

struct Definition
{
    QRegExp regexp;
    std::string shader;
    bool smooth;
    float transparency;
};

class CoviseConfigShader : public coVRPlugin
{
public:
    // constructor destructor
    CoviseConfigShader();
    virtual ~CoviseConfigShader();

    // variables of class
    static CoviseConfigShader *plugin;

    virtual bool init();
    virtual void addNode(osg::Node *, RenderObject *);
    virtual void guiToRenderMsg(const char *msg);
    virtual void preFrame();

private:
    void readConfig();
    void addShader(osg::Node *node);
    int getDefinitionIndex(osg::Node *node);
    void setTransparency(osg::Node *node, float transparency);

    std::vector<Definition> definitions;
    std::vector<std::string> transparencyList;
};

#endif
