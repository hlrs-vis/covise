/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _CADCV3D_PLUGIN_H
#define _CADCV3D_PLUGIN_H
/****************************************************************************\ 
 **                                                            (C)2009 HLRS  **
 **                                                                          **
 ** Description: CADCv3D plugin                                               **
 ** Load data from University of Cologne CAD server                          **
 **                                                                          **
 ** Author: Andreas Kopecki                                                  **
 **                                                                          **
 **                                                                          **
\****************************************************************************/

#include <cover/coVRPlugin.h>
#include <cover/coVRPluginSupport.h>
using namespace covise;
using namespace opencover;

#include <osg/Group>
#include <cover/coVRIOReader.h>

#include <QString>

class CADCv3DGeoList;

class CADCv3DPlugin : public coVRPlugin, public coVRIOReader
{
public:
    CADCv3DPlugin();
    ~CADCv3DPlugin();

    // this will be called in PreFrame
    void preFrame();

    bool init();

    static CADCv3DPlugin *plugin;

    virtual osg::Node *load(const std::string &location, osg::Group *);
    virtual IOStatus loadPart(const std::string &location);
    virtual bool unload(osg::Node *node)
    {
        (void)node;
        return true;
    }

    virtual bool canLoadParts() const
    {
        return true;
    }
    virtual bool canUnload() const
    {
        return true;
    }
    virtual bool inLoading() const
    {
        return this->loading;
    }

    virtual bool abortIO();
    virtual std::string getIOHandlerName() const
    {
        return "CADCv3Plugin";
    }

    int loadCad(const char *filename, osg::Group *group);
    int replaceCad(const char *filename, osg::Group *group);
    int unloadCad(const char *filename);

    static int loadCadHandler(const char *filename, osg::Group *group);
    static int replaceCadHandler(const char *filename, osg::Group *group);
    static int unloadCadHandler(const char *filename);

private:
    bool loadData(const std::string &location);

    CADCv3DGeoList *root;
    osg::ref_ptr<osg::Group> rootNode;
    QString currentFilename;
    bool loading;
    bool renderWhileLoading;
};
#endif
