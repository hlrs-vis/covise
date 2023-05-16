/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

/*! \file
 \brief  handle COVISE data objects

 \author Dirk Rantzau
 \author Daniela Rainer
 \author Frank Foehl
 \author (C) 1996
         Computer Centre University of Stuttgart,
         Allmandring 30,
         D-70550 Stuttgart,
         Germany

 \date   20.08.1997
 \date   10.07.1998 (Performer c++ interface)
 */

#ifndef OBJECT_MANAGER_H
#define OBJECT_MANAGER_H

#include <osg/Matrix>
#include <osg/ColorMask>

#include <util/coMaterial.h>
#include <map>

#define MAXSETS 8000

namespace osg
{
class Group;
class Image;
class Texture1D;
}
namespace covise
{
class coDistributedObject;
}
namespace vrui
{
class coTrackerButtonInteraction;
}
class CoviseSG;
class CoviseRenderObject;
namespace opencover
{
class RenderObject;
class coTUIUITab;
class coVRShader;
class coInteractor;
class coVRPlugin;

struct ColorMap
{
    ColorMap();
    void setMinMax(float min, float max);

    struct RGBA {
        unsigned char r, g, b, a;
    };

    float min, max;
    std::vector<RGBA> lut;
    osg::ref_ptr<osg::Image> img;
    osg::ref_ptr<osg::Texture1D> tex;
    coVRShader *vertexMapShader, *textureMapShader;
};

//================================================================
// ObjectManager
//================================================================

class ObjectManager
{
private:
    covise::coMaterialList *materialList;
    CoviseSG *coviseSG;

    //
    // coDoSet handling
    //

    int anzset;
    char *setnames[MAXSETS];
    int elemanz[MAXSETS];
    char **elemnames[MAXSETS];

    int c_feedback;
    int i_feedback;
    int t_feedback;

    int added_a_rotate_flag;
    bool depthPeeling;

    /*	until button is released */

    char currentFeedbackInfo[256];

    void addColorMap(const char *object, CoviseRenderObject *cmap);
    //void addGeometry(char *object, int doreplace,int is_timestep,
    //                 char *root,coDistributedObject *geometry,
    //		     coDistributedObject *normals,
    //	     coDistributedObject *colors);
    osg::Node *addGeometry(const char *object, osg::Group *root, CoviseRenderObject *geometry,
                           CoviseRenderObject *normals, CoviseRenderObject *colors, CoviseRenderObject *texture, CoviseRenderObject *vertexAttribute, CoviseRenderObject *container, const char *lod);
    void removeGeometry(const char *name, bool);

    opencover::coInteractor *handleInteractors(CoviseRenderObject *container,
                           CoviseRenderObject *geo, CoviseRenderObject *norm, CoviseRenderObject *col, CoviseRenderObject *tex) const;

    osg::ColorMask *noFrameBuffer;
    vrui::coTrackerButtonInteraction *interactionA; ///< interaction for first button

    typedef std::map<std::string, ColorMap> ColorMaps;
    ColorMaps colormaps;
    const ColorMap &getColorMap(const std::string &species);

    typedef std::map<std::string, CoviseRenderObject *> RenderObjectMap;
    RenderObjectMap m_roMap;
    coVRPlugin *m_plugin = nullptr;

public:
    static ObjectManager *instance();

    ~ObjectManager();
    ObjectManager(coVRPlugin *plugin);
    void deleteObject(const char *name, bool groupobj = true);
    void addObject(const char *name, const covise::coDistributedObject *obj = NULL);
    void coviseError(const char *error);

    void update(void);
};
}
#endif
