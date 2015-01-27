/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _AMBER_PLUGIN_H
#define _AMBER_PLUGIN_H
/****************************************************************************\
 **                                                           (C)2008 HLRS **
 **                                                                        **
 ** Description: Amber Plugin                                              **
 **                                                                        **
 **                                                                        **
 ** Author: Florian Niebling                                               **
 **                                                                        **
 ** History:                                                               **
 ** 2008-Jul-23  v1	     		                                   **
 **                                                                        **
 **                                                                        **
\****************************************************************************/
#include <cover/coVRPluginSupport.h>

#include <OpenVRUI/coMenuItem.h>
#include <cover/coVRTui.h>

#include <osg/Group>
#include <osg/Matrix>
#include <osg/Material>

#include <PluginUtil/coSphere.h>
#include <map>

using namespace vrui;
using namespace opencover;

namespace vrui
{
class coRowMenu;
class coSubMenuItem;
}

namespace opencover
{
class ColorBar;
}

// name of atoms, needed to specify if an atom really belongs to the backbone
enum
{
    OTHER,
    C,
    N,
    CA
};

// parser states for Amber file format
enum
{
    AMBER_TOP,
    AMBER_UNKNOWN,
    AMBER_ATOM_NAME,
    AMBER_ATOM_TYPE_INDEX,
    AMBER_AMBER_ATOM_TYPE,
    AMBER_RESIDUE_LABEL,
    AMBER_RESIDUE_POINTER,
    AMBER_RADII,
    AMBER_BONDS_INC_HYDROGEN,
    AMBER_BONDS_NO_HYDROGEN,
    AMBER_TREE_CHAIN_CLASSIFICATION
};

struct ltstr
{
    bool operator()(const char *s1, const char *s2) const
    {
        return strcmp(s1, s2) < 0;
    }
};

class Amber
{

public:
    Amber()
        : time_steps(0)
        , sphere(NULL)
        , x(NULL)
        , y(NULL)
        , z(NULL)
        , bonds(NULL)
        , bonds_geom(NULL)
        , parent(NULL)
    {
    }

    int time_steps;

    // group node containing subgroups for spheres and bonds
    std::vector<osg::Group *> groups;

    // name of the atoms
    std::vector<std::string> atom_name;

    // enum name of atoms for fast test if atom belongs to the backbone
    std::vector<int> atom_name_enum;

    // type of atom
    std::vector<int> atom_type_index;

    // name of residues
    std::vector<std::string> residue_label;

    // atom types
    std::vector<std::string> amber_atom_type;

    // index of residue into list of atoms
    std::vector<int> residue_pointer;

    // bonds between atoms
    std::vector<int> bonds_index;

    // radii of atom spheres
    std::vector<float> radii;

    // temporary radii of atom spheres, will be set to 0 to hide spheres
    std::vector<float> tmp_radii;

    // trajectory of atoms (contains all timesteps)
    std::vector<float> trajectories;

    // the name of the chain that a given atom belongs to
    std::vector<std::string> chain;

    // spheres for atoms
    coSphere *sphere;

    // current position of spheres
    float *x, *y, *z;

    osg::Vec3Array *bonds;
    osg::Geometry *bonds_geom;
    std::vector<bool> bonds_enabled;

    osg::Group *parent;
};

class PLUGINEXPORT AmberPlugin : public coVRPlugin, public coMenuListener, public coTUIListener
{

public:
    static AmberPlugin *plugin;
    static std::map<const char *, Amber *, ltstr> amber_map;
    static coRowMenu *files_menu;
    static coRowMenu *spheres_menu;
    static coRowMenu *sticks_menu;

    AmberPlugin();
    ~AmberPlugin();

    bool init();

    static int removeAmber(const char *name);
    static int loadAmber(const char *filename, osg::Group *parent, const char *ck = "");
    int loadAmber(const char *top, const char *trj, Amber *amber);

    static int unloadAmber(const char *filename, const char *ck = "");

    void setTimestep(int t);
    void menuEvent(coMenuItem *item);

    void tabletReleaseEvent(coTUIElement *);
    void tabletEvent(coTUIElement *);

private:
    coTUITab *tab;
    coTUILabel *res;
    coTUIEditField *residue;
    coTUIButton *showSticks;
    coTUIButton *hideSticks;

    int time_step;
    std::map<const char *, osg::Vec4, ltstr> color_map;
    std::set<std::string> amino;
};

#endif
