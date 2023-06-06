/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cover/coVRPluginSupport.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRMSController.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRTui.h>
#include <util/unixcompat.h>

#include <osg/Group>
#include <osg/PositionAttitudeTransform>
#include <osg/ShapeDrawable>
#include <osg/MatrixTransform>
#include <osg/Geometry>
#include <osg/LineWidth>

#include <OpenVRUI/coSubMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coLabel.h>
#include <OpenVRUI/coPanel.h>
#include <OpenVRUI/coFlatPanelGeometry.h>
#include <OpenVRUI/coButtonMenuItem.h>
#include <OpenVRUI/coCheckboxMenuItem.h>

#include <iostream>
#include <fstream>
#include <functional>

#include <string>
#include <vector>

#include "AmberPlugin.h"

AmberPlugin *AmberPlugin::plugin = NULL;
coRowMenu *AmberPlugin::files_menu = NULL;
coRowMenu *AmberPlugin::spheres_menu = NULL;
coRowMenu *AmberPlugin::sticks_menu = NULL;

// mapping of filenames to Amber structures containing the data in the files
std::map<const char *, Amber *, ltstr> AmberPlugin::amber_map;

static const int NUM_HANDLERS = 2;

static const FileHandler handlers[] = {
    { NULL,
      AmberPlugin::loadAmber,
      AmberPlugin::unloadAmber,
      "top" },
    { NULL,
      AmberPlugin::loadAmber,
      AmberPlugin::unloadAmber,
      "trj" }
};

AmberPlugin::AmberPlugin()
: coVRPlugin(COVER_PLUGIN_NAME)
, tab(NULL)
, res(NULL)
, residue(NULL)
, showSticks(NULL)
, hideSticks(NULL)
{
    time_step = 0;
}

int AmberPlugin::removeAmber(const char *name)
{

    std::map<const char *, Amber *, ltstr>::iterator it = amber_map.find(name);

    if (it != amber_map.end())
    {

        Amber *amber = amber_map[(char *)name];

        std::vector<osg::Group *>::iterator i;
        for (i = amber->groups.begin(); i != amber->groups.end(); i++)
        {

            if (amber->parent)
                amber->parent->removeChild(*i);
        }

        delete (it->first);
        amber_map.erase(it);

        delete amber;
        return 0;
    }
    return 1;
}

int AmberPlugin::loadAmber(const char *filename, osg::Group *parent, const char *)
{

    char top[256], trj[256];
    memset(top, 0, sizeof(top));
    memset(trj, 0, sizeof(trj));

    if (filename)
    {
        int len = strlen(filename);

        if (len > 4 && len < 251)
        {
            memcpy(top, filename, len - 4);
            memcpy(trj, filename, len - 4);

            removeAmber(top);

            sprintf(top + len - 4, ".top");
            sprintf(trj + len - 4, ".trj");

            if (plugin)
            {

                Amber *amber = new Amber();
                if (parent)
                    amber->parent = parent;
                else
                    amber->parent = cover->getObjectsRoot();

                amber_map[top] = amber;

                plugin->loadAmber(top, trj, amber);
            }
        }
    }

    return 0;
}

/*
 * load Amber topology and trajectory files. Atoms are represented
 * as coSphere, bonds as osg::PrimitiveSet::LINES
 *
 * Specification of the file format can be found at
 *    http://amber.scripps.edu/formats.html
 */
int AmberPlugin::loadAmber(const char *top, const char *trj, Amber *amber)
{

    if (files_menu)
    {

        coMenuItemVector items = files_menu->getAllItems();
        int index = 0;
        while (index < items.size())
        {

            if (!strcmp(items[index]->getName(), top))
            {
                coCheckboxMenuItem *check = dynamic_cast<coCheckboxMenuItem *>(items[index]);
                if (check)
                    check->setState(true);
                break;
            }
            index++;
        }

        if (!items[index])
        {
            coCheckboxMenuItem *menu_item = new coCheckboxMenuItem(top, true);
            menu_item->setMenuListener(this);
            files_menu->add(menu_item);
        }
    }

    time_step = 0;

    color_map["C"] = osg::Vec4(200.0 / 255.0, 200.0 / 255.0, 200.0 / 255.0, 1);
    color_map["H"] = osg::Vec4(1, 1, 1, 1);
    color_map["O"] = osg::Vec4(240.0 / 255.0, 0, 0, 1);
    color_map["N"] = osg::Vec4(143.0 / 255.0, 143.0 / 255.0, 1, 1);
    color_map["S"] = osg::Vec4(1, 200.0 / 255.0, 50.0 / 255.0, 1);
    color_map["P"] = osg::Vec4(1, 165.0 / 255.0, 0, 1);

    osg::Group *sphere_group = new osg::Group();
    amber->parent->addChild(sphere_group);

    osg::Group *bond_group = new osg::Group();
    amber->parent->addChild(bond_group);

    amber->groups.push_back(sphere_group);
    amber->groups.push_back(bond_group);

    osg::StateSet *state_set = bond_group->getOrCreateStateSet();
    state_set->setMode(GL_LIGHTING, osg::StateAttribute::PROTECTED | osg::StateAttribute::OFF);

    int state = AMBER_TOP, sect = AMBER_TOP;

    std::ifstream file(top);

    if (file.is_open())
    {

        char line[512];
        char format[8];

        char buf[512];

        int num;
        char type;
        int dig, dec;

        while (file.getline(line, sizeof(line)))
        {

            if (line[0] == '%')
                state = AMBER_TOP;

            switch (state)
            {

            case AMBER_TOP:
                if (sscanf(line, "%%VERSION %s", buf))
                {
                }
                else if (sscanf(line, "%%FLAG %s", buf))
                {

                    if (!strncmp(buf, "ATOM_NAME", 9))
                        sect = AMBER_ATOM_NAME;
                    else if (!strncmp(buf, "ATOM_TYPE_INDEX", 15))
                        sect = AMBER_ATOM_TYPE_INDEX;
                    else if (!strncmp(buf, "RESIDUE_LABEL", 13))
                        sect = AMBER_RESIDUE_LABEL;
                    else if (!strncmp(buf, "RESIDUE_POINTER", 15))
                        sect = AMBER_RESIDUE_POINTER;
                    else if (!strncmp(buf, "AMBER_ATOM_TYPE", 15))
                        sect = AMBER_AMBER_ATOM_TYPE;
                    else if (!strncmp(buf, "RADII", 5))
                        sect = AMBER_RADII;
                    else if (!strncmp(buf, "BONDS_INC_HYDROGEN", 18))
                        sect = AMBER_BONDS_INC_HYDROGEN;
                    else if (!strncmp(buf, "BONDS_WITHOUT_HYDROGEN", 18))
                        sect = AMBER_BONDS_NO_HYDROGEN;
                    else if (!strncmp(buf, "TREE_CHAIN_CLASSIFICATION", 21))
                        sect = AMBER_TREE_CHAIN_CLASSIFICATION;

                    else
                        sect = AMBER_UNKNOWN;
                }
                else if (sscanf(line, "%%FORMAT(%s)", buf))
                {

                    dec = 0;
                    if (sscanf(buf, "%d%c%d.%d", &num, &type, &dig, &dec))
                    {
                        //printf("format [%d %c %d %d]\n", num, type, dig, dec);
                    }
                    state = sect;
                }
                break;

            default: // data

                for (int index = 0; index < num; index++)
                {

                    if (type == 'a')
                    { // char
                        if (state != AMBER_UNKNOWN)
                        {
                            char *name = (char *)calloc(dig + 1, sizeof(char));
                            name[dig] = '\0';
                            snprintf(format, 8, "%%%dc", dig);
                            int n;
                            if ((n = sscanf(line + index * dig, format, name)) > 0)
                            {
                                if (state == AMBER_ATOM_NAME)
                                {
                                    amber->atom_name.push_back(std::string(name));
                                    if (!strcmp(name, "C   "))
                                        amber->atom_name_enum.push_back(C);
                                    else if (!strcmp(name, "N   "))
                                        amber->atom_name_enum.push_back(N);
                                    else if (!strcmp(name, "CA  "))
                                        amber->atom_name_enum.push_back(CA);
                                    else
                                        amber->atom_name_enum.push_back(OTHER);
                                }
                                else if (state == AMBER_RESIDUE_LABEL)
                                    amber->residue_label.push_back(std::string(name));
                                else if (state == AMBER_AMBER_ATOM_TYPE)
                                    amber->amber_atom_type.push_back(std::string(name));
                                else if (state == AMBER_TREE_CHAIN_CLASSIFICATION)
                                    amber->chain.push_back(std::string(name));
                            }
                            else if (n < 0)
                            {
                                // end of section
                                state = AMBER_TOP;
                                sect = AMBER_TOP;
                            }
                            free(name);
                        }
                        else
                        {
                            state = AMBER_TOP;
                            sect = AMBER_TOP;
                        }
                    }

                    else if (type == 'I')
                    { // int
                        if (state != AMBER_UNKNOWN)
                        {
                            int data;
                            snprintf(format, 8, "%%%dd", dig);
                            int n;
                            if ((n = sscanf(line + index * dig, format, &data)) > 0)
                            {
                                if (state == AMBER_ATOM_TYPE_INDEX)
                                    amber->atom_type_index.push_back(data);
                                else if (state == AMBER_RESIDUE_POINTER)
                                    amber->residue_pointer.push_back(data);
                                else if (state == AMBER_BONDS_INC_HYDROGEN)
                                {
                                    amber->bonds_index.push_back(data);
                                    amber->bonds_enabled.push_back(true);
                                }
                                else if (state == AMBER_BONDS_NO_HYDROGEN)
                                {
                                    amber->bonds_index.push_back(data);
                                    amber->bonds_enabled.push_back(true);
                                }
                            }
                            else if (n < 0)
                            {
                                // end of section
                                state = AMBER_TOP;
                                sect = AMBER_TOP;
                            }
                        }
                        else
                        {
                            state = AMBER_TOP;
                            sect = AMBER_TOP;
                        }
                    }
                    else if (type == 'E')
                    { // float
                        if (state != AMBER_UNKNOWN)
                        {
                            float data;
                            snprintf(format, 8, "%%%df", dig);
                            int n;
                            if ((n = sscanf(line + index * dig, format, &data)) > 0)
                            {
                                if (state == AMBER_RADII)
                                {
                                    amber->radii.push_back(data);
                                    amber->tmp_radii.push_back(data);
                                }
                            }
                            else if (n < 0)
                            {
                                // end of section
                                state = AMBER_TOP;
                                sect = AMBER_TOP;
                            }
                        }
                        else
                        {
                            state = AMBER_TOP;
                            sect = AMBER_TOP;
                        }
                    }
                }
                break;
            }

            memset(line, 0, sizeof(line));
        }
    }
    file.close();
    file.clear(); // close does not clear the error bits on windows

    // if trajectories file contains a header,
    // let's hope there are no numbers in it
    file.open(trj);
    if (file.is_open())
    {
        float num;
        while (!file.eof())
        {
            if (file >> num)
            {
                amber->trajectories.push_back(num);
            }
            else
            {
                file.clear();
                file.ignore();
            }
        }
    }
    else
    {
        cerr << "Could not open " << trj << endl;
    }
    if (amber->atom_type_index.size() == 0 || amber->trajectories.size() == 0)
    {
        return -1;
    }

    // spheres
    //float *r, *g, *b, *a;
    float *radii;
    //r = new float[amber->atom_type_index.size()];
    //g = new float[amber->atom_type_index.size()];
    //b = new float[amber->atom_type_index.size()];
    //a = new float[amber->atom_type_index.size()];
    amber->x = new float[amber->atom_type_index.size()];
    amber->y = new float[amber->atom_type_index.size()];
    amber->z = new float[amber->atom_type_index.size()];
    radii = new float[amber->atom_type_index.size()];

    amber->sphere = new coSphere();
    amber->sphere->setNumberOfSpheres(amber->atom_type_index.size());

    for (int index = 0; index < amber->atom_type_index.size(); index++)
    {

        amber->x[index] = amber->trajectories[index * 3];
        amber->y[index] = amber->trajectories[index * 3 + 1];
        amber->z[index] = amber->trajectories[index * 3 + 2];
        radii[index] = amber->radii[index];
    }

    amber->sphere->setCoords(amber->atom_type_index.size(), amber->x, amber->y, amber->z, radii);
    amber->sphere->setRenderMethod(coSphere::RENDER_METHOD_CG_SHADER);
    amber->sphere->setRenderMethod(coSphere::RENDER_METHOD_ARB_POINT_SPRITES);

    for (int index = 0; index < amber->atom_type_index.size(); index++)
    {

        char name[2];
        name[0] = amber->amber_atom_type[index][0];
        name[1] = 0;
        osg::Vec4 color;

        if (color_map.find(name) != color_map.end())
        {
            color = color_map[name];
            amber->sphere->setColor(index, color.r(), color.g(), color.b(), color.a());
        }
    }

    amber->sphere->setColorBinding(osg::Geometry::BIND_OVERALL);

    osg::Geode *geode = new osg::Geode();
    geode->addDrawable(amber->sphere);
    sphere_group->addChild(geode);

    amber->time_steps = amber->trajectories.size() / 3 / amber->atom_type_index.size();

    coVRAnimationManager::instance()->setNumTimesteps(amber->time_steps, plugin);
    coVRAnimationManager::instance()->setAnimationSpeed(1);
    coVRAnimationManager::instance()->enableAnimation(false);

    for (int index = 0; index < amber->residue_label.size(); index++)
        amino.insert(amber->residue_label[index]);

    spheres_menu->removeAll();
    coCheckboxMenuItem *menu_item = new coCheckboxMenuItem("all/none", true);
    menu_item->setMenuListener(this);
    spheres_menu->add(menu_item);

    std::set<std::string>::iterator i;
    for (i = amino.begin(); i != amino.end(); i++)
    {

        menu_item = new coCheckboxMenuItem(*i, true);
        menu_item->setMenuListener(this);
        spheres_menu->add(menu_item);
    }

    // bonds
    osg::LineWidth *lnw = new osg::LineWidth;
    lnw->setWidth(3);
    state_set = new osg::StateSet();
    state_set->setAttribute(lnw);

    amber->bonds = new osg::Vec3Array(4 * amber->bonds_index.size());

    osg::Vec4Array *color = new osg::Vec4Array();
    int num_bonds = amber->bonds_index.size() / 3;

    for (int index = 0; index < num_bonds; index++)
    {

        int a1 = amber->bonds_index[index * 3] / 3;
        int a2 = amber->bonds_index[index * 3 + 1] / 3;

        osg::Vec3 v1(amber->trajectories[a1 * 3],
                     amber->trajectories[a1 * 3 + 1],
                     amber->trajectories[a1 * 3 + 2]);

        osg::Vec3 v2(amber->trajectories[a2 * 3],
                     amber->trajectories[a2 * 3 + 1],
                     amber->trajectories[a2 * 3 + 2]);

        osg::Vec3 m = v2 + v1;
        m *= 0.5;

        (*amber->bonds)[index * 4] = v1;
        (*amber->bonds)[index * 4 + 1] = m;
        (*amber->bonds)[index * 4 + 2] = m;
        (*amber->bonds)[index * 4 + 3] = v2;

        char n1[2], n2[2];
        n1[0] = amber->amber_atom_type[a1][0];
        n1[1] = 0;
        n2[0] = amber->amber_atom_type[a2][0];
        n2[1] = 0;
        osg::Vec4 c1(1.0, 1.0, 1.0, 1.0);
        osg::Vec4 c2(1.0, 1.0, 1.0, 1.0);

        if (color_map.find(n1) != color_map.end())
            c1 = color_map[n1];
        if (color_map.find(n2) != color_map.end())
            c2 = color_map[n2];

        color->push_back(c1);
        color->push_back(c1);
        color->push_back(c2);
        color->push_back(c2);
    }

    amber->bonds_geom = new osg::Geometry();
    amber->bonds_geom->setStateSet(state_set);
    amber->bonds_geom->setVertexArray(amber->bonds);

    amber->bonds_geom->setColorArray(color);
    amber->bonds_geom->setColorBinding(osg::Geometry::BIND_PER_VERTEX);

    amber->bonds_geom->addPrimitiveSet(new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, num_bonds * 4));
    geode = new osg::Geode();
    geode->addDrawable(amber->bonds_geom);

    bond_group->addChild(geode);

    char buf[32];
    snprintf(buf, 32, "%lu", (unsigned long)(amber->residue_pointer.size() - 1));
    residue->setText(buf);

    return 0;
}

int AmberPlugin::unloadAmber(const char *filename, const char *)
{

    char buf[256];
    memset(buf, 0, 256);

    if (filename)
    {
        int len = strlen(filename);

        if (len > 4 && len < 260)
        {
            memcpy(buf, filename, len - 4);
            removeAmber(buf);
        }
    }

    // todo: build updated spheres menu

    return 0;
}

// this is called if the plugin is removed at runtime
AmberPlugin::~AmberPlugin()
{

    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->unregisterFileHandler(&handlers[index]);

    delete tab;
    delete res;
    delete residue;
    delete showSticks;
    delete hideSticks;
}

bool AmberPlugin::init()
{

    fprintf(stderr, "AmberPlugin::Plugin\n");
    if (plugin)
    {
        fprintf(stderr, "already have an instance of AmberPlugin\n");
        return false;
    }

    plugin = this;

    for (int index = 0; index < NUM_HANDLERS; index++)
        coVRFileManager::instance()->registerFileHandler(&handlers[index]);

    coMenu *cover_menu = cover->getMenu();

    if (cover_menu)
    {
        coSubMenuItem *button = new coSubMenuItem("Amber");
        coRowMenu *amber_menu = new coRowMenu("Amber");

        coSubMenuItem *files = new coSubMenuItem("Files");
        files_menu = new coRowMenu("Files");
        files->setMenu(files_menu);
        files->setMenuListener(this);

        coSubMenuItem *spheres = new coSubMenuItem("Amino Acid Spheres");
        spheres_menu = new coRowMenu("Amino Acid Spheres");
        spheres->setMenu(spheres_menu);
        spheres->setMenuListener(this);

        coSubMenuItem *sticks = new coSubMenuItem("Sticks");
        sticks_menu = new coRowMenu("Sticks");
        sticks->setMenu(sticks_menu);
        sticks->setMenuListener(this);

        coMenuItem *menu_item = new coCheckboxMenuItem("Main Chain only", false);
        menu_item->setMenuListener(this);
        sticks_menu->add(menu_item);

        amber_menu->add(files);
        amber_menu->add(spheres);
        amber_menu->add(sticks);

        button->setMenu(amber_menu);

        cover_menu->add(button);
    }

    tab = new coTUITab("Amber", coVRTui::instance()->mainFolder->getID());
    tab->setPos(0, 0);

    res = new coTUILabel("Sticks Residue #", tab->getID());
    res->setPos(0, 0);

    residue = new coTUIEditField("residue", tab->getID());
    residue->setEventListener(this);
    residue->setText("0");
    residue->setPos(1, 0);

    showSticks = new coTUIButton("show", tab->getID());
    showSticks->setEventListener(this);
    showSticks->setPos(2, 0);

    hideSticks = new coTUIButton("hide", tab->getID());
    hideSticks->setEventListener(this);
    hideSticks->setPos(3, 0);

    return true;
}

/*
 * TUI provides the possibility to show/hide sticks of bonds that
 * do not belong to the backbone.
 * User input can be a single number of a residue or a range of numbers 
 */
void AmberPlugin::tabletEvent(coTUIElement *elem)
{

    if (elem == showSticks || elem == hideSticks)
    {

        bool show = (elem == showSticks);

        std::map<const char *, Amber *, ltstr>::iterator i;
        for (i = AmberPlugin::amber_map.begin(); i != AmberPlugin::amber_map.end(); i++)
        {
            Amber *amber = (*i).second;
            // strip whitespace from user input
            std::string s(residue->getText());
            using namespace std::placeholders;
            std::string::iterator it = std::remove_if(s.begin(), s.end(), std::bind(std::equal_to<char>(), _1, ' '));
            s = std::string(s.begin(), it);
            int from = 0, to = 0;
            int num = sscanf(s.c_str(), "%d-%d", &from, &to);
            if (num == 0)
                return;
            if (num == 1)
                to = from;
            if (from > to)
            {
                int tmp = from;
                from = to;
                to = tmp;
            }

            for (int res = from; res <= to; res++)
            {
                if (amber->residue_pointer.size() > res)
                {
                    int first = amber->residue_pointer[res];
                    int last;
                    if (res < amber->residue_pointer.size() - 1)
                        last = amber->residue_pointer[res + 1] - 1;
                    else
                        last = amber->atom_type_index.size();

                    int num_bonds = amber->bonds_index.size() / 3;
                    int num = amber->atom_type_index.size();

                    for (int index = 0; index < num_bonds; index++)
                    {

                        int a1 = amber->bonds_index[index * 3] / 3;
                        int a2 = amber->bonds_index[index * 3 + 1] / 3;
                        if (a1 >= first && a1 < last && a2 >= first && a2 < last)
                        {

                            osg::Vec3 v1(amber->trajectories[(num * time_step + a1) * 3],
                                         amber->trajectories[(num * time_step + a1) * 3 + 1],
                                         amber->trajectories[(num * time_step + a1) * 3 + 2]);

                            osg::Vec3 v2(amber->trajectories[(num * time_step + a2) * 3],
                                         amber->trajectories[(num * time_step + a2) * 3 + 1],
                                         amber->trajectories[(num * time_step + a2) * 3 + 2]);

                            if (amber->atom_name_enum[a1] == OTHER || amber->atom_name_enum[a2] == OTHER)
                                amber->bonds_enabled[index] = show;

                            if (!amber->bonds_enabled[index])
                            {
                                // "hide" bonds that are disabled
                                v1 = osg::Vec3(0.0, 0.0, 0.0);
                                v2 = osg::Vec3(0.0, 0.0, 0.0);
                            }
                            osg::Vec3 m = v2 + v1;
                            m *= 0.5;

                            (*amber->bonds)[index * 4] = v1;
                            (*amber->bonds)[index * 4 + 1] = m;
                            (*amber->bonds)[index * 4 + 2] = m;
                            (*amber->bonds)[index * 4 + 3] = v2;
                        }
                    }
                }
            }
            amber->bonds_geom->setVertexArray(amber->bonds);
        }
    }
}

void AmberPlugin::tabletReleaseEvent(coTUIElement * /*tUIItem*/)
{
}

/*
 * Move spheres and sticks to the positions belonging to 
 * the given timestep.
 */
void AmberPlugin::setTimestep(int t)
{

    if (time_step == t)
        return;

    time_step = t;

    std::map<const char *, Amber *, ltstr>::iterator i;
    for (i = AmberPlugin::amber_map.begin(); i != AmberPlugin::amber_map.end(); i++)
    {
        Amber *amber = (*i).second;

        if (time_step > amber->time_steps)
            continue;

        int num = amber->atom_type_index.size();

        // spheres
        for (int index = 0; index < amber->atom_type_index.size(); index++)
        {

            amber->x[index] = amber->trajectories[(num * t + index) * 3];
            amber->y[index] = amber->trajectories[(num * t + index) * 3 + 1];
            amber->z[index] = amber->trajectories[(num * t + index) * 3 + 2];
        }
        amber->sphere->updateCoords(amber->x, amber->y, amber->z);

        int num_bonds = amber->bonds_index.size() / 3;

        // sticks
        for (int index = 0; index < num_bonds; index++)
        {

            int a1 = amber->bonds_index[index * 3] / 3;
            int a2 = amber->bonds_index[index * 3 + 1] / 3;

            osg::Vec3 v1(amber->trajectories[(num * t + a1) * 3],
                         amber->trajectories[(num * t + a1) * 3 + 1],
                         amber->trajectories[(num * t + a1) * 3 + 2]);

            osg::Vec3 v2(amber->trajectories[(num * t + a2) * 3],
                         amber->trajectories[(num * t + a2) * 3 + 1],
                         amber->trajectories[(num * t + a2) * 3 + 2]);

            if (amber->bonds_enabled[index])
            {
                osg::Vec3 m = v2 + v1;
                m *= 0.5;

                (*amber->bonds)[index * 4] = v1;
                (*amber->bonds)[index * 4 + 1] = m;
                (*amber->bonds)[index * 4 + 2] = m;
                (*amber->bonds)[index * 4 + 3] = v2;
            }
        }
        amber->bonds_geom->setVertexArray(amber->bonds);
    }
}

/*
 * files menu allows for loading/unloading of amber files
 *
 * spheres menu allows for showing/hiding spheres according to the type of 
 * amino acid.
 *
 * sticks menu allows for showing/hiding of bonds that do not belong to the
 * main chain
*/
void AmberPlugin::menuEvent(coMenuItem *item)
{

    coCheckboxMenuItem *check = dynamic_cast<coCheckboxMenuItem *>(item);

    if (check && check->getParentMenu() == spheres_menu)
    {

        if (!strcmp(item->getName(), "all/none"))
        {

            std::map<const char *, Amber *, ltstr>::iterator i;
            for (i = AmberPlugin::amber_map.begin(); i != AmberPlugin::amber_map.end(); i++)
            {
                Amber *amber = (*i).second;

                bool state = check->getState();
                for (int index = 0; index < amber->residue_label.size(); index++)
                {

                    int first = amber->residue_pointer[index] - 1;
                    int last;
                    if (index < amber->residue_pointer.size() - 1)
                        last = amber->residue_pointer[index + 1] - 2;
                    else
                        last = amber->atom_type_index.size() - 1;

                    for (int s = first; s <= last; s++)
                    {

                        if (state)
                            amber->tmp_radii[s] = amber->radii[index];
                        else
                            amber->tmp_radii[s] = 0;
                    }
                    amber->sphere->updateRadii(&(amber->tmp_radii[0]));
                }
            }
            coMenuItemVector items = spheres_menu->getAllItems();
            for (int index = 0; index < spheres_menu->getItemCount(); index++)
            {

                coCheckboxMenuItem *i = dynamic_cast<coCheckboxMenuItem *>(items[index]);
                if (i)
                    i->setState(check->getState());
            }
        }
        else
        {
            std::map<const char *, Amber *, ltstr>::iterator i;
            for (i = AmberPlugin::amber_map.begin(); i != AmberPlugin::amber_map.end(); i++)
            {
                Amber *amber = (*i).second;
                bool state = check->getState();
                for (int index = 0; index < amber->residue_label.size(); index++)
                {

                    if (!strcmp(amber->residue_label[index].c_str(), item->getName()))
                    {

                        int first = amber->residue_pointer[index] - 1;
                        int last;
                        if (index < amber->residue_pointer.size() - 1)
                            last = amber->residue_pointer[index + 1] - 2;
                        else
                            last = amber->atom_type_index.size() - 1;

                        for (int s = first; s <= last; s++)
                        {

                            if (state)
                                amber->tmp_radii[s] = amber->radii[index];
                            else
                                amber->tmp_radii[s] = 0;
                        }
                        amber->sphere->updateRadii(&(amber->tmp_radii[0]));
                    }
                }
            }
        }
    }
    else if (check && check->getParentMenu() == files_menu)
    {

        const char *filename = check->getName();
        if (!check->getState())
        {
            if (filename)
                removeAmber(filename);
        }
        else
        {
            if (filename)
                loadAmber(filename, cover->getObjectsRoot());
        }
    }
    else if (check && check->getParentMenu() == sticks_menu)
    {

        std::map<const char *, Amber *, ltstr>::iterator i;
        for (i = AmberPlugin::amber_map.begin(); i != AmberPlugin::amber_map.end(); i++)
        {
            Amber *amber = (*i).second;
            bool state = check->getState();
            int num = amber->atom_type_index.size();
            int num_bonds = amber->bonds_index.size() / 3;

            for (int index = 0; index < num_bonds; index++)
            {

                int a1 = amber->bonds_index[index * 3] / 3;
                int a2 = amber->bonds_index[index * 3 + 1] / 3;

                osg::Vec3 v1(amber->trajectories[(num * time_step + a1) * 3],
                             amber->trajectories[(num * time_step + a1) * 3 + 1],
                             amber->trajectories[(num * time_step + a1) * 3 + 2]);

                osg::Vec3 v2(amber->trajectories[(num * time_step + a2) * 3],
                             amber->trajectories[(num * time_step + a2) * 3 + 1],
                             amber->trajectories[(num * time_step + a2) * 3 + 2]);

                if (state && ((strcmp(amber->chain[a1].c_str(), "M   ") && strcmp(amber->chain[a2].c_str(), "M   ")) || (amber->atom_name_enum[a1] == OTHER || amber->atom_name_enum[a2] == OTHER)))
                {
                    v1 = osg::Vec3(0, 0, 0);
                    v2 = osg::Vec3(0, 0, 0);
                    amber->bonds_enabled[index] = false;
                }
                else if (!state)
                {
                    amber->bonds_enabled[index] = true;
                }

                osg::Vec3 m = v2 + v1;
                m *= 0.5;

                (*amber->bonds)[index * 4] = v1;
                (*amber->bonds)[index * 4 + 1] = m;
                (*amber->bonds)[index * 4 + 2] = m;
                (*amber->bonds)[index * 4 + 3] = v2;
            }
            amber->bonds_geom->setVertexArray(amber->bonds);
        }
    }
}

COVERPLUGIN(AmberPlugin)
