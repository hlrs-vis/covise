/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include <cassert>

#include <iostream>
#include <memory>
#include <ostream>

#include <boost/algorithm/string.hpp>

#include <osg/Sequence>

#include <config/CoviseConfig.h>

#include <cover/coTabletUI.h>
#include <cover/coVRAnimationManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRTui.h>
#include <cover/VRSceneGraph.h>
#include <cover/VRViewer.h>
#include <cover/OpenCOVER.h>

#include <OpenVRUI/coCheckboxGroup.h>
#include <OpenVRUI/coCheckboxMenuItem.h>
#include <OpenVRUI/coRowMenu.h>
#include <OpenVRUI/coSliderMenuItem.h>
#include <OpenVRUI/coSubMenuItem.h>

#include "renderer.h"
#include "state.h"
#include "visionaray_plugin.h"

namespace visionaray
{

    //-------------------------------------------------------------------------------------------------
    // Private implementation
    //

    struct Visionaray::impl : vrui::coMenuListener, opencover::coTUIListener
    {
        using tui_check_box     = std::unique_ptr<opencover::coTUIToggleButton>;
        using tui_combo_box     = std::unique_ptr<opencover::coTUIComboBox>;
        using tui_frame         = std::unique_ptr<opencover::coTUIFrame>;
        using tui_int_edit      = std::unique_ptr<opencover::coTUIEditIntField>;
        using tui_label         = std::unique_ptr<opencover::coTUILabel>;
        using tui_tab           = std::unique_ptr<opencover::coTUITab>;

        using vrui_check_box    = std::unique_ptr<vrui::coCheckboxMenuItem>;
        using vrui_menu         = std::unique_ptr<vrui::coMenu>;
        using vrui_radio_button = std::unique_ptr<vrui::coCheckboxMenuItem>;
        using vrui_radio_group  = std::unique_ptr<vrui::coCheckboxGroup>;
        using vrui_slider       = std::unique_ptr<vrui::coSliderMenuItem>;
        using vrui_sub_menu     = std::unique_ptr<vrui::coSubMenuItem>;

        impl()
        {
            init_state_from_config();
            rend.update_state(state, dev_state);
        }

        osg::Node::NodeMask objroot_node_mask;
        osg::ref_ptr<osg::Geode> geode;
        renderer rend;

        struct
        {
            tui_tab             main_tab;
            tui_frame           general_frame;
            tui_label           general_label;
            tui_frame           algo_frame;
            tui_label           algo_label;
            tui_frame           device_frame;
            tui_label           device_label;
            tui_frame           dev_frame;
            tui_label           dev_label;

            // main menu
            tui_check_box       toggle_update_mode;
            tui_label           bounces_label;
            tui_int_edit        bounces_edit;

            // algo menu
            tui_combo_box       algo_box;

            // device menu
            tui_combo_box       device_box;

            // dev menu
            tui_check_box       suppress_rendering;
            tui_check_box       toggle_bvh_display;
            tui_check_box       toggle_bvh_costs_display;
            tui_check_box       toggle_geometric_normal_display;
            tui_check_box       toggle_shading_normal_display;
            tui_check_box       toggle_tex_coord_display;
        } tui;

        struct
        {
            vrui_menu           main_menu;
            vrui_menu           algo_menu;
            vrui_menu           device_menu;
            vrui_menu           dev_menu;
            vrui_sub_menu       main_menu_entry;
            vrui_sub_menu       algo_menu_entry;
            vrui_sub_menu       device_menu_entry;
            vrui_sub_menu       dev_menu_entry;

            // main menu
            vrui_check_box      toggle_update_mode;
            vrui_slider         bounces_slider;

            // algo menu
            vrui_radio_group    algo_group;
            vrui_radio_button   simple_button;
            vrui_radio_button   whitted_button;
            vrui_radio_button   pathtracing_button;

            // device menu
            vrui_radio_group    device_group;
            vrui_radio_button   cpu_button;
            vrui_radio_button   gpu_button;

            // dev menu
            vrui_check_box      suppress_rendering;
            vrui_check_box      toggle_bvh_display;
            vrui_radio_group    debug_kernel_group;
            vrui_check_box      toggle_bvh_costs_display;
            vrui_check_box      toggle_geometric_normal_display;
            vrui_check_box      toggle_shading_normal_display;
            vrui_check_box      toggle_tex_coord_display;
        } /*vr*/ui;

        std::shared_ptr<render_state> state;
        std::shared_ptr<debug_state> dev_state;

        // State before HQ mode was activated
        render_state temporary_state;
        bool hqmode = false;

        // init

        void init_state_from_config();
        // init both (vr)ui and tui
        void init_ui();

        // menu listener interface

        void menuEvent(vrui::coMenuItem *item);

        // tui listener interface

        void tabletEvent(opencover::coTUIElement *item);

        // control state

        void set_data_variance(data_variance var);
        void set_algorithm(algorithm algo);
        void set_num_bounces(unsigned num_bounces);
        void set_device(device_type dev);
        void set_suppress_rendering(bool suppress_rendering);
        void set_show_bvh(bool show_bvh);
        void set_show_bvh_costs(bool show_costs);
        void set_show_geometric_normals(bool show_geometric_normals);
        void set_show_shading_normals(bool show_shading_normals);
        void set_show_tex_coords(bool show_tex_coords);
    };

    //-------------------------------------------------------------------------------------------------
    // Read state from COVISE config
    //

    void Visionaray::impl::init_state_from_config()
    {

        //
        //
        // <Visionaray>
        //     <DataVariance value="static"  />                 <!-- "static" | "dynamic" -->
        //     <Algorithm    value="simple"  />                 <!-- "simple" | "whitted" -->
        //     <NumBounces   value="4" min="1" max="10" />      <!-- value:Integer | [min:Integer|max:Integer]  -->
        //     <Device       value="CPU"     />                 <!-- "CPU"    | "GPU"     -->
        //     <CPUScheduler numThreads="16" />                 <!-- numThreads:Integer   -->
        // </Visionaray>
        //
        //

        state = std::make_shared<render_state>();
        dev_state = std::make_shared<debug_state>();

        // Read config

        using boost::algorithm::to_lower;

        auto algo_str = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.Algorithm");
        auto num_bounces = covise::coCoviseConfig::getInt("value", "COVER.Plugin.Visionaray.NumBounces", 4);
        auto min_bounces = covise::coCoviseConfig::getInt("min", "COVER.Plugin.Visionaray.NumBounces", 1);
        auto max_bounces = covise::coCoviseConfig::getInt("max", "COVER.Plugin.Visionaray.NumBounces", 10);
        auto device_str = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.Device");
        auto data_var_str = covise::coCoviseConfig::getEntry("COVER.Plugin.Visionaray.DataVariance");
        auto num_threads = covise::coCoviseConfig::getInt("numThreads", "COVER.Plugin.Visionaray.CPUScheduler", 0);

        to_lower(algo_str);
        to_lower(device_str);
        to_lower(data_var_str);

        // Update state

        if (algo_str == "whitted")
        {
            state->algo = Whitted;
        }
        else if (algo_str == "pathtracing")
        {
            state->algo = Pathtracing;
        }
        else
        {
            state->algo = Simple;
        }

        // TODO
        //  assert( min_bounces <= num_bounces && num_bounces <= max_bounces );

        state->num_bounces = num_bounces;
        state->min_bounces = min_bounces;
        state->max_bounces = max_bounces;

        if (device_str == "gpu")
        {
            state->device = GPU;
        }
        else
        {
            state->device = CPU;
        }

        state->data_var = data_var_str == "dynamic" ? Dynamic : AnimationFrames;
        state->num_threads = num_threads;
    }

    void Visionaray::impl::init_ui()
    {
        using namespace opencover;
        using namespace vrui;

        ui.main_menu_entry.reset(new coSubMenuItem("Visionaray..."));
        opencover::cover->getMenu()->add(ui.main_menu_entry.get());

        // main menu --------------------------------------

        // TUI
        tui.main_tab.reset(new coTUITab("Visionaray", coVRTui::instance()->mainFolder->getID()));
        tui.main_tab->setPos(0, 0);

        tui.general_frame.reset(new coTUIFrame("General", tui.main_tab->getID()));
        tui.general_frame->setPos(0, 0);
        tui.general_label.reset(new coTUILabel("<b>General</b>", tui.general_frame->getID()));
        tui.general_label->setPos(0, 0);

        tui.toggle_update_mode.reset(new coTUIToggleButton("Update scene per frame", tui.general_frame->getID()));
        tui.toggle_update_mode->setEventListener(this);
        tui.toggle_update_mode->setState(state->data_var == Dynamic);
        tui.toggle_update_mode->setPos(0, 1);

        tui.bounces_label.reset(new coTUILabel("Number of bounces", tui.general_frame->getID()));
        tui.bounces_label->setPos(0, 3);

        tui.bounces_edit.reset(new coTUIEditIntField("Number of bounces", tui.general_frame->getID()));
        tui.bounces_edit->setEventListener(this);
        tui.bounces_edit->setMin(state->min_bounces);
        tui.bounces_edit->setMax(state->max_bounces);
        tui.bounces_edit->setValue(state->num_bounces);
        tui.bounces_edit->setPos(0, 4);

        // VRUI
        ui.main_menu.reset(new coRowMenu("Visionaray", cover->getMenu()));
        ui.main_menu_entry->setMenu(ui.main_menu.get());

        ui.toggle_update_mode.reset(new coCheckboxMenuItem("Update scene per frame", state->data_var == Dynamic));
        ui.toggle_update_mode->setMenuListener(this);
        ui.main_menu->add(ui.toggle_update_mode.get());

        ui.bounces_slider.reset(new coSliderMenuItem("Number of bounces", state->min_bounces, state->max_bounces, state->num_bounces));
        ui.bounces_slider->setInteger(true);
        ui.bounces_slider->setMenuListener(this);
        ui.main_menu->add(ui.bounces_slider.get());

        // algorithm submenu ------------------------------

        // TUI
        tui.algo_frame.reset(new coTUIFrame("Rendering algorithm", tui.main_tab->getID()));
        tui.algo_frame->setPos(1, 0);
        tui.algo_label.reset(new coTUILabel("<b>Rendering algorithm</b>", tui.algo_frame->getID()));
        tui.algo_label->setPos(0, 0);

        tui.algo_box.reset(new coTUIComboBox("Rendering algorithm", tui.algo_frame->getID()));
        tui.algo_box->setEventListener(this);
        tui.algo_box->addEntry("Simple");
        tui.algo_box->addEntry("Whitted");
        tui.algo_box->addEntry("Pathtracing");
        tui.algo_box->setSelectedEntry(state->algo == Simple ? 0 : state->algo == Whitted ? 1 : 2);
        tui.algo_box->setPos(0, 1);

        // VRUI
        ui.algo_menu_entry.reset(new coSubMenuItem("Rendering algorithm..."));
        ui.main_menu->add(ui.algo_menu_entry.get());

        ui.algo_menu.reset(new coRowMenu("Rendering algorithm", ui.main_menu.get()));
        ui.algo_menu_entry->setMenu(ui.algo_menu.get());

        ui.algo_group.reset(new coCheckboxGroup(/* allow empty selection: */ false));

        ui.simple_button.reset(new coCheckboxMenuItem("Simple", state->algo == Simple, ui.algo_group.get()));
        ui.simple_button->setMenuListener(this);
        ui.algo_menu->add(ui.simple_button.get());

        ui.whitted_button.reset(new coCheckboxMenuItem("Whitted", state->algo == Whitted, ui.algo_group.get()));
        ui.whitted_button->setMenuListener(this);
        ui.algo_menu->add(ui.whitted_button.get());

        ui.pathtracing_button.reset(new coCheckboxMenuItem("Pathtracing", state->algo == Pathtracing, ui.algo_group.get()));
        ui.pathtracing_button->setMenuListener(this);
        ui.algo_menu->add(ui.pathtracing_button.get());

        // device submenu ---------------------------------

        // TUI
        tui.device_frame.reset(new coTUIFrame("Device", tui.main_tab->getID()));
        tui.device_frame->setPos(2, 0);
        tui.device_label.reset(new coTUILabel("<b>Device</b>", tui.device_frame->getID()));
        tui.device_label->setPos(0, 0);

        tui.device_box.reset(new coTUIComboBox("Device", tui.device_frame->getID()));
        tui.device_box->setEventListener(this);
        tui.device_box->addEntry("CPU");
        tui.device_box->addEntry("GPU");
        tui.device_box->setSelectedEntry(state->device == CPU ? 0 : 1);
        tui.device_box->setPos(0, 1);

        // VRUI
        ui.device_menu_entry.reset(new coSubMenuItem("Device..."));
        ui.main_menu->add(ui.device_menu_entry.get());

        ui.device_menu.reset(new coRowMenu("Device", ui.main_menu.get()));
        ui.device_menu_entry->setMenu(ui.device_menu.get());

        ui.device_group.reset(new coCheckboxGroup(/* allow empty selection: */ false));

        ui.cpu_button.reset(new coCheckboxMenuItem("CPU", state->device == CPU, ui.device_group.get()));
        ui.cpu_button->setMenuListener(this);
        ui.device_menu->add(ui.cpu_button.get());

        ui.gpu_button.reset(new coCheckboxMenuItem("GPU", state->device == GPU, ui.device_group.get()));
        ui.gpu_button->setMenuListener(this);
        ui.device_menu->add(ui.gpu_button.get());

        // dev submenu at the bottom! ---------------------

        if (dev_state->debug_mode)
        {
            // TUI
            tui.dev_frame.reset(new coTUIFrame("Developer", tui.main_tab->getID()));
            tui.dev_frame->setPos(3, 0);
            tui.dev_label.reset(new coTUILabel("<b>Developer</b>", tui.dev_frame->getID()));
            tui.dev_label->setPos(0, 0);

            tui.suppress_rendering.reset(new coTUIToggleButton("Suppress rendering with Visionaray", tui.dev_frame->getID()));
            tui.suppress_rendering->setEventListener(this);
            tui.suppress_rendering->setState(false);
            tui.suppress_rendering->setPos(0, 1);

            tui.toggle_bvh_display.reset(new coTUIToggleButton("Show BVH outlines", tui.dev_frame->getID()));
            tui.toggle_bvh_display->setEventListener(this);
            tui.toggle_bvh_display->setState(false);
            tui.toggle_bvh_display->setPos(0, 2);

            tui.toggle_bvh_costs_display.reset(new coTUIToggleButton("Show BVH traversal costs", tui.dev_frame->getID()));
            tui.toggle_bvh_costs_display->setEventListener(this);
            tui.toggle_bvh_costs_display->setState(false);
            tui.toggle_bvh_costs_display->setPos(0, 3);

            tui.toggle_geometric_normal_display.reset(new coTUIToggleButton("Show geometric normals", tui.dev_frame->getID()));
            tui.toggle_geometric_normal_display->setEventListener(this);
            tui.toggle_geometric_normal_display->setState(false);
            tui.toggle_geometric_normal_display->setPos(0, 4);

            tui.toggle_shading_normal_display.reset(new coTUIToggleButton("Show shading normals", tui.dev_frame->getID()));
            tui.toggle_shading_normal_display->setEventListener(this);
            tui.toggle_shading_normal_display->setState(false);
            tui.toggle_shading_normal_display->setPos(0, 5);

            tui.toggle_tex_coord_display.reset(new coTUIToggleButton("Show texture coordinates", tui.dev_frame->getID()));
            tui.toggle_tex_coord_display->setEventListener(this);
            tui.toggle_tex_coord_display->setState(false);
            tui.toggle_tex_coord_display->setPos(0, 6);

            // VRUI
            ui.dev_menu_entry.reset(new coSubMenuItem("Developer..."));
            ui.main_menu->add(ui.dev_menu_entry.get());

            ui.dev_menu.reset(new coRowMenu("Developer", ui.main_menu.get()));
            ui.dev_menu_entry->setMenu(ui.dev_menu.get());

            ui.suppress_rendering.reset(new coCheckboxMenuItem("Suppress rendering with Visionaray", false));
            ui.suppress_rendering->setMenuListener(this);
            ui.dev_menu->add(ui.suppress_rendering.get());

            ui.toggle_bvh_display.reset(new coCheckboxMenuItem("Show BVH outlines", false));
            ui.toggle_bvh_display->setMenuListener(this);
            ui.dev_menu->add(ui.toggle_bvh_display.get());

            ui.debug_kernel_group.reset(new coCheckboxGroup(/* allow empty selection: */ true));

            ui.toggle_bvh_costs_display.reset(new coCheckboxMenuItem("Show BVH traversal costs", false, ui.debug_kernel_group.get()));
            ui.toggle_bvh_costs_display->setMenuListener(this);
            ui.dev_menu->add(ui.toggle_bvh_costs_display.get());

            ui.toggle_geometric_normal_display.reset(new coCheckboxMenuItem("Show geometric normals", false, ui.debug_kernel_group.get()));
            ui.toggle_geometric_normal_display->setMenuListener(this);
            ui.dev_menu->add(ui.toggle_geometric_normal_display.get());

            ui.toggle_shading_normal_display.reset(new coCheckboxMenuItem("Show shading normals", false, ui.debug_kernel_group.get()));
            ui.toggle_shading_normal_display->setMenuListener(this);
            ui.dev_menu->add(ui.toggle_shading_normal_display.get());

            ui.toggle_tex_coord_display.reset(new coCheckboxMenuItem("Show texture coordinates", false, ui.debug_kernel_group.get()));
            ui.toggle_tex_coord_display->setMenuListener(this);
            ui.dev_menu->add(ui.toggle_tex_coord_display.get());
        }
    }

    void Visionaray::impl::menuEvent(vrui::coMenuItem *item)
    {
        // main menu
        if (item == ui.toggle_update_mode.get())
        {
            set_data_variance(ui.toggle_update_mode->getState() ? Dynamic : Static);
        }

        if (item == ui.bounces_slider.get())
        {
            set_num_bounces(ui.bounces_slider->getValue());
        }

        // algorithm submenu
        if (item == ui.simple_button.get())
        {
            set_algorithm(Simple);
        }
        else if (item == ui.whitted_button.get())
        {
            set_algorithm(Whitted);
        }
        else if (item == ui.pathtracing_button.get())
        {
            set_algorithm(Pathtracing);
        }

        // device submenu
        if (item == ui.cpu_button.get())
        {
            set_device(CPU);
        }
        else if (item == ui.gpu_button.get())
        {
            set_device(GPU);
        }

        // dev submenu
        if (item == ui.suppress_rendering.get())
        {
            set_suppress_rendering(ui.suppress_rendering->getState());
        }

        if (item == ui.toggle_bvh_display.get())
        {
            set_show_bvh(ui.toggle_bvh_display->getState());
        }

        if (item == ui.toggle_bvh_costs_display.get())
        {
            set_show_bvh_costs(ui.toggle_bvh_costs_display->getState());
        }
        else if (item == ui.toggle_geometric_normal_display.get())
        {
            set_show_geometric_normals(ui.toggle_geometric_normal_display->getState());
        }
        else if (item == ui.toggle_shading_normal_display.get())
        {
            set_show_shading_normals(ui.toggle_shading_normal_display->getState());
        }
        else if (item == ui.toggle_tex_coord_display.get())
        {
            set_show_tex_coords(ui.toggle_tex_coord_display->getState());
        }
    }

    void Visionaray::impl::tabletEvent(opencover::coTUIElement *item)
    {
        // main menu
        if (item == tui.toggle_update_mode.get())
        {
            set_data_variance(tui.toggle_update_mode->getState() ? Dynamic : Static);
        }

        if (item == tui.bounces_edit.get())
        {
            set_num_bounces(tui.bounces_edit->getValue());
        }

        // algorithm submenu
        if (item == tui.algo_box.get() && tui.algo_box->getSelectedEntry() == 0)
        {
            set_algorithm(Simple);
        }
        else if (item == tui.algo_box.get() && tui.algo_box->getSelectedEntry() == 1)
        {
            set_algorithm(Whitted);
        }
        else if (item == tui.algo_box.get() && tui.algo_box->getSelectedEntry() == 2)
        {
            set_algorithm(Pathtracing);
        }

        // device submenu
        if (item == tui.device_box.get() && tui.device_box->getSelectedEntry() == 0)
        {
            set_device(CPU);
        }
        else if (item == tui.device_box.get() && tui.device_box->getSelectedEntry() == 1)
        {
            set_device(GPU);
        }

        // dev submenu
        if (item == tui.suppress_rendering.get())
        {
            set_suppress_rendering(tui.suppress_rendering->getState());
        }

        if (item == tui.toggle_bvh_display.get())
        {
            set_show_bvh(tui.toggle_bvh_display->getState());
        }

        if (item == tui.toggle_bvh_costs_display.get())
        {
            set_show_bvh_costs(tui.toggle_bvh_costs_display->getState());
        }
        else if (item == tui.toggle_geometric_normal_display.get())
        {
            set_show_geometric_normals(tui.toggle_geometric_normal_display->getState());
        }
        else if (item == tui.toggle_shading_normal_display.get())
        {
            set_show_shading_normals(tui.toggle_shading_normal_display->getState());
        }
        else if (item == tui.toggle_tex_coord_display.get())
        {
            set_show_tex_coords(tui.toggle_tex_coord_display->getState());
        }
    }

    //-------------------------------------------------------------------------------------------------
    // Control state
    //

    void Visionaray::impl::set_data_variance(data_variance var)
    {
        state->data_var = var;
        ui.toggle_update_mode->setState(var == Dynamic, false);
        tui.toggle_update_mode->setState(var == Dynamic);
    }

    void Visionaray::impl::set_algorithm(algorithm algo)
    {
        state->algo = algo;
        ui.simple_button->setState(algo == Simple, false);
        ui.whitted_button->setState(algo == Whitted, false);
        ui.pathtracing_button->setState(algo == Pathtracing, false);
        tui.algo_box->setSelectedEntry(state->algo == Simple ? 0 : state->algo == Whitted ? 1 : 2);
    }

    void Visionaray::impl::set_num_bounces(unsigned num_bounces)
    {
        state->num_bounces = num_bounces;
        ui.bounces_slider->setValue(num_bounces);
        tui.bounces_edit->setValue(num_bounces);
    }

    void Visionaray::impl::set_device(device_type dev)
    {
        state->device = dev;
        ui.cpu_button->setState(dev == CPU, false);
        ui.gpu_button->setState(dev == GPU, false);
        tui.device_box->setSelectedEntry(dev == CPU ? 0 : 1);
    }

    void Visionaray::impl::set_suppress_rendering(bool suppress_rendering)
    {
        rend.set_suppress_rendering(suppress_rendering);
        ui.suppress_rendering->setState(suppress_rendering, false);
        tui.suppress_rendering->setState(suppress_rendering);
    }

    void Visionaray::impl::set_show_bvh(bool show_bvh)
    {
        dev_state->show_bvh = show_bvh;
        ui.toggle_bvh_display->setState(show_bvh, false);
        tui.toggle_bvh_display->setState(show_bvh);
    }

    void Visionaray::impl::set_show_bvh_costs(bool show_costs)
    {
        dev_state->show_bvh_costs = show_costs;
        ui.toggle_bvh_costs_display->setState(show_costs, false);
        tui.toggle_bvh_costs_display->setState(show_costs);
    }

    void Visionaray::impl::set_show_geometric_normals(bool show_geometric_normals)
    {
        dev_state->show_geometric_normals = show_geometric_normals;
        ui.toggle_geometric_normal_display->setState(show_geometric_normals, false);
        tui.toggle_geometric_normal_display->setState(show_geometric_normals);
    }

    void Visionaray::impl::set_show_shading_normals(bool show_shading_normals)
    {
        dev_state->show_shading_normals = show_shading_normals;
        ui.toggle_shading_normal_display->setState(show_shading_normals, false);
        tui.toggle_shading_normal_display->setState(show_shading_normals);
    }

    void Visionaray::impl::set_show_tex_coords(bool show_tex_coords)
    {
        dev_state->show_tex_coords = show_tex_coords;
        ui.toggle_tex_coord_display->setState(show_tex_coords, false);
        tui.toggle_tex_coord_display->setState(show_tex_coords);
    }

    //-------------------------------------------------------------------------------------------------
    // Visionaray plugin
    //

    Visionaray::Visionaray()
    : coVRPlugin(COVER_PLUGIN_NAME)
    , impl_(new impl)
    {
    }

    Visionaray::~Visionaray()
    {
        opencover::cover->getObjectsRoot()->setNodeMask(impl_->objroot_node_mask);
    }

    bool Visionaray::init()
    {
        using namespace osg;

        opencover::VRViewer::instance()->culling(false);

        std::cout << "Init Visionaray Plugin!!" << std::endl;

        impl_->init_ui();

        impl_->objroot_node_mask = opencover::cover->getObjectsRoot()->getNodeMask();

        return true;
    }

    void Visionaray::addNode(osg::Node *node, const opencover::RenderObject *obj)
    {
        impl_->state->rebuild = true;
    }

    void Visionaray::removeNode(osg::Node *node, bool isGroup, osg::Node *realNode)
    {
        impl_->state->rebuild = true;
    }

    void Visionaray::preDraw(osg::RenderInfo &info)
    {
        if (impl_->state->data_var == Dynamic)
            impl_->state->rebuild = true;

        if (impl_->state->rebuild && opencover::OpenCOVER::instance()->initDone())
        {
            auto seqs = opencover::coVRAnimationManager::instance()->getSequences();
            std::vector<osg::Sequence *> osg_seqs;
            osg_seqs.reserve(seqs.size());
            for (const auto &s: seqs)
            {
                osg_seqs.emplace_back(s.seq.get());
            }
            impl_->rend.acquire_scene_data(osg_seqs);
            impl_->state->rebuild = false;
        }

        impl_->state->animation_frame = opencover::coVRAnimationManager::instance()->getAnimationFrame();
        impl_->rend.render_frame(info);
    }

    void Visionaray::expandBoundingSphere(osg::BoundingSphere &bs)
    {
        impl_->rend.expandBoundingSphere(bs);
    }

    void Visionaray::key(int type, int key_sym, int /* mod */)
    {
        if (type == osgGA::GUIEventAdapter::KEYDOWN)
        {
            switch (key_sym)
            {
            case '1':
                impl_->set_algorithm(Simple);
                break;

            case '2':
                impl_->set_algorithm(Whitted);
                break;

            case '3':
                impl_->set_algorithm(Pathtracing);
                break;
            }
        }
    }

    bool Visionaray::update()
    {
        // TODO: store and restore old head tracking state
        if (opencover::cover->isHighQuality())
        {
            if (!impl_->hqmode)
            {
                impl_->temporary_state = *impl_->state;
                impl_->hqmode = true;
            }
            impl_->state->algo = Pathtracing;
            impl_->state->num_bounces = 5;
            opencover::VRSceneGraph::instance()->toggleHeadTracking(false);
        }
        else
        {
            if (impl_->hqmode)
            {
                impl_->hqmode = false;
                impl_->state->algo = impl_->temporary_state.algo;
                impl_->state->num_bounces = impl_->temporary_state.num_bounces;
                opencover::VRSceneGraph::instance()->toggleHeadTracking(true);
            }
        }

        return true;
    }

} // namespace visionaray

COVERPLUGIN(visionaray::Visionaray)
