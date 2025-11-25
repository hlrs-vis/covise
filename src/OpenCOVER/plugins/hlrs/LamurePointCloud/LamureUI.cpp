#include "LamureUI.h"
#include "Lamure.h"
#include "LamureEditTool.h"
#include <filesystem>
#include <lamure/ren/policy.h>
#include <lamure/ren/config.h>
#include <algorithm>
#include <iostream>

LamureUI::LamureUI(Lamure* plugin, const std::string& name) : m_plugin(plugin) 
{

}

LamureUI::~LamureUI() 
{

}

void LamureUI::setupUi() {
    m_lamure_menu = new opencover::ui::Menu("Lamure", m_plugin);
    m_lamure_menu->setText("Lamure");
    m_lamure_menu->allowRelayout(true);

    // Rendering visibility toggles
    m_rendering_group = new opencover::ui::Group(m_lamure_menu, "Rendering");

    m_pointcloud_button  = new opencover::ui::Button(m_rendering_group, "Pointcloud");
    m_boundingbox_button = new opencover::ui::Button(m_rendering_group, "BoundingBoxes");
    m_frustum_button     = new opencover::ui::Button(m_rendering_group, "Frustum");
    m_text_button        = new opencover::ui::Button(m_rendering_group, "Text");

    m_pointcloud_button->setShared(true);
    m_boundingbox_button->setShared(true);
    m_frustum_button->setShared(true);
    m_text_button->setShared(true);


    // Miscellaneous options
    m_misc_group = new opencover::ui::Group(m_lamure_menu, "Misc");

    m_sync_button        = new opencover::ui::Button(m_misc_group, "Sync");
    m_notify_button      = new opencover::ui::Button(m_misc_group, "Notify");
    m_dump_button        = new opencover::ui::Button(m_misc_group, "Dump");

    m_sync_button->setShared(true);
    m_notify_button->setShared(true);
    m_dump_button->setShared(true);

    // Keep notify setting in sync with UI
    m_notify_button->setState(m_plugin->getSettings().show_notify);
    m_notify_button->setCallback([this](bool on){
        m_plugin->getSettings().show_notify = on;
    });

    // Edit Menu
    m_edit_menu = new opencover::ui::Menu(m_lamure_menu, "Edit");
    m_edit_menu->setText("Edit");

    m_edit_button = new opencover::ui::Button(m_edit_menu, "EditMode");
    m_edit_button->setShared(true);
    m_edit_button->setState(m_plugin->isEditModeActive());
    m_edit_button->setCallback([this](bool on){
        m_plugin->setEditMode(on);
    });

    // Edit brush controls (sphere-only)
    m_edit_group = new opencover::ui::Group(m_edit_menu, "EditBrush");
    m_edit_group->setText("Brush");
    // Brush shape is fixed to sphere; no UI toggle needed.

    m_edit_action_move = new opencover::ui::Button(m_edit_group, "BrushMove");
    m_edit_action_move->setText("Move");
    m_edit_action_move->setShared(true);
    m_edit_action_erase = new opencover::ui::Button(m_edit_group, "BrushErase");
    m_edit_action_erase->setText("Erase");
    m_edit_action_erase->setShared(true);
    m_edit_action_restore = new opencover::ui::Button(m_edit_group, "BrushRestore");
    m_edit_action_restore->setText("Restore");
    m_edit_action_restore->setShared(true);

    auto selectAction = [this](LamureEditTool::BrushAction action) {
        m_plugin->setEditAction(action);
        if (m_edit_action_move)    m_edit_action_move->setState(action == LamureEditTool::BrushAction::None);
        if (m_edit_action_erase)   m_edit_action_erase->setState(action == LamureEditTool::BrushAction::Erase);
        if (m_edit_action_restore) m_edit_action_restore->setState(action == LamureEditTool::BrushAction::Restore);
    };

    selectAction(m_plugin->getEditAction());

    m_edit_action_move->setCallback([selectAction](bool on){
        if (on) selectAction(LamureEditTool::BrushAction::None);
    });
    m_edit_action_erase->setCallback([selectAction](bool on){
        if (on) selectAction(LamureEditTool::BrushAction::Erase);
    });
    m_edit_action_restore->setCallback([selectAction](bool on){
        if (on) selectAction(LamureEditTool::BrushAction::Restore);
    });

    // Primitives group (as before)
    m_primitives_group = new opencover::ui::Group(m_lamure_menu, "Primitives");

    m_point_button = new opencover::ui::Button(m_primitives_group, "point_button");
    m_point_button->setText("Point");
    m_point_button->setShared(true);

    m_surfel_button = new opencover::ui::Button(m_primitives_group, "surfel_button");
    m_surfel_button->setText("Surfel");
    m_surfel_button->setShared(true);
    m_surfel_button->setState(m_plugin->getSettings().surfel);

    m_splat_button = new opencover::ui::Button(m_primitives_group, "splat_button");
    m_splat_button->setText("Splatting");
    m_splat_button->setShared(true);
    m_splat_button->setState(m_plugin->getSettings().splatting);


    m_lod_menu = new opencover::ui::Menu(m_lamure_menu, "LOD");

    m_lod_button = new opencover::ui::Button(m_lod_menu, "LOD");
    m_lod_button->setText("LOD");
    m_lod_button->setShared(true);
    m_lod_button->setState(m_plugin->getSettings().lod_update);
    m_lod_button->setCallback([this](bool state) {
        m_plugin->getSettings().lod_update = state;
        });

    m_lod_group = new opencover::ui::Group(m_lod_menu, "");

    m_lod_error_slider = new opencover::ui::Slider(m_lod_group, "lod_error");
    m_lod_error_slider->setText("LOD Error");
    m_lod_error_slider->setBounds(LAMURE_MIN_THRESHOLD, LAMURE_MAX_THRESHOLD);
    m_lod_error_slider->setValue(m_plugin->getSettings().lod_error);
    m_lod_error_slider->setShared(true);
    m_lod_error_slider->setCallback([this](double value, bool released) {
        m_plugin->getSettings().lod_error = static_cast<float>(value);
        });

    m_scaling_menu = new opencover::ui::Menu(m_lamure_menu, "Scaling");

    // Anisotropic scaling mode (Off / Auto / On) at top of Scaling
    //m_aniso_mode_group = new opencover::ui::Group(m_scaling_menu, "");
    m_aniso_auto_btn = new opencover::ui::Button(m_scaling_menu, "Auto");
    m_aniso_off_btn  = new opencover::ui::Button(m_scaling_menu, "Off");
    m_aniso_on_btn   = new opencover::ui::Button(m_scaling_menu, "On");
    m_aniso_off_btn->setShared(true);
    m_aniso_auto_btn->setShared(true);
    m_aniso_on_btn->setShared(true);

    // Exclusive selection helper
    auto selectAniso = [this](int idx){
        auto &st = m_plugin->getSettings();
        // idx: 0=Off, 1=Auto, 2=On
        switch (idx) {
            case 0: st.anisotropic_surfel_scaling = 0; break;
            case 2: st.anisotropic_surfel_scaling = 2; break;
            default: st.anisotropic_surfel_scaling = 1; break;
        }
        if (m_aniso_off_btn)  m_aniso_off_btn->setState(idx == 0);
        if (m_aniso_auto_btn) m_aniso_auto_btn->setState(idx == 1);
        if (m_aniso_on_btn)   m_aniso_on_btn->setState(idx == 2);
    };

    // Initial state from settings
    {
        int v = std::max(0, std::min(2, m_plugin->getSettings().anisotropic_surfel_scaling));
        int idx = (v == 0 ? 0 : (v == 2 ? 2 : 1));
        selectAniso(idx);
    }

    // Callbacks enforcing radio behavior
    m_aniso_off_btn->setCallback([this, selectAniso](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (st.anisotropic_surfel_scaling == 0) m_aniso_off_btn->setState(true);
            return;
        }
        selectAniso(0);
    });
    m_aniso_auto_btn->setCallback([this, selectAniso](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (st.anisotropic_surfel_scaling == 1) m_aniso_auto_btn->setState(true);
            return;
        }
        selectAniso(1);
    });
    m_aniso_on_btn->setCallback([this, selectAniso](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (st.anisotropic_surfel_scaling == 2) m_aniso_on_btn->setState(true);
            return;
        }
        selectAniso(2);
    });

    // Scaling sliders (grouped, similar to LOD)
    m_scaling_group = new opencover::ui::Group(m_scaling_menu, "");
    m_scale_radius_slider = new opencover::ui::Slider(m_scaling_group, "scale_radius");
    m_scale_radius_slider->setText("Scale Radius");
    m_scale_radius_slider->setBounds(0.0001f, 1.0f);
    m_scale_radius_slider->setScale(opencover::ui::Slider::Logarithmic);
    //m_scale_radius_slider->setBounds(0.0f, 3.0f);
    m_scale_radius_slider->setValue(
        m_plugin->getSettings().scale_radius
    );
    m_scale_radius_slider->setShared(true);
    m_scale_radius_slider->setCallback([this](double value, bool released)
        { m_plugin->getSettings().scale_radius = static_cast<float>(value); });

    m_scale_radius_gamma_slider = new opencover::ui::Slider(m_scaling_group, "scale_radius_gamma");
    m_scale_radius_gamma_slider->setText("Scale Gamma");
    m_scale_radius_gamma_slider->setBounds(0.0f, 1.0f);
    m_scale_radius_gamma_slider->setValue(m_plugin->getSettings().scale_radius_gamma);
    m_scale_radius_gamma_slider->setShared(true);
    m_scale_radius_gamma_slider->setCallback([this](double value, bool released)
        { m_plugin->getSettings().scale_radius_gamma = static_cast<float>(value); });

    m_max_radius_cut_slider = new opencover::ui::Slider(m_scaling_group, "max_radius_cut");
    m_max_radius_cut_slider->setText("Cut Radius (world)");
    m_max_radius_cut_slider->setBounds(0.0f, std::max(m_plugin->getSettings().max_radius_cut, 10.0f));
    m_max_radius_cut_slider->setValue(m_plugin->getSettings().max_radius_cut);
    m_max_radius_cut_slider->setShared(true);
    m_max_radius_cut_slider->setCallback([this](double value, bool released)
        { m_plugin->getSettings().max_radius_cut = static_cast<float>(value); });

    m_max_radius_slider = new opencover::ui::Slider(m_scaling_group, "max_radius");
    m_max_radius_slider->setText("Max. Radius (world)");
    m_max_radius_slider->setBounds(0.0f, std::max(m_plugin->getSettings().max_radius, 3.0f));
    m_max_radius_slider->setValue(m_plugin->getSettings().max_radius);
    m_max_radius_slider->setShared(true);
    m_max_radius_slider->setCallback([this](double value, bool released)
        { m_plugin->getSettings().max_radius = static_cast<float>(value); });

    m_min_radius_slider = new opencover::ui::Slider(m_scaling_group, "min_radius");
    m_min_radius_slider->setText("Min. Radius (world)");
    m_min_radius_slider->setBounds(0.0f, std::max(m_plugin->getSettings().min_radius, 0.1f));
    m_min_radius_slider->setValue(m_plugin->getSettings().min_radius);
    m_min_radius_slider->setShared(true);
    m_min_radius_slider->setCallback([this](double value, bool /*released*/) {
        m_plugin->getSettings().min_radius = static_cast<float>(value);
        });

    m_max_screen_size_slider = new opencover::ui::Slider(m_scaling_group, "max_screen_size");
    m_max_screen_size_slider->setText("Max. Radius (screen)");
    m_max_screen_size_slider->setBounds(1.0f, std::max(m_plugin->getSettings().max_screen_size, 10000.0f));
    m_max_screen_size_slider->setScale(opencover::ui::Slider::Logarithmic);
    m_max_screen_size_slider->setValue(m_plugin->getSettings().max_screen_size);
    m_max_screen_size_slider->setShared(true);
    m_max_screen_size_slider->setCallback([this](double value, bool released)
        { m_plugin->getSettings().max_screen_size = static_cast<float>(value); });

    m_min_screen_size_slider = new opencover::ui::Slider(m_scaling_group, "min_screen_size");
    m_min_screen_size_slider->setText("Min. Radius (screen)");
    m_min_screen_size_slider->setBounds(0.0f, std::max(m_plugin->getSettings().min_screen_size, 10.0f));
    m_min_screen_size_slider->setValue(m_plugin->getSettings().min_screen_size);
    m_min_screen_size_slider->setShared(true);
    m_min_screen_size_slider->setCallback([this](double value, bool /*released*/) {
        m_plugin->getSettings().min_screen_size = static_cast<float>(value);
        });

    m_scale_surfel_slider = new opencover::ui::Slider(m_scaling_group, "scale_surfel");
    m_scale_surfel_slider->setText("Surfel Scale Multiplier");
    m_scale_surfel_slider->setBounds(0.1f, 5.0f);
    m_scale_surfel_slider->setValue(m_plugin->getSettings().scale_surfel);
    m_scale_surfel_slider->setShared(true);
    m_scale_surfel_slider->setCallback([this](double v, bool released) {
        auto &st = m_plugin->getSettings();
        st.scale_surfel = static_cast<float>(v);
        if (st.surfel || st.splatting)
            st.scale_element = st.scale_surfel;
        });

    // (SelectionList removed in favor of atomic checkboxes)

    // --- Multi-Pass Blending Sliders ---
    m_depth_range_slider = new opencover::ui::Slider(m_scaling_group, "depth_range");
    m_depth_range_slider->setText("Depth Range");
    m_depth_range_slider->setBounds(0.0f, 10.0f);
    m_depth_range_slider->setValue(m_plugin->getSettings().depth_range);
    m_depth_range_slider->setShared(true);
    m_depth_range_slider->setCallback([this](double value, bool released) {
        m_plugin->getSettings().depth_range = static_cast<float>(value);
        });

    m_flank_lift_slider = new opencover::ui::Slider(m_scaling_group, "flank_lift");
    m_flank_lift_slider->setText("Flank Lift");
    m_flank_lift_slider->setBounds(0.0f, 1.0f);
    m_flank_lift_slider->setValue(m_plugin->getSettings().flank_lift);
    m_flank_lift_slider->setShared(true);
    m_flank_lift_slider->setCallback([this](double value, bool released) {
        m_plugin->getSettings().flank_lift = static_cast<float>(value);
        });


    // Coloring options
    m_coloring_menu = new opencover::ui::Menu(m_lamure_menu, "Coloring");
    m_coloring_button = new opencover::ui::Button(m_coloring_menu, "coloring_button");
    m_coloring_button->setText("Coloring");
    m_coloring_button->setShared(true);
    m_coloring_button->setState(m_plugin->getSettings().coloring);

    m_mode_group = new opencover::ui::Group(m_coloring_menu, "Modes");

    m_mode_normals_btn = new opencover::ui::Button(m_mode_group, "Normals");
    m_mode_accuracy_btn = new opencover::ui::Button(m_mode_group, "Accuracy");
    m_mode_radius_dev_btn = new opencover::ui::Button(m_mode_group, "RadiusDeviation");
    m_mode_output_sens_btn = new opencover::ui::Button(m_mode_group, "OutputSensitivity");

    m_mode_normals_btn->setShared(true);
    m_mode_accuracy_btn->setShared(true);
    m_mode_radius_dev_btn->setShared(true);
    m_mode_output_sens_btn->setShared(true);

    // Exclusive mode selection (UI + Settings)
    auto selectMode = [this](int idx){
        auto &st = m_plugin->getSettings();
        st.show_normals            = (idx == 0);
        st.show_accuracy           = (idx == 1);
        st.show_radius_deviation   = (idx == 2);
        st.show_output_sensitivity = (idx == 3);

        // UI spiegeln
        if (m_mode_normals_btn)      m_mode_normals_btn->setState(idx == 0);
        if (m_mode_accuracy_btn)     m_mode_accuracy_btn->setState(idx == 1);
        if (m_mode_radius_dev_btn)   m_mode_radius_dev_btn->setState(idx == 2);
        if (m_mode_output_sens_btn)  m_mode_output_sens_btn->setState(idx == 3);
        };

    // Initialize mode from settings (fallback: Normals)
    auto &s = m_plugin->getSettings();
    int initial_mode_idx = 0;
    if      (s.show_accuracy)           initial_mode_idx = 1;
    else if (s.show_radius_deviation)   initial_mode_idx = 2;
    else if (s.show_output_sensitivity) initial_mode_idx = 3;
    selectMode(initial_mode_idx);

    // Radio-style callbacks (ensure at least one active)
    m_mode_normals_btn->setCallback([this, selectMode](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) { // verhindern, dass gar nichts aktiv ist
            if (!st.show_accuracy && !st.show_radius_deviation && !st.show_output_sensitivity)
                m_mode_normals_btn->setState(true);
            return;
        }
        selectMode(0);
        });

    m_mode_accuracy_btn->setCallback([this, selectMode](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (!st.show_normals && !st.show_radius_deviation && !st.show_output_sensitivity)
                m_mode_accuracy_btn->setState(true);
            return;
        }
        selectMode(1);
        });

    m_mode_radius_dev_btn->setCallback([this, selectMode](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (!st.show_normals && !st.show_accuracy && !st.show_output_sensitivity)
                m_mode_radius_dev_btn->setState(true);
            return;
        }
        selectMode(2);
        });

    m_mode_output_sens_btn->setCallback([this, selectMode](bool on){
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (!st.show_normals && !st.show_accuracy && !st.show_radius_deviation)
                m_mode_output_sens_btn->setState(true);
            return;
        }
        selectMode(3);
        });

    m_lighting_menu = new opencover::ui::Menu(m_lamure_menu, "Lighting");

    m_lighting_button = new opencover::ui::Button(m_lighting_menu, "lighting_button");
    m_lighting_button->setText("Lighting");
    m_lighting_button->setShared(true);
    m_lighting_button->setState(m_plugin->getSettings().lighting);

    m_lighting_group = new opencover::ui::Group(m_lighting_menu, "");

    m_point_light_intensity_slider = new opencover::ui::Slider(m_lighting_group, "point_light_intensity");
    m_point_light_intensity_slider->setText("Point Light Intensity");
    m_point_light_intensity_slider->setBounds(0, 1.0);
    m_point_light_intensity_slider->setValue(m_plugin->getSettings().point_light_intensity);
    m_point_light_intensity_slider->setShared(true);
    m_point_light_intensity_slider->setCallback([this](double value, bool) { 
        m_plugin->getSettings().point_light_intensity = (float)value;
        });

    m_ambient_intensity_slider = new opencover::ui::Slider(m_lighting_group, "ambient_intensity");
    m_ambient_intensity_slider->setText("Ambient Light");
    m_ambient_intensity_slider->setBounds(0.0, 1.0);
    m_ambient_intensity_slider->setValue(m_plugin->getSettings().ambient_intensity);
    m_ambient_intensity_slider->setShared(true);
    m_ambient_intensity_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().ambient_intensity = (float)value;
        });

    m_specular_intenity_slider = new opencover::ui::Slider(m_lighting_group, "specular_intensity");
    m_specular_intenity_slider->setText("Specular Intensity");
    m_specular_intenity_slider->setBounds(0.0, 1.0);
    m_specular_intenity_slider->setValue(m_plugin->getSettings().specular_intensity);
    m_specular_intenity_slider->setShared(true);
    m_specular_intenity_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().specular_intensity = (float)value;
        });

    m_shininess_slider = new opencover::ui::Slider(m_lighting_group, "shininess");
    m_shininess_slider->setText("Shininess");
    m_shininess_slider->setBounds(0.0, 1.0);
    m_shininess_slider->setValue(m_plugin->getSettings().shininess);
    m_shininess_slider->setShared(true);
    m_shininess_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().shininess = static_cast<float>(value);
        });

    m_gamma_slider = new opencover::ui::Slider(m_lighting_group, "gamma");
    m_gamma_slider->setText("Gamma");
    m_gamma_slider->setBounds(0.0, 3.0);
    m_gamma_slider->setValue(m_plugin->getSettings().gamma);
    m_gamma_slider->setShared(true);
    m_gamma_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().gamma = static_cast<float>(value);
        });

    // Light Position Sliders
    m_light_pos_x_slider = new opencover::ui::Slider(m_lighting_group, "light_pos_x");
    m_light_pos_x_slider->setText("Light Pos X");
    m_light_pos_x_slider->setBounds(-1000.0, 1000.0);
    m_light_pos_x_slider->setValue(m_plugin->getSettings().point_light_pos.x);
    m_light_pos_x_slider->setShared(true);
    m_light_pos_x_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().point_light_pos.x = static_cast<float>(value);
        });

    m_light_pos_y_slider = new opencover::ui::Slider(m_lighting_group, "light_pos_y");
    m_light_pos_y_slider->setText("Light Pos Y");
    m_light_pos_y_slider->setBounds(-1000.0, 1000.0);
    m_light_pos_y_slider->setValue(m_plugin->getSettings().point_light_pos.y);
    m_light_pos_y_slider->setShared(true);
    m_light_pos_y_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().point_light_pos.y = static_cast<float>(value);
        });

    m_light_pos_z_slider = new opencover::ui::Slider(m_lighting_group, "light_pos_z");
    m_light_pos_z_slider->setText("Light Pos Z");
    m_light_pos_z_slider->setBounds(-1000.0, 1000.0);
    m_light_pos_z_slider->setValue(m_plugin->getSettings().point_light_pos.z);
    m_light_pos_z_slider->setShared(true);
    m_light_pos_z_slider->setCallback([this](double value, bool) {
        m_plugin->getSettings().point_light_pos.z = static_cast<float>(value);
        });

    m_tone_mapping_button = new opencover::ui::Button(m_lighting_group, "tone_mapping");
    m_tone_mapping_button->setText("Tone Mapping");
    m_tone_mapping_button->setShared(true);
    m_tone_mapping_button->setState(m_plugin->getSettings().use_tone_mapping);
    m_tone_mapping_button->setCallback([this](bool state) {
        m_plugin->getSettings().use_tone_mapping = state;
        });

    {
        auto &st = m_plugin->getSettings();
        if (!st.point && !st.surfel && !st.splatting)
            st.point = true;

        if (st.splatting)        { st.point = false; st.surfel = false; }
        else if (st.surfel)      { st.point = false; st.splatting = false; }
        else /* point selected */{ st.surfel = false; st.splatting = false; }

        m_point_button ->setState(st.point);
        m_surfel_button->setState(st.surfel);
        m_splat_button ->setState(st.splatting);

        if (st.coloring && !st.lighting) { m_coloring_button->setState(true);  m_lighting_button->setState(false); }
        else if (!st.coloring && st.lighting) { m_coloring_button->setState(false); m_lighting_button->setState(true);  }
        else { 
            st.coloring = false; m_coloring_button->setState(false);
            st.lighting = false; m_lighting_button->setState(false);
        }
    }
    {
        auto &st = m_plugin->getSettings();
        st.scale_element = st.point ? st.scale_point : st.scale_surfel;
    }

    const auto &available_shaders = m_plugin->getRenderer()->getPclShader();

    auto applyShader = [this, &available_shaders](LamureRenderer::ShaderType t) {
        auto &st = m_plugin->getSettings();
        st.shader_type = t;
        std::string name;
        for (const auto &si : available_shaders) {
            if (si.type == t) { name = si.name; break; }
        }
        st.shader = name;
    };

    // POINT
    m_point_button->setCallback([this, applyShader](bool on) {
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (!st.surfel && !st.splatting) m_point_button->setState(true);
            return;
        }
        st.point = true; st.surfel = false; st.splatting = false;
        if (m_surfel_button) m_surfel_button->setState(false);
        if (m_splat_button)  m_splat_button ->setState(false);

        // statt 1.0f:
        st.scale_element = st.scale_point;   // <<<< hier anpassen

        applyShader(st.lighting ? LamureRenderer::ShaderType::PointColorLighting
            : (st.coloring ? LamureRenderer::ShaderType::PointColor
                : LamureRenderer::ShaderType::Point));
        });

    // SURFEL
    m_surfel_button->setCallback([this, applyShader](bool on) {
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (!st.point && !st.splatting) m_surfel_button->setState(true);
            return;
        }
        st.point = false; st.surfel = true; st.splatting = false;
        if (m_point_button) m_point_button->setState(false);
        if (m_splat_button) m_splat_button->setState(false);

        // >> neu/entscheidend:
        st.scale_element = st.scale_surfel;

        applyShader(st.lighting ? LamureRenderer::ShaderType::SurfelColorLighting
            : (st.coloring ? LamureRenderer::ShaderType::SurfelColor
                : LamureRenderer::ShaderType::Surfel));
        });

    // SPLAT
    m_splat_button->setCallback([this, applyShader](bool on) {
        auto &st = m_plugin->getSettings();
        if (!on) {
            if (!st.point && !st.surfel) m_splat_button->setState(true);
            return;
        }
        st.point = false; st.surfel = false; st.splatting = true;
        if (m_point_button)  m_point_button ->setState(false);
        if (m_surfel_button) m_surfel_button->setState(false);

        // >> neu/entscheidend:
        st.scale_element = st.scale_surfel;

        applyShader(LamureRenderer::ShaderType::SurfelMultipass);
        });

    // COLORING
    m_coloring_button->setCallback([this, applyShader](bool on) {
        auto &st = m_plugin->getSettings();
        st.coloring = on;
        if (on && st.lighting) {
            st.lighting = false;
            if (m_lighting_button && m_lighting_button->state() != st.lighting)
                m_lighting_button->setState(st.lighting);
        }
        LamureRenderer::ShaderType t;
        if (st.splatting) t = LamureRenderer::ShaderType::SurfelMultipass;
        else if (st.surfel) t = st.lighting ? LamureRenderer::ShaderType::SurfelColorLighting
                             : (st.coloring ? LamureRenderer::ShaderType::SurfelColor
                                            : LamureRenderer::ShaderType::Surfel);
        else  t = st.lighting ? LamureRenderer::ShaderType::PointColorLighting
               : (st.coloring ? LamureRenderer::ShaderType::PointColor
                              : LamureRenderer::ShaderType::Point);
        applyShader(t); 
        });

    // LIGHTING
    m_lighting_button->setCallback([this, applyShader](bool on) {
        auto &st = m_plugin->getSettings();
        st.lighting = on;
        if (on && st.coloring) {
            st.coloring = false;
            if (m_coloring_button && m_coloring_button->state() != st.coloring)
                m_coloring_button->setState(st.coloring);
        }
        LamureRenderer::ShaderType t;
        if (st.splatting)   t = LamureRenderer::ShaderType::SurfelMultipass;
        else if (st.surfel) t = st.lighting ? LamureRenderer::ShaderType::SurfelColorLighting
                             : (st.coloring ? LamureRenderer::ShaderType::SurfelColor
                                            : LamureRenderer::ShaderType::Surfel);
        else t = st.lighting ? LamureRenderer::ShaderType::PointColorLighting
              : (st.coloring ? LamureRenderer::ShaderType::PointColor
                             : LamureRenderer::ShaderType::Point);
        applyShader(t);
        });


    // --- Measurement ---

    m_measure_menu  = new opencover::ui::Menu (m_lamure_menu, "Measurement");

    // --- Settings normalisieren (Priorität: Full > Light > Off)
    {
        auto& S = m_plugin->getSettings();
        if (S.measure_full) {
            S.measure_full  = true;  S.measure_light = false; S.measure_off = false;
        } else if (S.measure_light) {
            S.measure_full  = false; S.measure_light = true;  S.measure_off = false;
        } else if (S.measure_off) {
            S.measure_full  = false; S.measure_light = false; S.measure_off = true;
        } else {
            // Nichts gesetzt -> Full als Default
            S.measure_full  = true;  S.measure_light = false; S.measure_off = false;
        }
    }

    // Hilfsfunktion: Modus anwenden (Settings + UI-Buttons synchron halten)
    auto applyMeasure = [this](bool full, bool light, bool off) {

        auto& S = m_plugin->getSettings();
        S.measure_full  = full;
        S.measure_light = light;
        S.measure_off   = off;

        // UI spiegeln
        m_measure_full->setState (full);
        m_measure_light->setState(light);
        m_measure_off->setState  (off);

        };

    // --- Buttons (jedes Element braucht einen eindeutigen Namen!)
    m_measure_full = new opencover::ui::Button(m_measure_menu, "full");
    m_measure_full->setText("Messung full");
    m_measure_full->setShared(true);

    m_measure_light = new opencover::ui::Button(m_measure_menu, "light");
    m_measure_light->setText("Messung light");
    m_measure_light->setShared(true);

    m_measure_off = new opencover::ui::Button(m_measure_menu, "off");
    m_measure_off->setText("Messung off");
    m_measure_off->setShared(true);

    // Initiale UI-States aus Settings übernehmen
    {
        auto& S = m_plugin->getSettings();
        m_measure_full ->setState(S.measure_full);
        m_measure_light->setState(S.measure_light);
        m_measure_off  ->setState(S.measure_off);
    }

    // Callbacks: Exklusivität erzwingen + aktiven Button nicht ausschalten lassen
    m_measure_full->setCallback([this, applyMeasure](bool state){
        if (!state) { m_measure_full->setState(true); return; }
        applyMeasure(true, false, false);
        });

    m_measure_light->setCallback([this, applyMeasure](bool state){
        if (!state) { m_measure_light->setState(true); return; }
        applyMeasure(false, true, false);
        });

    m_measure_off->setCallback([this, applyMeasure](bool state){
        if (!state) { m_measure_off->setState(true); return; }
        applyMeasure(false, false, true);
        });


    m_measure_sample = new opencover::ui::Slider(m_measure_menu, "sampling");
    m_measure_sample->setText("Sampling");
    m_measure_sample->setBounds(1, 60);
    m_measure_sample->setIntegral(true);
    m_measure_sample->setValue(m_plugin->getSettings().measure_sample);
    m_measure_sample->setCallback([this](double value, bool released){
        m_plugin->getSettings().measure_sample = static_cast<float>(value);
        });

    // Run Measurement (Toggle)
    m_measure_button = new opencover::ui::Button(m_measure_menu, "run_measurement");
    m_measure_button->setText("Run Measurement");
    m_measure_button->setShared(true);
    m_measure_button->setState(false);
    m_measure_button->setCallback([this](bool on){
        if (on) m_plugin->startMeasurement();
        else    m_plugin->stopMeasurement();
        });

}
