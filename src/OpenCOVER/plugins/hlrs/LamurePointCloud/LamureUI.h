#ifndef _LAMURE_UI_H
#define _LAMURE_UI_H

#include <string>
#include <vector>

// OpenCOVER UI
#include <cover/ui/Owner.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Group.h>
#include <cover/ui/Button.h>
#include <cover/ui/Slider.h>
#include <cover/ui/SelectionList.h>
#include <cover/ui/Label.h>

class Lamure;

class LamureUI {
public:
    LamureUI(Lamure* lamure_plugin, const std::string& name);
    ~LamureUI();

    void setupUi();
    opencover::ui::Button* getPointcloudButton() { return m_pointcloud_button; }
    opencover::ui::Button* getBoundingboxButton() { return m_boundingbox_button; }
    opencover::ui::Button* getFrustumButton() { return m_frustum_button; }
    opencover::ui::Button* getSyncButton() { return m_sync_button; }
    opencover::ui::Button* getTextButton() { return m_text_button; }
    opencover::ui::Button* getDumpButton() { return m_dump_button; }
    opencover::ui::Button* getProvButton() { return m_prov_button; }
    opencover::ui::Button* getMeasureButton() { return m_measure_button; }
    opencover::ui::Button* getNotifyButton() { return m_notify_button; }

    std::vector<bool> getModelVisibility() { return m_model_visible; }

    void updateShader();
    void updateCheckboxesFromShaderType();

private:
    Lamure* m_plugin;
    bool m_ui_updating = false;
    opencover::ui::Slider* m_scale_point_slider = nullptr;

    //opencover::ui::Label* m_shader_label = nullptr;

    // Shader options
    opencover::ui::Group*  m_shader_options_group   = nullptr;
    opencover::ui::Button* m_check_point            = nullptr;
    opencover::ui::Button* m_check_surfel           = nullptr;
    opencover::ui::Button* m_check_splatting        = nullptr;
    opencover::ui::Button* m_check_coloring         = nullptr;
    opencover::ui::Button* m_check_lighting         = nullptr;

    // Models
    std::vector<opencover::ui::Button*> m_model_buttons;
    std::vector<bool>                   m_model_visible;

    opencover::ui::SelectionList *m_mode_choice     = nullptr;
    opencover::ui::SelectionList *m_shader_choice   = nullptr;

    // Main buttons
    opencover::ui::Button* m_pointcloud_button   = nullptr;
    opencover::ui::Button* m_boundingbox_button  = nullptr;
    opencover::ui::Button* m_frustum_button      = nullptr;
    opencover::ui::Button* m_sync_button         = nullptr;
    opencover::ui::Button* m_notify_button       = nullptr;
    opencover::ui::Button* m_text_button         = nullptr;
    opencover::ui::Button* m_dump_button         = nullptr;
    opencover::ui::Button* m_prov_button         = nullptr;

    // Groups and menus
    opencover::ui::Menu*   m_lamure_menu        = nullptr;
    opencover::ui::Group*  m_selection_group    = nullptr;
    opencover::ui::Group*  m_misc_group         = nullptr;
    opencover::ui::Group*  m_model_group        = nullptr;
    opencover::ui::Group*  m_adaption_group     = nullptr;
    opencover::ui::Group*  m_prov_group         = nullptr;
    opencover::ui::Group*  m_color_group        = nullptr;
    opencover::ui::Group*  m_rendering_group    = nullptr;
    opencover::ui::Group*  m_shader_group       = nullptr;

    opencover::ui::Menu*   m_measure_menu        = nullptr;
    opencover::ui::Button* m_measure_button      = nullptr;    
    opencover::ui::Button* m_measure_full        = nullptr;
    opencover::ui::Button* m_measure_light       = nullptr;
    opencover::ui::Button* m_measure_off         = nullptr;
    opencover::ui::Slider* m_measure_sample      = nullptr;

    // Scaling/LOD sliders and controls
    opencover::ui::Slider* m_scale_radius_slider       = nullptr;
    opencover::ui::Slider* m_scale_surfel_slider       = nullptr;
    opencover::ui::Slider* m_min_radius_slider         = nullptr;
    opencover::ui::Slider* m_max_radius_slider         = nullptr;
    opencover::ui::Slider* m_min_screen_size_slider    = nullptr;
    opencover::ui::Slider* m_max_screen_size_slider    = nullptr;
    opencover::ui::Slider* m_max_radius_cut_slider     = nullptr;
    opencover::ui::Slider* m_lod_error_slider          = nullptr;
    opencover::ui::Slider* m_upload_budget_slider      = nullptr;
    opencover::ui::Slider* m_video_memory_budget_slider= nullptr;
    opencover::ui::Slider* m_depth_range_slider        = nullptr;
    opencover::ui::Slider* m_flank_lift_slider         = nullptr;
    // Anisotropic scaling mode (exclusive buttons)
    opencover::ui::Button* m_aniso_off_btn             = nullptr;
    opencover::ui::Button* m_aniso_auto_btn            = nullptr;
    opencover::ui::Button* m_aniso_on_btn              = nullptr;

    opencover::ui::Menu*    m_scaling_menu                  = nullptr;
    opencover::ui::Group*   m_scaling_group                 = nullptr;

    opencover::ui::Menu*   m_lod_menu                       = nullptr;
    opencover::ui::Group*  m_lod_group                      = nullptr;
    opencover::ui::Button* m_lod_button                     = nullptr;

    opencover::ui::Menu*   m_lighting_menu                  = nullptr;
    opencover::ui::Button* m_lighting_button                = nullptr;
    opencover::ui::Group*  m_lighting_group                 = nullptr;
    opencover::ui::Slider* m_light_color_slider             = nullptr;
    opencover::ui::Slider* m_point_light_intensity_slider   = nullptr;
    opencover::ui::Slider* m_ambient_intensity_slider       = nullptr;
    opencover::ui::Slider* m_specular_intenity_slider       = nullptr;
    opencover::ui::Slider* m_shininess_slider               = nullptr;
    opencover::ui::Slider* m_gamma_slider                   = nullptr;
    opencover::ui::Slider* m_scale_radius_gamma_slider      = nullptr;
    opencover::ui::Slider* m_light_pos_x_slider             = nullptr;
    opencover::ui::Slider* m_light_pos_y_slider             = nullptr;
    opencover::ui::Slider* m_light_pos_z_slider             = nullptr;
    opencover::ui::Button* m_tone_mapping_button            = nullptr;

    // Primitive selection
    opencover::ui::Group*  m_primitives_group   = nullptr;
    opencover::ui::Button* m_point_button       = nullptr;
    opencover::ui::Button* m_surfel_button      = nullptr;
    opencover::ui::Button* m_splat_button       = nullptr;

    opencover::ui::Menu*   m_coloring_menu      = nullptr;
    opencover::ui::Group*  m_coloring_group     = nullptr;
    opencover::ui::Button* m_coloring_button    = nullptr;

    opencover::ui::Menu*   m_model_menu             = nullptr;
    opencover::ui::Group*  m_mode_group             = nullptr;
    opencover::ui::Button* m_mode_normals_btn       = nullptr;
    opencover::ui::Button* m_mode_accuracy_btn      = nullptr;
    opencover::ui::Button* m_mode_radius_dev_btn    = nullptr;
    opencover::ui::Button* m_mode_output_sens_btn   = nullptr;
};

#endif // _LAMURE_UI_H
