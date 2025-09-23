
#ifndef _Lamure_PC_PLUGIN_H
#define _Lamure_PC_PLUGIN_H

//gl
#ifndef __gl_h_
    #include <GL/glew.h>
#endif

#include <cover/coVRPluginSupport.h>
#include <cover/coVRMSController.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRCommunication.h>
#include <cover/coVRConfig.h>
#include <cover/coVRTui.h>
#include <cover/coVRShader.h>
#include <cover/VRViewer.h>
#include <cover/PluginMenu.h>
#include <cover/ui/ButtonGroup.h>
#include <cover/ui/Button.h>
#include <cover/ui/Label.h>
#include <cover/ui/Menu.h>
#include <cover/ui/Slider.h>
#include <cover/ui/Action.h>
#include <cover/ui/Manager.h>
#include <cover/ui/Owner.h>
#include <cover/ui/SelectionList.h>
#include <cover/coVRStatsDisplay.h>
#include <cover/VRSceneGraph.h>
#include "cover/OpenCOVER.h"
#include <cover/VRWindow.h>
#include <cover/coVRFileManager.h>

#include <osg/Version>
#include <osg/Geometry>
#include <osg/Vec3>
#include <osg/Vec3ui>
#include <osg/BufferObject>
#include <osg/Point>
#include <osg/PointSprite>
#include <osg/Texture2D>
#include <osgDB/ReadFile>
#include <osgViewer/Viewer>
#include <osgViewer/ViewerEventHandlers>
#include <osg/LineStipple>
#include <osg/BufferTemplate>
#include <osg/State>

#include "LamureDrawable.h"
#include "LamureGeometry.h"
#include <lamure/types.h>
#include <lamure/ren/camera.h>
#include <lamure/lmr_camera.h>
#include <lamure/ren/trackball.h>

#include <scm/gl_util/font/font_face.h>
#include <scm/gl_util/font/text.h>
#include <scm/gl_util/font/text_renderer.h>
#include <scm/gl_core/shader_objects/shader_objects_fwd.h>

#include "measurement.h"
#include <ft2build.h>
#include <LamurePointCloudInteractor.h>
#include FT_FREETYPE_H

namespace opencover {
    namespace ui {
        class Element;
        class Group;
        class Slider;
        class Menu;
        class Button;
    }
}

using namespace opencover;
class LamurePointCloudPlugin : public coVRPlugin, public ui::Owner
{
    class ImageFileEntry
    {
    public:
        string menuName;
        string fileName;
        ui::Element* fileMenuItem;

        ImageFileEntry(const char* menu, const char* file, ui::Element* menuitem)
        {
            menuName = menu;
            fileName = file;
            fileMenuItem = menuitem;
        }
    };

public:
    LamurePointCloudPlugin();
    ~LamurePointCloudPlugin();

    static LamurePointCloudPlugin* instance();
    bool init2();
    static int loadLMR(const char* filename, osg::Group* parent, const char* ck = "");
    static int unloadLMR(const char* filename, const char* ck = "");
    void preFrame();
    //void preDraw();
    size_t query_video_memory_in_mb();

    // shared functions
    void init_lamure_shader();
    void init_schism_objects();
    bool read_shader(std::string const& path_string, std::string& shader_string, bool keep_optional_shader_code);
    void create_aux_resources();
    void set_lamure_uniforms(scm::gl::program_ptr shader);
    void create_framebuffers();
    void init_render_states();
    void init_camera();
    void debug_print_settings() const;
    void setViewerPos(float x, float y, float z);

    void init_pcl_resources();
    void init_box_resources();
    void init_coord_resources();
    void init_frustum_resources();

    GLuint compile_and_link_shaders(std::string vs_source, std::string fs_source);
    GLuint compile_and_link_shaders(std::string vs_source, std::string gs_source, std::string fs_source);
    unsigned int create_shader(const std::string& vertexShader, const std::string& fragmentShader, uint8_t ctx_id);
    unsigned int compile_shader(unsigned int type, const std::string& source, uint8_t ctx_id);

    void init_uniforms();
    void set_point_uniforms();
    void set_surfel_uniforms();

    void startMeasurement();
    void stopMeasurement();

    LamurePointCloudInteractor* interactor;

    // util
    bool parse_prefix(std::string& in_string, std::string const& prefix);
    string getConfigEntry(string scope);
    string getConfigEntry(string scope, string name);
    
    // objects and pointers
    ui::Group* FileGroup;
    scm::math::mat4d load_matrix(const std::string& filename);
    void load_settings(const std::string &filename);
    bool rendering_ = false;

    HWND HWND_cover;
    HWND HWND_opencover;
    HWND HWND_draw;
    HWND HWND_init;

    HGLRC HGLRC_cover;
    HGLRC HGLRC_opencover;
    HGLRC HGLRC_draw;
    HGLRC HGLRC_init;

    HDC HDC_cover;
    HDC HDC_opencover;
    HDC HDC_draw;
    HDC HDC_init;

    scm::math::mat4d gl_modelview_matrix;
    scm::math::mat4d gl_projection_matrix;


private:
    static LamurePointCloudPlugin* plugin;
    std::vector<ImageFileEntry> pointVec;
    std::string const strip_whitespace(std::string const& in_string);
    void readMenuConfigData(const char*, std::vector<ImageFileEntry>&, ui::Group*);
    float pointSizeValue = 4;
    bool adaptLOD = true;
    std::vector<osg::Vec3> _path;
    float                  _speed = 1.0f;
    Measurement* _measurement = nullptr;
    bool measurement_running = 0;
    osgViewer::ViewerBase::FrameScheme rendering_scheme;
    std::function<void(bool)> _measureCB;


public:
    void printNodePath(osg::ref_ptr<osg::Node> pointer);
    std::vector<vector<float>> getSerializedBvhMinMax(const std::vector<scm::gl::boxf>);
    std::vector<float> getBoxCorners(scm::gl::boxf);
    osg::ref_ptr<osg::MatrixTransform> transform;

    ui::Menu* lamure_menu = nullptr;
    ui::Group* group = nullptr;
    ui::Group* model_group = nullptr;
    ui::Group* selection_group = nullptr;
    ui::Group* adaption_group = nullptr;
    ui::Group* rendering_group = nullptr;

    std::vector<ui::Button*>    model_buttons_;
    std::vector<bool>           model_visible_;

    ui::Button* pointcloud_button = nullptr;
    ui::Button* boundingbox_button = nullptr;
    ui::Button* frustum_button = nullptr;
    ui::Button* coord_button = nullptr;
    ui::Button* sync_button = nullptr;
    ui::Button* notify_button = nullptr;
    ui::Button* text_button = nullptr;
    ui::Button* dump_button = nullptr;

    ui::Button* _provButton = nullptr;
    ui::Button* _surfelButton = nullptr;
    ui::Button* _measureButton = nullptr;
    ui::Button* _faceEyeBotton = nullptr;

    bool dump = false;

    osg::ref_ptr<osg::Camera> pcl_camera;
    osg::ref_ptr<osg::Camera> hud_camera;
    osg::ref_ptr<osg::Group> LamureGroup;
    osg::ref_ptr<osg::Node> file;

    osg::ref_ptr<osg::Geode> init_geode;
    osg::ref_ptr<osg::Geode> pointcloud_geode;
    osg::ref_ptr<osg::Geode> boundingbox_geode;
    osg::ref_ptr<osg::Geode> frustum_geode;
    osg::ref_ptr<osg::Geode> coord_geode;
    osg::ref_ptr<osg::Geode> text_geode;

    osg::ref_ptr<osg::StateSet> init_stateset;
    osg::ref_ptr<osg::StateSet> pointcloud_stateset;
    osg::ref_ptr<osg::StateSet> boundingbox_stateset;
    osg::ref_ptr<osg::StateSet> frustum_stateset;
    osg::ref_ptr<osg::StateSet> coord_stateset;
    osg::ref_ptr<osg::StateSet> text_stateset;

    osg::ref_ptr<osg::Geometry> init_geometry;
    osg::ref_ptr<osg::Geometry> pointcloud_geometry;
    osg::ref_ptr<osg::Geometry> boundingbox_geometry;
    osg::ref_ptr<osg::Geometry> frustum_geometry;
    osg::ref_ptr<osg::Geometry> coord_geometry;

    // Slider-Deklarationen

    ui::Slider* cameraPosXSlider = nullptr;
    ui::Slider* cameraPosYSlider = nullptr;
    ui::Slider* cameraPosZSlider = nullptr;

    ui::Slider* modelPosXSlider = nullptr;
    ui::Slider* modelPosYSlider = nullptr;
    ui::Slider* modelPosZSlider = nullptr;

    ui::Slider* rotationXSlider = nullptr;
    ui::Slider* rotationYSlider = nullptr;
    ui::Slider* rotationZSlider = nullptr;

    // UI-Elemente für Kamera X-Position
    ui::Label* cameraPosXLabel;
    ui::Button* cameraPosXPlusButton;
    ui::Button* cameraPosXMinusButton;

    // UI-Elemente für Kamera Y-Position
    ui::Label* cameraPosYLabel;
    ui::Button* cameraPosYPlusButton;
    ui::Button* cameraPosYMinusButton;

    // UI-Elemente für Kamera Z-Position
    ui::Label* cameraPosZLabel;
    ui::Button* cameraPosZPlusButton;
    ui::Button* cameraPosZMinusButton;

    scm::math::vec3d cameraPosition;

    scm::math::vec3d rotationAngles = scm::math::vec3d(0.0, 0.0, 0.0);

    scm::math::mat4d swapMiddleColumns(const scm::math::mat4d& m);
    scm::math::mat4d swapMiddleRows(const scm::math::mat4d& m);
    scm::math::mat4d swapMiddleColumns(scm::math::mat4d& m);
    scm::math::mat4d swapMiddleRows(scm::math::mat4d& m);

protected:
    ui::Group* loadGroup = nullptr;
    ui::Group* viewGroup = nullptr;
    ui::Menu* loadMenu = nullptr;
    ui::Button* rotPointsButton = nullptr;
    ui::Button* rotAxisButton = nullptr;
    ui::Button* moveButton = nullptr;
    ui::Button* saveButton = nullptr;
    ui::Button* fileButton = nullptr;
    ui::Button* deselectButton = nullptr;
    ui::Button* createNurbsSurface = nullptr;
    //ui::Button *deleteButton = nullptr;
    ui::Button* adaptLODButton = nullptr;

    ui::Slider* maxRadiusSlider = nullptr;
    ui::Slider* scaleRadiusSlider = nullptr;
    ui::Slider* lodErrorSlider = nullptr;

    ui::SelectionList* shader_list = nullptr;

    ui::Slider* lodFarDistanceSlider = nullptr;
    ui::Slider* lodNearDistanceSlider = nullptr;

    ui::Slider* lodErrorSlider_;
    ui::Slider* uploadBudgetSlider_;
    ui::Slider* videoMemoryBudgetSlider_;
    ui::Slider* mainMemoryBudgetSlider_;
    ui::Slider* maxUpdatesSlider_;

    ui::SelectionList* modelList_;

};

#endif