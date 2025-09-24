#define GLFW_EXPOSE_NATIVE_WIN32
//local
#include "Lamure.h" 
#include "gl_state.h"
#include "osg_util.h"
//#include "LamurePointCloudInteractor.h"

#include <lamure/imgui.h>
#include <lamure/imgui_internal.h>
#include <lamure/imgui_impl_glfw_gl3.h>

// std
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <sstream>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <math.h>
#include <winbase.h>
#include <mutex>
#include <filesystem>
#include <memory>
#include <algorithm>
#include <cmath>

//boost
#include <boost/regex.hpp>
#include <regex>

//schism
#include <scm/core/math.h>

//lamure
#include <lamure/pvs/pvs_database.h>
#include <lamure/prov/prov_aux.h>
#include <lamure/prov/octree.h>
#include "lamure/ren/controller.h"
#include <lamure/ren/cut.h>
#include <lamure/ren/policy.h>

#include <config/coConfigConstants.h>
#include <config/CoviseConfig.h>

#include <cover/VRSceneGraph.h>
#include "cover/OpenCOVER.h"
#include <cover/VRViewer.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRPluginSupport.h>
#include <cover/coVRConfig.h>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <cover/coVRPluginList.h>
#include <cover/coVRNavigationManager.h>


#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllexport) DWORD NvOptimusEnablement = 1;
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#ifdef __cplusplus
}
#endif

COVERPLUGIN(Lamure)
Lamure* Lamure::plugin = nullptr;
static opencover::FileHandler handler = {NULL, Lamure::loadBvh, Lamure::unloadBvh, "bvh"};
Lamure::Lamure() :coVRPlugin(COVER_PLUGIN_NAME), opencover::ui::Owner("Lamure", opencover::cover->ui)
{
    opencover::coVRFileManager::instance()->registerFileHandler(&handler);
	plugin = this;
    m_ui = std::make_unique<LamureUI>(this, "LamureUI");
    m_renderer = std::make_unique<LamureRenderer>(this);
}

Lamure* Lamure::instance()
{
	return plugin;
}

Lamure::~Lamure()
{
	fprintf(stderr, "LamurePlugin::~LamurePlugin\n");
	opencover::coVRFileManager::instance()->unregisterFileHandler(&handler);
}

int Lamure::loadBvh(const char* filename, osg::Group* parent, const char* covise_key) 
{
    std::string bvh_file = filename ? std::string(filename) : std::string();
#ifdef _WIN32
    std::replace(bvh_file.begin(), bvh_file.end(), '\\', '/');
#endif

    if (bvh_file.empty())
        return 0;

    auto &s = plugin->getSettings();

    if (std::find(s.models.begin(), s.models.end(), bvh_file) == s.models.end())
        s.models.push_back(bvh_file);

    s.num_models = static_cast<int>(s.models.size());
    return 1;
}

int Lamure::unloadBvh(const char* filename, const char* covise_key)
{
    std::string path = filename ? std::string(filename) : std::string();
#ifdef _WIN32
    std::replace(path.begin(), path.end(), '\\', '/');
#endif
    if (path.empty()) return 0;

    if (!Lamure::plugin) return 0; // schon weg

    auto &s = Lamure::plugin->getSettings();
    s.models.erase(std::remove(s.models.begin(), s.models.end(), path), s.models.end());
    s.num_models = static_cast<int>(s.models.size());

    std::printf("[Lamure] unloadBvh '%s' -> num_models=%d\n", path.c_str(), s.num_models);

    if (s.models.empty()) {
        // Instanz-Zeiger über PluginList holen und entladen -> ~Lamure() läuft
        if (auto *p = opencover::coVRPluginList::instance()->getPlugin("Lamure")) {
            opencover::coVRPluginList::instance()->unload(p);  // Destruktor wird hier aufgerufen
            // Danach NICHT mehr auf Lamure::plugin zugreifen.
        }
    }
}

namespace {
    inline std::string getStr(const char* path, const std::string& def){
        return covise::coCoviseConfig::getEntry(std::string("value"), std::string(path), def, nullptr);
    }

    template<typename T> T getNum(const char* attr, const char* path, T def);

    template<> inline int getNum<int>(const char* attr, const char* path, int def){
        return covise::coCoviseConfig::getInt(std::string(attr), std::string(path), def);
    }
    template<> inline float getNum<float>(const char* attr, const char* path, float def){
        return covise::coCoviseConfig::getFloat(std::string(attr), std::string(path), def);
    }
    template<> inline double getNum<double>(const char* attr, const char* path, double def){
        return static_cast<double>(covise::coCoviseConfig::getFloat(std::string(attr), std::string(path), static_cast<float>(def)));
    }

    inline bool getOn(const char* path, bool def){
        return covise::coCoviseConfig::isOn(std::string("value"), std::string(path), def);
    }
} // namespace

void Lamure::loadSettingsFromCovise(){
    auto& s = m_settings;
    const char* root = "COVER.Plugin.LamurePointCloud";

    // ---- Budgets / LODs ----
    s.frame_div = getNum<int>("value", (std::string(root) + ".frame_div").c_str(), s.frame_div);
    s.vram      = getNum<int>("value", (std::string(root) + ".vram").c_str(),      s.vram);
    s.ram       = getNum<int>("value", (std::string(root) + ".ram").c_str(),       s.ram);
    s.upload    = getNum<int>("value", (std::string(root) + ".upload").c_str(),    s.upload);
    s.lod_error = getNum<float>("value", (std::string(root) + ".lod_error").c_str(), s.lod_error);

    // ---- Tuning / Flags ----
    s.face_eye             = getOn((std::string(root) + ".face_eye").c_str(),             s.face_eye);
    s.pvs_culling          = getOn((std::string(root) + ".pvs_culling").c_str(),          s.pvs_culling);
    s.use_pvs              = getOn((std::string(root) + ".use_pvs").c_str(),              s.use_pvs);
    s.create_aux_resources = getOn((std::string(root) + ".create_aux_resources").c_str(), s.create_aux_resources);
    s.max_brush_size       = getNum<int>("value", (std::string(root) + ".max_brush_size").c_str(), s.max_brush_size);
    s.channel              = getNum<int>("value", (std::string(root) + ".channel").c_str(), s.channel);

    // ---- Visual toggles ----
    s.show_pointcloud         = getOn((std::string(root) + ".show_pointcloud").c_str(),        s.show_pointcloud);
    s.show_boundingbox        = getOn((std::string(root) + ".show_boundingbox").c_str(),       s.show_boundingbox);
    s.show_frustum            = getOn((std::string(root) + ".show_frustum").c_str(),           s.show_frustum);
    s.show_coord              = getOn((std::string(root) + ".show_coord").c_str(),             s.show_coord);
    s.show_text               = getOn((std::string(root) + ".show_text").c_str(),              s.show_text);
    s.show_sync               = getOn((std::string(root) + ".show_sync").c_str(),              s.show_sync);
    s.show_notify             = getOn((std::string(root) + ".show_notify").c_str(),            s.show_notify);
    s.show_sparse             = getOn((std::string(root) + ".show_sparse").c_str(),            s.show_sparse);
    s.show_views              = getOn((std::string(root) + ".show_views").c_str(),             s.show_views);
    s.show_photos             = getOn((std::string(root) + ".show_photos").c_str(),            s.show_photos);
    s.show_octrees            = getOn((std::string(root) + ".show_octrees").c_str(),           s.show_octrees);
    s.show_bvhs               = getOn((std::string(root) + ".show_bvhs").c_str(),              s.show_bvhs);
    s.show_pvs                = getOn((std::string(root) + ".show_pvs").c_str(),               s.show_pvs);

    // ---- Lighting / ToneMapping ----
    s.point_light_intensity = getNum<float>("value", (std::string(root) + ".point_light_intensity").c_str(), s.point_light_intensity);
    s.ambient_intensity     = getNum<float>("value", (std::string(root) + ".ambient_intensity").c_str(),     s.ambient_intensity);
    s.specular_intensity    = getNum<float>("value", (std::string(root) + ".specular_intensity").c_str(),    s.specular_intensity);
    s.shininess             = getNum<float>("value", (std::string(root) + ".shininess").c_str(),             s.shininess);
    s.gamma                 = getNum<float>("value", (std::string(root) + ".gamma").c_str(),                 s.gamma);
    s.use_tone_mapping      = getOn((std::string(root) + ".use_tone_mapping").c_str(),                       s.use_tone_mapping);

    s.point_light_pos.x = getNum<float>("value", (std::string(root) + ".point_light_pos_x").c_str(), s.point_light_pos.x);
    s.point_light_pos.y = getNum<float>("value", (std::string(root) + ".point_light_pos_y").c_str(), s.point_light_pos.y);
    s.point_light_pos.z = getNum<float>("value", (std::string(root) + ".point_light_pos_z").c_str(), s.point_light_pos.z);

    // ---- Heatmap ----
    s.heatmap     = getOn((std::string(root) + ".heatmap").c_str(),     s.heatmap);
    s.heatmap_min = getNum<float>("value", (std::string(root) + ".heatmap_min").c_str(), s.heatmap_min);
    s.heatmap_max = getNum<float>("value", (std::string(root) + ".heatmap_max").c_str(), s.heatmap_max);
    auto clamp255 = [](int v){ return std::max(0, std::min(255, v)); };
    int hmin_r = getNum<int>("value", (std::string(root) + ".heatmap_min_r").c_str(), int(std::round(s.heatmap_color_min.x * 255.f)));
    int hmin_g = getNum<int>("value", (std::string(root) + ".heatmap_min_g").c_str(), int(std::round(s.heatmap_color_min.y * 255.f)));
    int hmin_b = getNum<int>("value", (std::string(root) + ".heatmap_min_b").c_str(), int(std::round(s.heatmap_color_min.z * 255.f)));
    int hmax_r = getNum<int>("value", (std::string(root) + ".heatmap_max_r").c_str(), int(std::round(s.heatmap_color_max.x * 255.f)));
    int hmax_g = getNum<int>("value", (std::string(root) + ".heatmap_max_g").c_str(), int(std::round(s.heatmap_color_max.y * 255.f)));
    int hmax_b = getNum<int>("value", (std::string(root) + ".heatmap_max_b").c_str(), int(std::round(s.heatmap_color_max.z * 255.f)));
    s.heatmap_color_min = scm::math::vec3f(clamp255(hmin_r)/255.f, clamp255(hmin_g)/255.f, clamp255(hmin_b)/255.f);
    s.heatmap_color_max = scm::math::vec3f(clamp255(hmax_r)/255.f, clamp255(hmax_g)/255.f, clamp255(hmax_b)/255.f);

    // ---- Radii / Shader-Name (frei) ----
    s.shader            = getStr((std::string(root) + ".shader").c_str(), s.shader);
    s.min_radius        = getNum<float>("value", (std::string(root) + ".min_radius").c_str(),        s.min_radius);
    s.max_radius        = getNum<float>("value", (std::string(root) + ".max_radius").c_str(),        s.max_radius);
    s.min_screen_size   = getNum<float>("value", (std::string(root) + ".min_screen_size").c_str(),   s.min_screen_size);
    s.max_screen_size   = getNum<float>("value", (std::string(root) + ".max_screen_size").c_str(),   s.max_screen_size);
    s.scale_radius      = getNum<float>("value", (std::string(root) + ".scale_radius").c_str(),      s.scale_radius);
    s.max_radius_cut    = getNum<float>("value", (std::string(root) + ".max_radius_cut").c_str(),    s.max_radius_cut);
    s.scale_radius_gamma= getNum<float>("value", (std::string(root) + ".radius_scale_gamma").c_str(),s.scale_radius_gamma);
    s.scale_point  = getNum<float>("value", (std::string(root) + ".scale_point").c_str(),  s.scale_point);
    s.scale_surfel = getNum<float>("value", (std::string(root) + ".scale_surfel").c_str(), s.scale_surfel);
    s.scale_element = (s.point) ? s.scale_point : s.scale_surfel;

    // ---- Primitive + Modes aus Config ----
    const bool p0  = getOn((std::string(root)+".point").c_str(),     s.point);
    const bool sf0 = getOn((std::string(root)+".surfel").c_str(),    s.surfel);
    const bool sp0 = getOn((std::string(root)+".splatting").c_str(), s.splatting);
    s.point     = p0; s.surfel = sf0; s.splatting = sp0;

    s.lighting  = getOn((std::string(root)+".lighting").c_str(), s.lighting);
    s.coloring  = getOn((std::string(root)+".color").c_str(),    s.coloring);

    std::string color_mode = getStr((std::string(root)+".color_mode").c_str(), "normals");
    for (char &c : color_mode) c = (char)std::tolower((unsigned char)c);

    // ---- Exklusivität ----
    // Primitive: genau eins. Priorität: Surfel > Splatting > Point.
    {
        int prim_count = (s.point?1:0) + (s.surfel?1:0) + (s.splatting?1:0);
        if (prim_count != 1) {
            s.point = s.surfel = s.splatting = false;
            if (sf0) s.surfel = true;
            else if (sp0) s.splatting = true;
            else s.point = true;
        }
    }
    // Modes: Lighting > Color
    if (s.lighting && s.coloring) s.coloring = false;

    // ---- color_mode immer übernehmen (auch wenn coloring=off), UI soll Zustand zeigen ----
    s.show_normals = s.show_accuracy = s.show_radius_deviation = s.show_output_sensitivity = false;
    if      (color_mode == "accuracy")    s.show_accuracy = true;
    else if (color_mode == "derivation")  s.show_radius_deviation = true;   // „derivation“ -> radius deviation
    else if (color_mode == "sensitivity") s.show_output_sensitivity = true;
    else                                  s.show_normals = true;

    // ---- ShaderType strikt aus den Booleans ableiten + Namen setzen ----
    auto decideShaderType = [&](){
        using ST = LamureRenderer::ShaderType;
        if (s.splatting) return ST::SurfelMultipass;
        if (s.surfel)    return s.lighting ? ST::SurfelColorLighting
            : (s.coloring ? ST::SurfelColor : ST::Surfel);
        return s.lighting ? ST::PointColorLighting
            : (s.coloring ? ST::PointColor : ST::Point);
        };
    auto typeToName = [](LamureRenderer::ShaderType t)->std::string{
        switch (t) {
        case LamureRenderer::ShaderType::Point:               return "Point";
        case LamureRenderer::ShaderType::PointColor:          return "Point Color";
        case LamureRenderer::ShaderType::PointColorLighting:  return "Point Color Lighting";
        case LamureRenderer::ShaderType::PointProv:           return "Point Prov";
        case LamureRenderer::ShaderType::Surfel:              return "Surfel";
        case LamureRenderer::ShaderType::SurfelColor:         return "Surfel Color";
        case LamureRenderer::ShaderType::SurfelColorLighting: return "Surfel Color Lighting";
        case LamureRenderer::ShaderType::SurfelProv:          return "Surfel Prov";
        case LamureRenderer::ShaderType::SurfelMultipass:     return "Surfel Multipass";
        default:                                              return "Point";
        }
        };
    s.shader_type = decideShaderType();
    s.shader      = typeToName(s.shader_type);
    if (s.surfel || s.splatting) s.scale_element = s.scale_surfel;

    // ---- Multi-Pass Blending ----
    s.depth_range = getNum<float>("value", (std::string(root) + ".depth_range").c_str(), s.depth_range);
    s.flank_lift  = getNum<float>("value", (std::string(root) + ".flank_lift").c_str(),  s.flank_lift);

    // ---- Dateien / Pfade ----
    s.pvs              = getStr((std::string(root) + ".pvs").c_str(),              s.pvs);
    s.background_image = getStr((std::string(root) + ".background_image").c_str(), s.background_image);

    // ---- Modelle: models (Semikolon), optional data_dir (rekursiv .bvh) ----
    s.models.clear();
    const std::string models_list = getStr((std::string(root) + ".models").c_str(), "");
    const std::string data_dir    = getStr((std::string(root) + ".data_dir").c_str(), "");
    for (const auto& m : LamureUtil::splitSemicolons(models_list))
        s.models.push_back(std::filesystem::absolute(m).string());
    if (!data_dir.empty()){
        for (auto& e : std::filesystem::recursive_directory_iterator(data_dir)){
            if (e.is_regular_file() && e.path().extension() == ".bvh")
                s.models.push_back(std::filesystem::absolute(e.path()).string());
        }
    }

    const std::string sel = getStr((std::string(root) + ".initial_selection").c_str(), "");
    auto parseIndices = [&](const std::string& str, size_t N){
        std::vector<uint32_t> out; if (str.empty()) return out;
        std::istringstream ss(str); std::string part;
        auto trim=[&](std::string t){ auto b=t.find_first_not_of(" \t"); auto e=t.find_last_not_of(" \t");
        return (b==std::string::npos)?std::string():t.substr(b,e-b+1); };
        while(std::getline(ss,part,',')){
            part=trim(part); auto dash=part.find('-');
            if(dash!=std::string::npos){
                int a=std::stoi(part.substr(0,dash)), b=std::stoi(part.substr(dash+1)); if(a>b) std::swap(a,b);
                for(int i=a;i<=b;++i) if(i>=0 && (size_t)i<N) out.push_back((uint32_t)i);
            } else {
                int v=std::stoi(part); if(v>=0 && (size_t)v<N) out.push_back((uint32_t)v);
            }
        }
        std::sort(out.begin(),out.end()); out.erase(std::unique(out.begin(),out.end()),out.end()); return out;
        };
    s.initial_selection = parseIndices(sel, s.models.size());


    // ---- Initial matrices ----
    {
        const std::string navKey  = std::string(root) + ".initial_navigation";
        const std::string viewKey = std::string(root) + ".initial_view";
        const std::string navStr  = getStr(navKey.c_str(),  "");
        const std::string viewStr = getStr(viewKey.c_str(), "");

        osg::Matrixd M;
        auto tryParse = [](const std::string& label, const std::string& s, osg::Matrixd& out)->bool{
            if (!LamureUtil::readIndexedMatrix(s, out)) {
                if (!s.empty()) std::cerr << "[Lamure] " << label << " parse failed: " << s << "\n";
                return false;
            }
            return true;
            };
        if (tryParse("initial_navigation", navStr, M)) { s.initial_navigation = M; s.use_initial_navigation = true; }
        if (tryParse("initial_view",       viewStr, M)) { s.initial_view       = M; s.use_initial_view       = true; }

        if (!s.use_initial_navigation) {
            if (auto* sg = opencover::VRSceneGraph::instance()) {
                s.initial_navigation = sg->getTransform()->getMatrix();
                s.use_initial_navigation = true;
            }
        }
        if (!s.use_initial_view) {
            if (auto* viewer = opencover::VRViewer::instance()) {
                if (auto* cam = viewer->getCamera()) {
                    s.initial_view = cam->getViewMatrix();
                    s.use_initial_view = true;
                }
            }
        }
    }

    s.measurement_dir  = getStr((std::string(root) + ".measurement_dir").c_str(),  s.measurement_dir);
    s.measurement_name = getStr((std::string(root) + ".measurement_name").c_str(), s.measurement_name);

    // --- Measurement-Segmente ---
    const std::string segs = getStr((std::string(root) + ".measurement_segments").c_str(), "");
    s.measurement_segments = parseMeasurementSegments(segs);
    if (s.measurement_segments.empty()) {
        s.measurement_segments = {
            {{0,-500,0},{0,0,360},200.f,30.f},
            {{0,0,0},{45,0,360},200.f,30.f},
            {{0,-400,0},{0,0,0},200.f,30.f}
        };
    }

    // ---- Provenance & JSON ----
    s.json = getStr((std::string(root) + ".json").c_str(), "");
    if (!s.json.empty() && !std::filesystem::exists(s.json)) { std::cerr << "[Lamure] config json not found: " << s.json << " -> ignore\n"; s.json.clear(); }

    bool prov_valid = true; std::string first_json;
    if(!s.models.empty()){
        for(const auto& model_path: s.models){
            std::filesystem::path p(model_path), prov_file=p; prov_file.replace_extension(".prov");
            std::filesystem::path json_file=p; json_file.replace_extension(".json");
            if(!std::filesystem::exists(prov_file) || !std::filesystem::exists(json_file)){ prov_valid=false; break; }
            if(first_json.empty()) first_json=json_file.string();
        }
    }
    s.provenance = prov_valid;
    if (s.json.empty() && !first_json.empty()) s.json = first_json;

    // ---- Transforms default ----
    s.transforms.clear();
    for (lamure::model_t mid=0; mid < s.models.size(); ++mid)
        s.transforms[mid] = scm::math::mat4d::identity();

    // ---- Hintergrundfarbe ----
    s.background_color = scm::math::vec3(
        covise::coCoviseConfig::getFloat("r", "COVER.Background", 0.0f),
        covise::coCoviseConfig::getFloat("g", "COVER.Background", 0.0f),
        covise::coCoviseConfig::getFloat("b", "COVER.Background", 0.0f)
    );
}


void Lamure::dumpSettings(const char* tag){
    const auto& s = m_settings;
    auto b2 = [](bool v){ return v ? "on" : "off"; };
    auto v3 = [](const scm::math::vec3f& v){ std::ostringstream o; o<<v.x<<","<<v.y<<","<<v.z; return o.str(); };

    auto shaderTypeToStr = [](LamureRenderer::ShaderType t){
        switch(t){
        case LamureRenderer::ShaderType::Point:               return "Point";
        case LamureRenderer::ShaderType::PointColor:          return "PointColor";
        case LamureRenderer::ShaderType::PointColorLighting:  return "PointColorLighting";
        case LamureRenderer::ShaderType::PointProv:           return "PointProv";
        case LamureRenderer::ShaderType::Surfel:              return "Surfel";
        case LamureRenderer::ShaderType::SurfelColor:         return "SurfelColor";
        case LamureRenderer::ShaderType::SurfelColorLighting: return "SurfelColorLighting";
        case LamureRenderer::ShaderType::SurfelProv:          return "SurfelProv";
        case LamureRenderer::ShaderType::SurfelMultipass:     return "SurfelMultipass";
        default:                                              return "Point";
        }
        };

    // abgeleiteter Color-Mode (nur einer aktiv)
    auto colorModeStr = [&](){
        if (s.show_normals)            return "normals";
        if (s.show_accuracy)           return "accuracy";
        if (s.show_radius_deviation)   return "deviation";
        if (s.show_output_sensitivity) return "sensitivity";
        return "none";
        };

    // Konsistenzchecks (nur Ausgabe, keine Mutation)
    int prim_count = (s.point?1:0) + (s.surfel?1:0) + (s.splatting?1:0);
    bool prim_ok   = (prim_count == 1);
    bool mode_conflict = (s.coloring && s.lighting);

    double coverScale = opencover::cover
        ? static_cast<double>(opencover::cover->getScale())
        : static_cast<double>(covise::coCoviseConfig::getFloat("value","COVER.DefaultScaleFactor",1.0f));

    std::cout << "--- Lamure::Settings " << (tag?tag:"") << " ---\n";

    // Modelle
    std::cout << "models: " << s.models.size() << " (loaded=" << s.num_models << ")\n";
    for(size_t i=0;i<std::min<size_t>(s.models.size(),3);++i)
        std::cout << "  ["<<i<<"] " << s.models[i] << "\n";
    if (s.models.size() > 3) std::cout << "  ...\n";
    if(!s.initial_selection.empty()){
        std::cout << "initial_selection=";
        for(size_t i=0;i<s.initial_selection.size();++i){
            if(i) std::cout << ",";
            std::cout << s.initial_selection[i];
        }
        std::cout << "\n";
    }

    // Provenance
    std::cout << "provenance=" << b2(s.provenance) << "; json=" << (s.json.empty()?"<empty>":s.json) << "\n";

    // Primitive + Modes
    std::cout << "primitive: point="<<b2(s.point)
        << " surfel="<<b2(s.surfel)
        << " splatting="<<b2(s.splatting)
        << "  [" << (prim_ok ? "OK" : "INVALID") << "]\n";

    std::cout << "modes: lighting="<<b2(s.lighting)
        << " color="<<b2(s.coloring)
        << " color_mode="<< colorModeStr()
        << (mode_conflict ? "  [CONFLICT: lighting wins]" : "")
        << "\n";

    // Shader
    std::cout << "shader: '" << s.shader << "' (" << shaderTypeToStr(s.shader_type) << ")\n";

    // Budgets / LOD
    std::cout << "budgets: frame_div="<<s.frame_div
        << " vramMB="<<s.vram
        << " ramMB="<<s.ram
        << " uploadMB="<<s.upload
        << " lod_error="<<s.lod_error << "\n";

    // Radii/Scaling kompakt
    std::cout << "radii: world[min="<<s.min_radius<<", max="<<s.max_radius<<", cut="<<s.max_radius_cut<<"]"
        << " screen[min="<<s.min_screen_size<<", max="<<s.max_screen_size<<"]"
        << " scale_radius="<<s.scale_radius
        << " gamma="<<s.scale_radius_gamma
        << " scale_surfel="<<s.scale_surfel << "\n";

    // Lighting kompakt
    std::cout << "lighting: ambient="<<s.ambient_intensity
        << " point_I="<<s.point_light_intensity
        << " specular="<<s.specular_intensity
        << " shininess="<<s.shininess
        << " gamma="<<s.gamma
        << " tone_map="<<b2(s.use_tone_mapping)
        << " light_pos=("<<v3(s.point_light_pos)<<")\n";

    // Visual toggles
    std::cout << "visuals: pointcloud="<<b2(s.show_pointcloud)
        << " bbox="<<b2(s.show_boundingbox)
        << " frustum="<<b2(s.show_frustum)
        << " coord="<<b2(s.show_coord)
        << " text="<<b2(s.show_text)
        << " sync="<<b2(s.show_sync)
        << " notify="<<b2(s.show_notify) << "\n";

    // Heatmap
    std::cout << "heatmap: on="<<b2(s.heatmap)
        << " range=["<<s.heatmap_min<<","<<s.heatmap_max<<"]"
        << " cmin=("<<v3(s.heatmap_color_min)<<")"
        << " cmax=("<<v3(s.heatmap_color_max)<<")\n";

    // Hintergrund / Scale
    std::cout << "background_color=("<<v3(s.background_color)<<")  cover_scale="<<coverScale << "\n";

    // Initial matrices (einmalig)
    std::cout << "initial_navigation=" << LamureUtil::matConv4F(s.initial_navigation) << "\n";
    std::cout << "initial_view="       << LamureUtil::matConv4F(s.initial_view)       << "\n";

    // Transforms (nur Anzahl + kurze Vorschau)
    std::cout << "transforms: " << s.transforms.size() << " (showing up to 3 keys)\n";
    size_t shown=0;
    for(const auto& kv : s.transforms){
        if(shown++>=3){ std::cout<<"  ...\n"; break; }
        std::cout << "  [" << kv.first << "] scm::mat4d present\n";
    }

    // Measurement
    std::cout << "measurement: dir='"<<s.measurement_dir<<"' name='"<<s.measurement_name<<"'\n";
    if(!s.measurement_segments.empty()){
        std::cout << "measurement_segments.count="<<s.measurement_segments.size()<<"\n";
        const size_t N=std::min<size_t>(s.measurement_segments.size(),3);
        for(size_t i=0;i<N;++i){
            const auto& q=s.measurement_segments[i];
            std::cout<<"  ["<<i<<"] tra("<<q.tra.x()<<","<<q.tra.y()<<","<<q.tra.z()
                <<") rot("<<q.rot.x()<<","<<q.rot.y()<<","<<q.rot.z()
                <<") vT="<<q.transSpeed<<" vR="<<q.rotSpeed<<"\n";
        }
        if(s.measurement_segments.size()>3) std::cout<<"  ...\n";
    }

    std::cout << "--- end settings ---\n";
}


bool Lamure::init2() {
	std::cout << "init2()" << std::endl;

    plugin->loadSettingsFromCovise();
    //dumpSettings();

    if (plugin->m_settings.provenance && plugin->m_settings.json != "") {
        std::cout << "json: " << plugin->m_settings.json << std::endl;
        if (plugin->m_settings.provenance && !plugin->m_settings.json.empty()) {
            std::cout << "Provenance data is valid. Loading from: " << plugin->m_settings.json << std::endl;
            plugin->m_data_provenance = lamure::ren::Data_Provenance::parse_json(plugin->m_settings.json);
            std::cout << "size of provenance: " << plugin->m_data_provenance.get_size_in_bytes() << std::endl;
        }
        else { std::cout << "Provenance data not found or incomplete. Disabling provenance-based shaders." << std::endl; }
    }

    const osg::GraphicsContext::Traits *traits = opencover::coVRConfig::instance()->windows[0].context->getTraits();
    uint32_t render_width = traits->width / plugin->m_settings.frame_div;
    uint32_t render_height = traits->height / plugin->m_settings.frame_div;

    lamure::ren::policy* policy = lamure::ren::policy::get_instance();
    policy->set_max_upload_budget_in_mb(plugin->m_settings.upload);
    policy->set_render_budget_in_mb(plugin->m_settings.vram);
    policy->set_out_of_core_budget_in_mb(plugin->m_settings.ram);
    policy->set_window_width(render_width);
    policy->set_window_height(render_height);

    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();

    uint16_t num_models = 0;
    for (const auto &input_file : plugin->m_settings.models)
    {
        lamure::model_t model_id = database->add_model(input_file, std::to_string(num_models));
        plugin->m_model_info.model_transformations.push_back(plugin->m_settings.transforms[num_models] * scm::math::mat4d(scm::math::make_translation(database->get_model(num_models)->get_bvh()->get_translation())));
        ++num_models;
    }
    plugin->m_settings.num_models = num_models;
	std::cerr << "hostname: " << covise::coConfigConstants::getHostname() << std::endl;
    //osg_util::waitForOpenGLContext();
	m_ui->setupUi();
    m_renderer->init();
    opencover::cover->getObjectsRoot()->addChild(m_renderer->getGroup());
    applyShaderToRendererFromSettings();
    opencover::coVRNavigationManager::instance()->setNavMode("Point");
    if (m_settings.use_initial_view || m_settings.use_initial_navigation)
        plugin->applyInitialTransforms();

    return 1;
}

void Lamure::key(int type, int keySym, int /*mod*/) {
    const int idx = clampKeyIndex(keySym);
    if (type == osgGA::GUIEventAdapter::KEYDOWN) m_keyDown_.set(idx, true);
    else if (type == osgGA::GUIEventAdapter::KEYUP) m_keyDown_.set(idx, false);
    // kein return – Signatur ist void
}

void Lamure::preFrame() {
    if (m_measurement && !m_measurement->isActive()) {
        stopMeasurement();
    }
    float deltaTime = std::clamp(float(opencover::cover->frameDuration()), 1.0f/60.0f, 1.0f/15.0f);
    float moveAmount = 1000.0f * deltaTime;
    osg::Matrix m = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();

#ifdef _WIN32
    if (GetAsyncKeyState(VK_NUMPAD4) & 0x8000) m.postMult(osg::Matrix::translate(+moveAmount, 0.0, 0.0));
    if (GetAsyncKeyState(VK_NUMPAD6) & 0x8000) m.postMult(osg::Matrix::translate(-moveAmount, 0.0, 0.0));
    if (GetAsyncKeyState(VK_NUMPAD8) & 0x8000) m.postMult(osg::Matrix::translate(0.0, -moveAmount, 0.0));
    if (GetAsyncKeyState(VK_NUMPAD5) & 0x8000) m.postMult(osg::Matrix::translate(0.0, +moveAmount, 0.0));
#endif

    opencover::VRSceneGraph::instance()->getTransform()->setMatrix(m);
}


void Lamure::startMeasurement() {
    std::cout << "startMeasurement(): " << m_ui->getMeasureButton()->state() << std::endl;
    if (m_settings.measurement_segments.empty()) {
        std::cerr << "[Lamure] No measurement segments.\n";
        return;
    }

    if (m_settings.use_initial_navigation || m_settings.use_initial_view) {
        applyInitialTransforms();
    }

    rendering_scheme = opencover::VRViewer::instance()->getRunFrameScheme();
    opencover::VRViewer::instance()->setRunFrameScheme(osgViewer::Viewer::CONTINUOUS);
    prev_frame_rate_ = opencover::coVRConfig::instance()->frameRate();
    const float target_unlimited_fps = 1000.0f; // effektiv „keine“ Kappung
    if (prev_frame_rate_ > 0.f && prev_frame_rate_ < target_unlimited_fps) {
        opencover::coVRConfig::instance()->setFrameRate(target_unlimited_fps);
        fps_cap_modified_ = true;
    } else {
        fps_cap_modified_ = false;
    }

    osg::ref_ptr ds = osg::DisplaySettings::instance();
    prev_vsync_frames_ = ds->getSyncSwapBuffers();
    if (prev_vsync_frames_ != 0u) {
        ds->setSyncSwapBuffers(0u);
        vsync_modified_ = true;
    } else {
        vsync_modified_ = false;
    }
    const std::string outFile = buildMeasurementOutputPath();
    m_measurement = std::make_unique<LamureMeasurement>(this, opencover::VRViewer::instance(), m_settings.measurement_segments, outFile);
}

void Lamure::stopMeasurement() {
    std::cout << "stopMeasurement(): " << m_ui->getMeasureButton()->state() << std::endl;
    if (!m_measurement) return;
    if (opencover::VRViewer::instance() && opencover::VRViewer::instance()->getCamera()) {
        opencover::VRViewer::instance()->getCamera()->setPreDrawCallback(nullptr);
        opencover::VRViewer::instance()->getCamera()->setPostDrawCallback(nullptr);
    }
    m_measurement->stop();
    m_measurement->writeLogAndStop();
    m_measurement.reset();
    m_ui->getMeasureButton()->setState(false);
    opencover::VRViewer::instance()->setRunFrameScheme(rendering_scheme);
    if (fps_cap_modified_) { opencover::coVRConfig::instance()->setFrameRate(prev_frame_rate_); }
    if (vsync_modified_) { osg::DisplaySettings::instance()->setSyncSwapBuffers(prev_vsync_frames_); }
}


void Lamure::addMarkMs(MarkField f, double ms) noexcept
{
    if (!m_measurement) return; // nur akkumulieren, wenn Messung aktiv

    switch (f) {
    case MarkField::DrawCB_Total: m_marks.draw_cb_ms      = ms; break;
    case MarkField::Dispatch:     m_marks.dispatch_ms     = ms; break;
    case MarkField::ContextBind:  m_marks.context_bind_ms = ms; break;
    case MarkField::Estimates:    m_marks.estimates_ms    = ms; break;
    case MarkField::Pass1:        m_marks.pass1_ms        = ms; break;
    case MarkField::Pass2:        m_marks.pass2_ms        = ms; break;
    case MarkField::Pass3:        m_marks.pass3_ms        = ms; break;
    case MarkField::SinglePass:   m_marks.singlepass_ms   = ms; break;
    default: break;
    }
}


void Lamure::applyShaderToRendererFromSettings() {
    if (!m_renderer) return;
    // Falls jemand s.shader_type später überschreibt, hier noch mal robust ableiten:
    auto& s = m_settings;
    auto decideShaderType = [&](){
        using ST = LamureRenderer::ShaderType;
        if (s.splatting) return ST::SurfelMultipass;
        if (s.surfel)    return s.lighting ? ST::SurfelColorLighting
            : (s.coloring ? ST::SurfelColor : ST::Surfel);
        return s.lighting ? ST::PointColorLighting
            : (s.coloring ? ST::PointColor : ST::Point);
        };
    s.shader_type = decideShaderType();
    m_renderer->setActiveShaderType(s.shader_type);
}


void Lamure::applyInitialTransforms(){
    auto* viewer = opencover::VRViewer::instance();
    auto* cam    = viewer ? viewer->getCamera() : nullptr;
    if(!opencover::cover || !cam) return;

    if(m_settings.use_initial_navigation){
        opencover::VRSceneGraph::instance()->getTransform()->setMatrix(m_settings.initial_navigation);
    }

    if(m_settings.use_initial_view){
        cam->setViewMatrix(m_settings.initial_view);
    }
}


std::vector<LamureMeasurement::Segment>
Lamure::parseMeasurementSegments(const std::string& cfg) const {
    std::vector<LamureMeasurement::Segment> out;
    if (cfg.empty()) return out;
    const auto trim = [](const std::string& s)->std::string {
        const char* ws = " \t\r\n";
        size_t b = s.find_first_not_of(ws), e = s.find_last_not_of(ws);
        return (b == std::string::npos) ? std::string() : s.substr(b, e - b + 1);
        };
    const auto split = [&](const std::string& s, char sep)->std::vector<std::string> {
        std::vector<std::string> v; std::string tok; std::stringstream ss(s);
        while (std::getline(ss, tok, sep)) v.push_back(trim(tok));
        return v;
        };
    const auto parseVec3 = [&](const std::string& s, osg::Vec3& v)->bool {
        auto t = split(s, ','); if (t.size() != 3) return false;
        try { v.set(std::stof(t[0]), std::stof(t[1]), std::stof(t[2])); }
        catch (...) { return false; }
        return true;
        };
    const auto parseF = [&](const std::string& s, float& f)->bool {
        try { f = std::stof(s); } catch (...) { return false; }
        return true;
        };
    const auto segs = split(cfg, ';');
    out.reserve(segs.size());
    for (const auto& segStr : segs) {
        if (segStr.empty()) continue;
        auto parts = split(segStr, '|');
        if (parts.size() != 4) { 
            std::cerr << "[Lamure] Bad segment (need 4 parts): \"" << segStr << "\"\n";
            continue; 
        }
        osg::Vec3 tra, rot; float vt = 0.f, vr = 0.f;
        if (!parseVec3(parts[0], tra) || !parseVec3(parts[1], rot) || !parseF(parts[2], vt) || !parseF(parts[3], vr)) {
            std::cerr << "[Lamure] Bad segment tokens: \"" << segStr << "\" (dx,dy,dz|rx,ry,rz|v_trans|v_rot)\n";
            continue;
        }
        out.push_back(LamureMeasurement::Segment{tra, rot, vt, vr});
    }

    return out;
}


std::string Lamure::buildMeasurementOutputPath() const {
    namespace fs = std::filesystem;
    fs::path dir = m_settings.measurement_dir.empty() ? fs::current_path() : fs::path(m_settings.measurement_dir);
    std::error_code ec; fs::create_directories(dir, ec); if (ec) std::cerr << "[Lamure] create_directories " << dir << " failed: " << ec.message() << "\n";

    std::string name = m_settings.measurement_name.empty() ? "measurement.txt" : m_settings.measurement_name;
    fs::path np(name);
    std::string ext = np.has_extension() ? np.extension().string() : ".txt"; if (ext.empty() || ext[0] != '.') ext = "." + ext;
    std::string stem = np.stem().string();

    // ggf. manuell gesetztes _NNNN am Ende entfernen
    stem = std::regex_replace(stem, std::regex("_(\\d+)$"), "");

    auto rxEscape = [](const std::string& s){ std::string r; r.reserve(s.size()*2);
    for(char c: s){ switch(c){ case '.': case '^': case '$': case '|': case '(': case ')': case '[': case ']': case '*': case '+': case '?': case '{': case '}': case '\\': r.push_back('\\'); default: r.push_back(c);} } return r; };

    // 1) Höchste Nummer aus ANY stem_####*.* (Frames/Summary/etc. inklusive)
    const std::regex pat("^" + rxEscape(stem) + "_(\\d+)(?:$|[^0-9].*)", std::regex::icase);
    unsigned maxN = 0;
    for (const auto& de : fs::directory_iterator(dir)) {
        if (!de.is_regular_file()) continue;
        const std::string fn = de.path().filename().string();
        std::smatch m;
        if (std::regex_search(fn, m, pat) && m.size() >= 2) {
            try { unsigned v = static_cast<unsigned>(std::stoul(m[1].str())); if (v > maxN) maxN = v; } catch (...) {}
        }
    }

    // 2) Kollisionen vermeiden: prüfe Prefix stem_#### gegen alle Dateien
    auto anyWithPrefix = [&](unsigned n)->bool{
        std::ostringstream os; os << stem << '_' << std::setw(4) << std::setfill('0') << n;
        const std::string pref = os.str();
        for (const auto& de : fs::directory_iterator(dir)) {
            if (!de.is_regular_file()) continue;
            const std::string fn = de.path().filename().string();
            if (fn.rfind(pref, 0) == 0) return true; // beginnt mit Prefix?
        }
        return false;
        };

    unsigned next = maxN + 1;
    while (anyWithPrefix(next)) ++next;

    std::ostringstream num; num << std::setw(4) << std::setfill('0') << next;
    fs::path out = dir / (stem + "_" + num.str() + ext);
    return fs::absolute(out).string();
}


bool Lamure::writeSettingsJson(const Lamure::Settings& s, const std::string& outPath)
{
    namespace fs = std::filesystem;

    // --- Helpers (lokal, ohne statics) ---
    auto esc = [](const std::string& str){
        std::string r; r.reserve(str.size()+8);
        for(char c: str){
            switch(c){
            case '\"': r += "\\\""; break;
            case '\\': r += "\\\\"; break;
            case '\n': r += "\\n";  break;
            case '\r': r += "\\r";  break;
            case '\t': r += "\\t";  break;
            default:   r.push_back(c);
            }
        }
        return r;
        };
    auto v3f_scm = [](const scm::math::vec3f& v){
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(6);
        o<<'['<<v.x<<','<<v.y<<','<<v.z<<']'; return o.str();
        };
    auto mat4_osg = [](const osg::Matrixd& M){
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(6);
        o<<"["
            <<"["<<M(0,0)<<","<<M(0,1)<<","<<M(0,2)<<","<<M(0,3)<<"],"
            <<"["<<M(1,0)<<","<<M(1,1)<<","<<M(1,2)<<","<<M(1,3)<<"],"
            <<"["<<M(2,0)<<","<<M(2,1)<<","<<M(2,2)<<","<<M(2,3)<<"],"
            <<"["<<M(3,0)<<","<<M(3,1)<<","<<M(3,2)<<","<<M(3,3)<<"]"
            <<"]";
        return o.str();
        };
    auto mat4_scm = [](const scm::math::mat4d& m){
        // Wir interpretieren m[0..15] als 4x4 in der sichtbaren Reihenfolge:
        // [ [m00 m01 m02 m03],
        //   [m10 m11 m12 m13],
        //   [m20 m21 m22 m23],
        //   [m30 m31 m32 m33] ]
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(6);
        o << "["
            << "[" << m[0]  << "," << m[1]  << "," << m[2]  << "," << m[3]  << "],"
            << "[" << m[4]  << "," << m[5]  << "," << m[6]  << "," << m[7]  << "],"
            << "[" << m[8]  << "," << m[9]  << "," << m[10] << "," << m[11] << "],"
            << "[" << m[12] << "," << m[13] << "," << m[14] << "," << m[15] << "]"
            << "]";
        return o.str();
        };

    // --- NEU: transforms-Map serialisieren mit obigem Helfer ---
    auto mapTransforms = [&](const std::map<uint32_t, scm::math::mat4d>& m){
        std::ostringstream o; o << '{';
        bool first = true;
        for (const auto& [id, mat] : m) {
            if (!first) o << ',';
            first = false;
            o << '\"' << id << '\"' << ':' << mat4_scm(mat);
        }
        o << '}';
        return o.str();
        };
    auto segs = [](const std::vector<LamureMeasurement::Segment>& S){
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(6);
        o<<'[';
        for(size_t i=0;i<S.size();++i){
            if(i) o<<',';
            o<<"{"
                << "\"tra\":["<<S[i].tra.x()<<','<<S[i].tra.y()<<','<<S[i].tra.z()<<"],"
                << "\"rot\":["<<S[i].rot.x()<<','<<S[i].rot.y()<<','<<S[i].rot.z()<<"],"
                << "\"transSpeed\":"<<S[i].transSpeed<<","
                << "\"rotSpeed\":"  <<S[i].rotSpeed
                << "}";
        }
        o<<']'; return o.str();
        };
    auto vecStr = [&](const std::vector<std::string>& v){
        std::ostringstream o; o<<'[';
        for(size_t i=0;i<v.size();++i){ if(i) o<<','; o<<'\"'<<esc(v[i])<<'\"'; }
        o<<']'; return o.str();
        };
    auto vecU32 = [&](const std::vector<uint32_t>& v){
        std::ostringstream o; o<<'[';
        for(size_t i=0;i<v.size();++i){ if(i) o<<','; o<<v[i]; }
        o<<']'; return o.str();
        };
    auto vecF = [&](const std::vector<float>& v){
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(6);
        o<<'[';
        for(size_t i=0;i<v.size();++i){ if(i) o<<','; o<<v[i]; }
        o<<']'; return o.str();
        };
    auto mapU32Str = [&](const std::map<uint32_t, std::string>& m){
        std::ostringstream o; o<<'{';
        bool first=true;
        for(const auto& [id,txt] : m){
            if(!first) o<<','; first=false;
            o<<'\"'<<id<<'\"'<<":\"" << esc(txt) << '\"';
        }
        o<<'}'; return o.str();
        };

    // Zielordner (non-fatal bei Fehlern)
    try {
        fs::path p(outPath);
        if (p.has_parent_path()) {
            std::error_code ec;
            fs::create_directories(p.parent_path(), ec);
            if (ec) {
                std::cerr << "[LamureUtil] create_directories failed: "
                    << p.parent_path().string() << " : " << ec.message() << "\n";
            }
        }
    } catch (...) {}

    std::ofstream js(outPath, std::ios::out | std::ios::trunc);
    if (!js) {
        std::cerr << "[LamureUtil] Failed to open settings JSON: " << outPath << "\n";
        return false;
    }

    // Komma-sicher sammeln
    std::vector<std::string> items; items.reserve(170);
    auto add_raw = [&](const std::string& k, const std::string& v){ items.emplace_back("  \""+k+"\": "+v); };
    auto add_str = [&](const std::string& k, const std::string& v){ items.emplace_back("  \""+k+"\": \""+esc(v)+"\""); };
    auto add_bool= [&](const std::string& k, bool b){ items.emplace_back("  \""+k+"\": "+std::string(b?"true":"false")); };
    auto add_i   = [&](const std::string& k, long long v){ items.emplace_back("  \""+k+"\": "+std::to_string(v)); };
    auto add_f   = [&](const std::string& k, double v){
        std::ostringstream o; o.setf(std::ios::fixed); o.precision(6); o<<v;
        items.emplace_back("  \""+k+"\": "+o.str());
        };

    // --- Primitive bestimmen (Point/Surfel/Splat) ---
    auto determinePrimitive = [&]()->std::string {
        int cnt = (s.point?1:0) + (s.surfel?1:0) + (s.splatting?1:0);
        if (cnt == 1) {
            if (s.point)    return "Point";
            if (s.surfel)   return "Surfel";
            /*s.splatting*/ return "Splat";
        }
        if (cnt == 0) {
            // Fallback: versuche aus s.shader zu raten, sonst "Point"
            std::string sh = s.shader; 
            std::string shLow = sh; std::transform(shLow.begin(), shLow.end(), shLow.begin(), ::tolower);
            if (shLow.find("surfel") != std::string::npos) return "Surfel";
            if (shLow.find("splat")  != std::string::npos) return "Splat";
            if (shLow.find("point")  != std::string::npos) return "Point";
            return "Point";
        }
        // Mehrere Flags aktiv – kennzeichne als "Mixed"
        return "Mixed";
        }();

    // Dateien / Namen
    add_str("measurement_dir",  s.measurement_dir);
    add_str("measurement_name", s.measurement_name);

    // Modelle & Auswahl
    add_raw("models",            vecStr(s.models));
    add_raw("initial_selection", vecU32(s.initial_selection));

    // Budgets / LOD
    add_i ("frame_div", s.frame_div);
    add_i ("vram",      s.vram);
    add_i ("ram",       s.ram);
    add_i ("upload",    s.upload);
    add_bool("lod_update", s.lod_update);
    add_f ("lod_error", s.lod_error);

    // GUI / Travel
    add_i ("gui",          s.gui);
    add_i ("travel",       s.travel);
    add_f ("travel_speed", s.travel_speed);

    // Shader / Primitive & Skalen
    add_str("shader",      s.shader);
    add_str("primitive",   determinePrimitive); // <<<< hier die gewünschte Ableitung
    add_f ("scale_element",      s.scale_element);
    add_f ("scale_point",        s.scale_point);
    add_f ("scale_surfel",       s.scale_surfel);
    add_f ("scale_radius",       s.scale_radius);
    add_f ("radius_scale_gamma", s.scale_radius_gamma);
    add_f ("min_radius",         s.min_radius);
    add_f ("max_radius",         s.max_radius);
    add_f ("min_screen_size",    s.min_screen_size);
    add_f ("max_screen_size",    s.max_screen_size);
    add_f ("max_radius_cut",     s.max_radius_cut);

    // Flags / Visual toggles
    add_bool("provenance",              s.provenance);
    add_bool("create_aux_resources",    s.create_aux_resources);
    add_bool("face_eye",                s.face_eye);
    add_i   ("vis",                     s.vis);
    add_i   ("show_normals",            s.show_normals); // int32_t im Struct -> als Zahl schreiben
    add_bool("show_accuracy",           s.show_accuracy);
    add_bool("show_radius_deviation",   s.show_radius_deviation);
    add_bool("show_output_sensitivity", s.show_output_sensitivity);
    add_bool("show_sparse",             s.show_sparse);
    add_bool("show_views",              s.show_views);
    add_bool("show_photos",             s.show_photos);
    add_bool("show_octrees",            s.show_octrees);
    add_bool("show_bvhs",               s.show_bvhs);
    add_bool("show_pvs",                s.show_pvs);
    add_bool("pvs_culling",             s.pvs_culling);
    add_bool("use_pvs",                 s.use_pvs);

    // Kanal / Aux Parameter
    add_i ("channel",            s.channel);
    add_f ("aux_point_size",     s.aux_point_size);
    add_f ("aux_point_distance", s.aux_point_distance);
    add_f ("aux_point_scale",    s.aux_point_scale);
    add_f ("aux_focal_length",   s.aux_focal_length);
    add_i ("max_brush_size",     s.max_brush_size);

    // Lighting / ToneMapping
    add_f   ("point_light_intensity", s.point_light_intensity);
    add_f   ("ambient_intensity",     s.ambient_intensity);
    add_f   ("specular_intensity",    s.specular_intensity);
    add_f   ("shininess",             s.shininess);
    add_f   ("gamma",                 s.gamma);
    add_bool("use_tone_mapping",      s.use_tone_mapping);
    add_raw ("point_light_pos",       v3f_scm(s.point_light_pos));

    // Heatmap
    add_bool("heatmap",           s.heatmap);
    add_f   ("heatmap_min",       s.heatmap_min);
    add_f   ("heatmap_max",       s.heatmap_max);
    add_raw ("heatmap_color_min", v3f_scm(s.heatmap_color_min));
    add_raw ("heatmap_color_max", v3f_scm(s.heatmap_color_max));

    // Multi-Pass
    add_f ("depth_range", s.depth_range);
    add_f ("flank_lift",  s.flank_lift);

    // Pfade / Dateien
    add_str("atlas_file",       s.atlas_file);
    add_str("json",             s.json);
    add_str("pvs",              s.pvs);
    add_str("background_image", s.background_image);

    // Farben / Background
    add_raw("bvh_color",        vecF(s.bvh_color));
    add_raw("frustum_color",    vecF(s.frustum_color));
    add_raw("background_color", v3f_scm(s.background_color));

    // Sichtbarkeiten UI
    add_bool("show_pointcloud",  s.show_pointcloud);
    add_bool("show_boundingbox", s.show_boundingbox);
    add_bool("show_frustum",     s.show_frustum);
    add_bool("show_coord",       s.show_coord);
    add_bool("show_text",        s.show_text);
    add_bool("show_sync",        s.show_sync);
    add_bool("show_notify",      s.show_notify);

    // Initial Matrices
    add_bool("use_initial_navigation", s.use_initial_navigation);
    add_bool("use_initial_view",       s.use_initial_view);
    add_bool("initial_tf_overrides",   s.initial_tf_overrides);
    add_raw ("initial_navigation",     mat4_osg(s.initial_navigation));
    add_raw ("initial_view",           mat4_osg(s.initial_view));

    // Pro-Modell Daten
    add_raw("transforms", mapTransforms(s.transforms));
    add_raw("aux",        mapU32Str(s.aux));

    // Messpfade
    add_raw("measurement_segments", segs(s.measurement_segments));

    // Messungs-Modi/Flags
    add_bool("measure_off",    s.measure_off);
    add_bool("measure_light",  s.measure_light);
    add_bool("measure_full",   s.measure_full);
    add_i   ("measure_sample", s.measure_sample);

    // Modelle (Meta)
    add_i("num_models", s.num_models);

    // Laufzeitobjekte nur als Zähler (keine Deep-Serialisierung)
    add_i("octrees_count", static_cast<long long>(s.octrees.size()));
    add_i("views_count",   static_cast<long long>(s.views.size()));

    // --- Ausgabe ---
    js << "{\n";
    for(size_t i=0;i<items.size();++i){
        js << items[i] << (i+1<items.size() ? ",\n" : "\n");
    }
    js << "}\n";
    js.flush();

    std::cout << "[LamureUtil] Settings JSON written: " << outPath << "\n";
    return true;
}

