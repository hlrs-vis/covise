//local
#include "Lamure.h" 
#include "LamureEditTool.h"
#include "gl_state.h"
#include "osg_util.h"

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
#include <mutex>
#include <filesystem>
#include <memory>
#include <thread>
#include <cmath>
#ifdef WIN32
#define GLFW_EXPOSE_NATIVE_WIN32
#include <winbase.h>
#endif

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
#include <cover/coVRNavigationManager.h>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>
#include <cover/coVRPluginList.h>
#include <numeric>
#include <osg/Geode>
#include <osg/MatrixTransform>


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

    class RenderPauseGuard {
    public:
        RenderPauseGuard(LamureRenderer* renderer, uint32_t extraDrainFrames)
            : m_renderer(renderer)
        {
            if (m_renderer)
                m_shouldResume = m_renderer->pauseAndWaitForIdle(extraDrainFrames);
        }

        ~RenderPauseGuard()
        {
            if (m_renderer && m_shouldResume)
                m_renderer->resumeRendering();
        }

        RenderPauseGuard(const RenderPauseGuard&) = delete;
        RenderPauseGuard& operator=(const RenderPauseGuard&) = delete;

    private:
        LamureRenderer* m_renderer;
        bool m_shouldResume{false};
    };
} // namespace

namespace {
    constexpr const char* kLamureRegistrationKey = "LamureRegister";
}

static std::mutex g_settings_mutex;
static std::mutex g_load_bvh_mutex;
static std::atomic<bool> g_is_resetting(false);
// replaced by Lamure members (m_bootstrap_files and resolver)

#if 0
#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllexport) DWORD NvOptimusEnablement = 1;
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#ifdef __cplusplus
}
#endif
#endif

COVERPLUGIN(Lamure)
Lamure* Lamure::plugin = nullptr;

static opencover::FileHandler handler = {
    NULL, 
    Lamure::loadBvh,
    Lamure::unloadBvh, 
    "bvh"
};


Lamure::Lamure() :coVRPlugin(COVER_PLUGIN_NAME), opencover::ui::Owner("LamurePlugin", opencover::cover->ui)
{
    fprintf(stderr, "LamurePlugin\n");
    opencover::coVRFileManager::instance()->registerFileHandler(&handler);
	plugin = this;
    m_ui = std::make_unique<LamureUI>(this, "LamureUI");
    m_renderer = std::make_unique<LamureRenderer>(this);
    m_edit_tool = std::make_unique<LamureEditTool>(this);
    m_edit_tool->setBrushAction(m_edit_action);
}


Lamure* Lamure::instance()
{
	return plugin;
}


Lamure::~Lamure()
{
    fprintf(stderr, "LamurePlugin::~LamurePlugin\n");

    if (m_edit_tool)
        m_edit_tool->disable();
    if (m_renderer && m_lamure_grp)
        m_renderer->destroyEditBrushNode(m_lamure_grp.get());
    opencover::cover->getObjectsRoot()->removeChild(m_lamure_grp);
    opencover::coVRFileManager::instance()->unregisterFileHandler(&handler);
}


void Lamure::setModelVisible(uint16_t idx, bool v) {
    std::lock_guard<std::mutex> lock(g_settings_mutex);
    auto &visible = m_model_info.model_visible;
    if (idx >= visible.size()) return;
    visible[idx] = v;
}

bool Lamure::isModelVisible(uint16_t idx) const {
    std::lock_guard<std::mutex> lock(g_settings_mutex);
    const auto &visible = m_model_info.model_visible;
    return idx < visible.size() ? visible[idx] : false;
}

void Lamure::setEditMode(bool enabled) {
    if (m_edit_mode == enabled)
        return;
    m_edit_mode = enabled;

    if (!m_edit_mode) {
        if (m_edit_tool)
            m_edit_tool->disable();
    } else {
        if (m_edit_tool) {
            m_edit_tool->setBrushAction(m_edit_action);
            m_edit_tool->enable();
            m_edit_tool->update();
        }
    }

    if (m_settings.show_notify) {
        std::cout << "[Lamure] Edit mode " << (enabled ? "enabled" : "disabled") << std::endl;
    }
    if (m_ui) {
        if (auto* btn = m_ui->getEditModeButton()) {
            if (btn->state() != enabled)
                btn->setState(enabled);
        }
    }
}

void Lamure::setEditAction(LamureEditTool::BrushAction action) {
    m_edit_action = action;
    if (m_edit_tool)
        m_edit_tool->setBrushAction(action);
}


int Lamure::unloadBvh(const char* filename, const char* /*covise_key*/)
{
    if (plugin && plugin->getSettings().show_notify) {
        std::cout << "[Lamure] unloadBvh: filename=" << (filename ? filename : "null") << std::endl;
    }
    
    if (!filename || !Lamure::plugin) return 0;

    std::string path = std::filesystem::absolute(std::filesystem::path(filename)).string();
#ifdef _WIN32
    std::replace(path.begin(), path.end(), '\\', '/');
#endif
    if (path.empty()) return 0;

    std::lock_guard<std::mutex> lock(g_settings_mutex);

    auto it_idx = plugin->m_model_idx.find(path);
    if (it_idx == plugin->m_model_idx.end()) {
        if (plugin->m_settings.show_notify) {
            std::printf("[Lamure] unloadBvh: '%s' not found (no-op)\n", path.c_str());
        }
        return 0;
    }
    uint16_t idx = it_idx->second;

    if (idx < plugin->m_model_info.model_visible.size()) {
        plugin->m_model_info.model_visible[idx] = false;
    }

    auto it_node = plugin->m_model_nodes.find(path);
    if (it_node != plugin->m_model_nodes.end()) {
        osg::Group* node = it_node->second.get();
        if (node) {
            while (node->getNumParents() > 0) {
                osg::Node* parentNode = node->getParent(0);
                auto* parentGroup = dynamic_cast<osg::Group*>(parentNode);
                if (!parentGroup) {
                    break;
                }
                parentGroup->removeChild(node);
            }
        }
        plugin->m_model_nodes.erase(it_node);
    }

    plugin->m_model_idx.erase(it_idx);
    plugin->m_pendingTransformUpdate.erase(path);
    plugin->m_vrmlTransforms.erase(path);
    plugin->m_scene_nodes.erase(path);
    plugin->m_registeredFiles.erase(path);
    plugin->m_model_source_keys.erase(path);

    return 1;
}


int Lamure::loadBvh(const char *filename, osg::Group *parent, const char *covise_key)
{
    std::lock_guard<std::mutex> lock(g_load_bvh_mutex);
    if (!filename || !plugin)
        return 0;

    std::string file = std::filesystem::absolute(std::filesystem::path(filename)).string();
#ifdef _WIN32
    std::replace(file.begin(), file.end(), '\\', '/');
#endif

    const bool isMenuCall = (covise_key && std::strcmp(covise_key, kLamureRegistrationKey) == 0);
    const bool isRegistrationCall = (isMenuCall && parent == nullptr);
    const bool isMenuReload = (isMenuCall && parent != nullptr);

    if (isRegistrationCall)
        return 0;

    if (!isMenuCall) {
        std::string sourceKey = "<direct>";
        if (covise_key && covise_key[0])
            sourceKey = covise_key;
        plugin->m_model_source_keys[file] = sourceKey;
    } else if (plugin->m_model_source_keys.find(file) == plugin->m_model_source_keys.end()) {
        plugin->m_model_source_keys[file] = std::string();
    }

    SceneNodes *scene_nodes = nullptr;
    {
        auto itScene = plugin->m_scene_nodes.find(file);
        if (itScene == plugin->m_scene_nodes.end()) {
            SceneNodes nodes;
            nodes.root = new osg::Group();
            nodes.root->setName(file);
            nodes.config = new osg::MatrixTransform();
            nodes.config->setName(file + "_config");
            nodes.bvh = new osg::MatrixTransform();
            nodes.bvh->setName(file + "_bvh");
            nodes.root->addChild(nodes.config);
            nodes.config->addChild(nodes.bvh);
            itScene = plugin->m_scene_nodes.emplace(file, std::move(nodes)).first;
        }
        scene_nodes = &itScene->second;
    }

    auto ensureSceneGraph = [&](SceneNodes &nodes) {
        if (!nodes.root) {
            nodes.root = new osg::Group();
            nodes.root->setName(file);
        }
        if (!nodes.config) {
            nodes.config = new osg::MatrixTransform();
            nodes.config->setName(file + "_config");
        }
        if (!nodes.bvh) {
            nodes.bvh = new osg::MatrixTransform();
            nodes.bvh->setName(file + "_bvh");
        }
        if (!nodes.root->containsNode(nodes.config))
            nodes.root->addChild(nodes.config);
        if (!nodes.config->containsNode(nodes.bvh))
            nodes.config->addChild(nodes.bvh);
        };

    ensureSceneGraph(*scene_nodes);
    osg::Group *modelNode = scene_nodes->root.get();

    auto captureVrmlTransform = [&](const osg::Matrix &mat) {
        const auto newMat = LamureUtil::matConv4D(mat);
        bool needsReload = false;
        {
            std::lock_guard<std::mutex> lock(g_settings_mutex);
            auto itPrev = plugin->m_vrmlTransforms.find(file);
            needsReload = (itPrev != plugin->m_vrmlTransforms.end() && itPrev->second != newMat);
            plugin->m_vrmlTransforms[file] = newMat;
        }
        if (needsReload) {
            plugin->m_reloaded_files.insert(file);
            plugin->m_reload_imminent = true;
        }
        plugin->ensureFileMenuEntry(file);
        };

    if (!plugin->initialized) {
        if (std::find(plugin->m_bootstrap_files.begin(), plugin->m_bootstrap_files.end(), file) == plugin->m_bootstrap_files.end())
            plugin->m_bootstrap_files.push_back(file);

        if (parent) {
            bool hasParent = false;
            for (unsigned i = 0; i < modelNode->getNumParents(); ++i) {
                if (modelNode->getParent(i) == parent) { hasParent = true; break; }
            }
            if (!hasParent)
                parent->addChild(modelNode);
            plugin->m_pendingTransformUpdate[file] = modelNode;
            if (auto *mt = dynamic_cast<osg::MatrixTransform *>(parent))
                captureVrmlTransform(mt->getMatrix());
        }
        std::lock_guard<std::mutex> settings_lock(g_settings_mutex);
        plugin->m_model_nodes[file] = modelNode;
        return 1;
    }

    std::lock_guard<std::mutex> settings_lock(g_settings_mutex);

    auto itExisting = plugin->m_model_nodes.find(file);
    if (!isMenuReload && itExisting != plugin->m_model_nodes.end()) {
        osg::ref_ptr<osg::Group> node = itExisting->second;
        if (!node)
            return 0;

        if (parent) {
            bool hasParent = false;
            for (unsigned i = 0; i < node->getNumParents(); ++i) {
                if (node->getParent(i) == parent) { hasParent = true; break; }
            }
            if (!hasParent)
                parent->addChild(node);
            plugin->m_pendingTransformUpdate[file] = node.get();
            if (auto *mt = dynamic_cast<osg::MatrixTransform *>(parent))
                captureVrmlTransform(mt->getMatrix());
        }
        return 0;
    }

    if (std::find(plugin->m_files_to_load.begin(), plugin->m_files_to_load.end(), file) != plugin->m_files_to_load.end())
        return 0;

    if (parent) {
        bool hasParent = false;
        for (unsigned i = 0; i < modelNode->getNumParents(); ++i) {
            if (modelNode->getParent(i) == parent) { hasParent = true; break; }
        }
        if (!hasParent)
            parent->addChild(modelNode);
        plugin->m_pendingTransformUpdate[file] = modelNode;
        if (auto *mt = dynamic_cast<osg::MatrixTransform *>(parent))
            captureVrmlTransform(mt->getMatrix());
    }
    plugin->m_model_nodes[file] = modelNode;

    if (plugin->m_files_to_load.empty()) {
        plugin->m_frames_to_wait = isMenuReload ? 0 : static_cast<int>(plugin->m_settings.pause_frames);
        plugin->setBrushFrozen(true);
    }
    plugin->m_files_to_load.push_back(file);

    return 1;
}



bool Lamure::init2() {

    if (initialized)
        return false;

    loadSettingsFromCovise();
    m_settings.models = resolveAndNormalizeModels();
    updateModelDependentSettings();

    lamure::ren::policy* policy = lamure::ren::policy::get_instance();
    policy->set_max_upload_budget_in_mb(m_settings.upload);
    policy->set_render_budget_in_mb(m_settings.vram);
    policy->set_out_of_core_budget_in_mb(m_settings.ram);

    m_lamure_grp = new osg::Group();
    m_lamure_grp->setName("LamureGroup");
    opencover::cover->getObjectsRoot()->addChild(m_lamure_grp.get());

    m_ui->setupUi();

    opencover::coVRNavigationManager::instance()->setNavMode("Point");

    m_first_frame = true;
    initialized = true;

    if (m_settings.show_notify)
        std::cout << "[Lamure] Init complete; defer model load to first preFrame" << std::endl;

    return true;
}


void Lamure::preFrame() {

    // First-frame initialization
    if (m_first_frame) {
        m_first_frame = false;

        if (!m_settings.models.empty()) {
            if (m_settings.show_notify)
                std::cout << "[Lamure] First frame: start initial model load" << std::endl;

            // Load phase: when using config, register nodes via FileManager for UI
            if (m_models_from_config) {
                auto *fm = opencover::coVRFileManager::instance();
                for (const auto& model_file : m_settings.models) {
                    fm->loadFile(model_file.c_str(), nullptr, m_lamure_grp.get(), "");
                }
            }

            // Perform initial full reset
            perform_system_reset();

            if (m_settings.use_initial_view || m_settings.use_initial_navigation) {
                applyInitialTransforms();
            }
        }
    }

    // Continue staged reset if in progress
    if (m_reset_in_progress) {
        perform_system_reset();
    }

    // Dynamic loading after start
    if (!m_files_to_load.empty()) {
        if (!g_is_resetting) {
            if (m_frames_to_wait > 0) {
                m_frames_to_wait--;
            }
        }

        if (m_frames_to_wait == 0) {
            bool expected = false;
            if (g_is_resetting.compare_exchange_strong(expected, true)) {
                if (m_settings.show_notify)
                    std::cout << "[Lamure] Dynamic load triggered" << std::endl;
                perform_system_reset();
            }
        }
    }
    else if (m_reload_imminent) {
        if (!g_is_resetting) {
            bool expected = false;
            if (g_is_resetting.compare_exchange_strong(expected, true)) {
                if (m_settings.show_notify)
                    std::cout << "[Lamure] VRML reload detected" << std::endl;
                setBrushFrozen(true);
                perform_system_reset();
            }
        }
        m_reload_imminent = false;
        m_reloaded_files.clear();
    }

    if (!m_pendingTransformUpdate.empty()) {
        std::vector<std::string> finished;
        finished.reserve(m_pendingTransformUpdate.size());
        for (auto &entry : m_pendingTransformUpdate) {
            const std::string &path = entry.first;
            osg::Node *node = entry.second.get();
            if (!node)
                continue;

            if (node->getNumParents() == 0)
                continue;

            osg::Node *parentNode = node->getParent(0);
            auto *mt = dynamic_cast<osg::MatrixTransform *>(parentNode);
            if (!mt)
                continue;

            const auto vrmlMat = LamureUtil::matConv4D(mt->getMatrix());
            bool needsReload = false;
            {
                std::lock_guard<std::mutex> lock(g_settings_mutex);
                auto itPrev = m_vrmlTransforms.find(path);
                needsReload = (itPrev != m_vrmlTransforms.end() && itPrev->second != vrmlMat);
                m_vrmlTransforms[path] = vrmlMat;
            }
            if (needsReload) {
                m_reloaded_files.insert(path);
                m_reload_imminent = true;
            }
            ensureFileMenuEntry(path);
            finished.push_back(path);
        }
        for (const auto &path : finished) {
            m_pendingTransformUpdate.erase(path);
        }
    }

    if (m_measurement && !m_measurement->isActive()) {
        stopMeasurement();
    }

    if (m_edit_mode) {
        if (m_edit_tool)
            m_edit_tool->update();
    }

#ifdef _WIN32
    float deltaTime = std::clamp(float(opencover::cover->frameDuration()), 1.0f / 60.0f, 1.0f / 15.0f);
    float moveAmount = 1000.0f * deltaTime;
    osg::Matrix m = opencover::VRSceneGraph::instance()->getTransform()->getMatrix();
    if (GetAsyncKeyState(VK_NUMPAD4) & 0x8000) m.postMult(osg::Matrix::translate(+moveAmount, 0.0, 0.0));
    if (GetAsyncKeyState(VK_NUMPAD6) & 0x8000) m.postMult(osg::Matrix::translate(-moveAmount, 0.0, 0.0));
    if (GetAsyncKeyState(VK_NUMPAD8) & 0x8000) m.postMult(osg::Matrix::translate(0.0, -moveAmount, 0.0));
    if (GetAsyncKeyState(VK_NUMPAD5) & 0x8000) m.postMult(osg::Matrix::translate(0.0, +moveAmount, 0.0));

    {
        static SHORT lastF9 = 0;
        SHORT now = GetAsyncKeyState(VK_F9); // 0x78
        const bool down = (now & 0x8000) != 0;
        const bool wasDown = (lastF9 & 0x8000) != 0;

        if (down && !wasDown) {
            if (m_ui && m_ui->getMeasureButton()) {
                auto *btn = m_ui->getMeasureButton();
                bool newState = !btn->state();
                btn->setState(newState);
                if (btn->callback())
                    btn->callback()(newState);
            } else {
                if (!m_measurement) startMeasurement();
                else stopMeasurement();
            }
        }
        lastF9 = now;
    }
    opencover::VRSceneGraph::instance()->getTransform()->setMatrix(m);
#endif
}

void Lamure::perform_system_reset()
{
    std::lock_guard<std::mutex> lock(g_load_bvh_mutex);

    if (m_renderer && m_lamure_grp) {
        m_renderer->destroyEditBrushNode(m_lamure_grp.get());
    }

    // Phase 1: pause rendering and initiate shutdown
    if (!m_reset_in_progress) {
        if (m_settings.show_notify)
            std::cout << "[Lamure] Reset: pause and shutdown" << std::endl;
        m_renderer_paused_for_reset = m_renderer ? m_renderer->pauseAndWaitForIdle(m_settings.pause_frames) : false;
        if (m_renderer) {
            m_renderer->detachCallbacks();
            m_renderer->shutdown();
        }

        // merge dynamic files into final list (case-insensitive on Windows)
        auto &models_ref = m_settings.models;
        auto make_key = [](std::string p){
#ifdef _WIN32
            for (auto &c : p) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
            return p;
        };
        std::unordered_set<std::string> model_keys;
        model_keys.reserve(models_ref.size() + m_files_to_load.size());
        for (const auto &m : models_ref) model_keys.insert(make_key(m));
        for (const auto &file_to_load : m_files_to_load) {
            const std::string k = make_key(file_to_load);
            if (model_keys.insert(k).second) {
                models_ref.push_back(file_to_load);
            }
        }
        m_files_to_load.clear();

        // reset internal tracking structures
        m_model_idx.clear();
        m_model_info.model_transformations.clear();
        m_model_info.root_bb_min.clear();
        m_model_info.root_bb_max.clear();
        m_model_info.root_center.clear();

        // wait a few frames after shutdown
        m_post_shutdown_delay = std::max(0, static_cast<int>(m_settings.pause_frames));
        m_reset_in_progress = true;
        return;
    }

    // Phase 2: wait delay frames
    if (m_post_shutdown_delay > 0) {
        --m_post_shutdown_delay;
        return;
    }

    // Phase 3: rebuild
    auto &models = m_settings.models;
    lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
    lamure::ren::controller* controller = lamure::ren::controller::get_instance();

    const uint16_t N = static_cast<uint16_t>(models.size());
    scm::math::vec3f global_min( std::numeric_limits<float>::max()
                               , std::numeric_limits<float>::max()
                               , std::numeric_limits<float>::max());
    scm::math::vec3f global_max(-std::numeric_limits<float>::max()
                               ,-std::numeric_limits<float>::max()
                               ,-std::numeric_limits<float>::max());
    m_model_info.model_visible.assign(N, true);
    m_model_info.config_transforms.assign(N, scm::math::mat4d::identity());

    for (uint16_t mid = 0; mid < N; ++mid) {
        const std::string &model_path = models[mid];
        m_model_idx.emplace(model_path, mid);
        lamure::model_t model_id = database->add_model(model_path, std::to_string(mid));
        const auto trafo_config = m_model_info.config_transforms[mid];
        const auto trafo_bvh  = scm::math::mat4d(scm::math::make_translation(database->get_model(model_id)->get_bvh()->get_translation()));
        const auto itVrml = m_vrmlTransforms.find(model_path);
        const auto trafo_vrml = itVrml != m_vrmlTransforms.end() ? itVrml->second : scm::math::mat4d::identity();
        const auto final_trafo = trafo_vrml * trafo_config * trafo_bvh;
        m_model_info.model_transformations.push_back(final_trafo);
        auto sceneIt = m_scene_nodes.find(model_path);
        if (sceneIt != m_scene_nodes.end()) {
            if (sceneIt->second.config.valid()) {
                auto cfgCopy = trafo_config;
                sceneIt->second.config->setMatrix(LamureUtil::matConv4D(cfgCopy));
            }
            if (sceneIt->second.bvh.valid()) {
                auto bvhCopy = trafo_bvh;
                sceneIt->second.bvh->setMatrix(LamureUtil::matConv4D(bvhCopy));
            }
        }

        // Compute transformed root bbox min/max/center (approx)
        const auto &boxes = database->get_model(model_id)->get_bvh()->get_bounding_boxes();
        if (!boxes.empty()) {
            const auto &bb = boxes[0];
            const scm::math::vec4f min4(bb.min_vertex().x, bb.min_vertex().y, bb.min_vertex().z, 1.f);
            const scm::math::vec4f max4(bb.max_vertex().x, bb.max_vertex().y, bb.max_vertex().z, 1.f);
            const scm::math::vec4f cen4(bb.center().x,     bb.center().y,     bb.center().z,     1.f);

            const auto M = scm::math::mat4f(final_trafo);
            const auto tmin = M * min4;
            const auto tmax = M * max4;
            const auto tcen = M * cen4;

            scm::math::vec3f bbmin(tmin.x, tmin.y, tmin.z);
            scm::math::vec3f bbmax(tmax.x, tmax.y, tmax.z);
            m_model_info.root_bb_min.push_back(bbmin);
            m_model_info.root_bb_max.push_back(bbmax);
            m_model_info.root_center.push_back(scm::math::vec3f(tcen.x, tcen.y, tcen.z));

            global_min = scm::math::vec3f(std::min(global_min.x, bbmin.x), std::min(global_min.y, bbmin.y), std::min(global_min.z, bbmin.z));
            global_max = scm::math::vec3f(std::max(global_max.x, bbmax.x), std::max(global_max.y, bbmax.y), std::max(global_max.z, bbmax.z));
        } else {
            m_model_info.root_bb_min.push_back(scm::math::vec3f(0.f));
            m_model_info.root_bb_max.push_back(scm::math::vec3f(0.f));
            m_model_info.root_center.push_back(scm::math::vec3f(0.f));
        }
    }

    database->apply();
    m_settings.num_models = N;
    // Store aggregated bounds and center
    m_model_info.models_min = global_min;
    m_model_info.models_max = global_max;
    m_model_info.models_center = scm::math::vec3d( (global_min.x + global_max.x) * 0.5,
                                                  (global_min.y + global_max.y) * 0.5,
                                                  (global_min.z + global_max.z) * 0.5 );

    if (m_renderer) {
        m_renderer->init();
        
        // Unfreeze and force update to apply correct transform to the new renderer node
        setBrushFrozen(false);
        if (m_edit_mode && m_edit_tool) {
            m_edit_tool->update();
        }

        applyShaderToRendererFromSettings();
    }

    controller->signal_system_reset();

    if (m_renderer && m_renderer_paused_for_reset) {
        m_renderer->resumeRendering();
    }

    if (m_settings.show_notify) {
        if (!m_did_initial_build)
            std::cout << "[Lamure] Built " << static_cast<int>(N) << " models" << std::endl;
        else
            std::cout << "[Lamure] Reset: rebuilt " << static_cast<int>(N) << " models" << std::endl;
    }

    m_reset_in_progress = false;
    g_is_resetting = false;
    m_did_initial_build = true;
}


void Lamure::loadSettingsFromCovise() {
    auto& s = m_settings;
    const char* root = "COVER.Plugin.LamurePointCloud";

    // ---- Budgets / LODs ----
    s.frame_div = getNum<int>("value", (std::string(root) + ".frame_div").c_str(), s.frame_div);
    s.vram      = getNum<int>("value", (std::string(root) + ".vram").c_str(),      s.vram);
    s.ram       = getNum<int>("value", (std::string(root) + ".ram").c_str(),       s.ram);
    s.upload    = getNum<int>("value", (std::string(root) + ".upload").c_str(),    s.upload);
    s.pause_frames = static_cast<uint32_t>(std::max(0, getNum<int>("value", (std::string(root) + ".pause_frames").c_str(), s.pause_frames)));
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

    // ---- Attachment behavior ----
    s.prefer_parent           = getOn((std::string(root) + ".prefer_parent").c_str(),          s.prefer_parent);

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

    // ---- Exclusivity ----
    // Primitive: exactly one; priority Surfel > Splatting > Point.
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

    // ---- always honor the color mode so the UI reflects the actual state ----
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

    // ---- Surfel anisotropic scaling ----
    {
        const std::string key = std::string(root) + ".anisotropic_surfel_scaling";
        std::string mode = getStr(key.c_str(), "auto");
        for (auto &c : mode) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
        if (mode == "off" || mode == "0") s.anisotropic_surfel_scaling = 0;
        else if (mode == "on" || mode == "2" || mode == "true") s.anisotropic_surfel_scaling = 2;
        else s.anisotropic_surfel_scaling = 1; // auto (default)
        // Optional threshold for auto mode; defaults to 0.05 if unset
        s.anisotropic_auto_threshold = getNum<float>("value", (std::string(root) + ".anisotropic_auto_threshold").c_str(), s.anisotropic_auto_threshold);
    }

    // ---- Dateien / Pfade ----
    s.pvs              = getStr((std::string(root) + ".pvs").c_str(),              s.pvs);
    s.background_image = getStr((std::string(root) + ".background_image").c_str(), s.background_image);

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
    m_model_info.config_transforms.clear();
    m_model_info.config_transforms.resize(s.models.size(), scm::math::mat4d::identity());
    m_model_info.model_visible.clear();
    m_model_info.model_visible.resize(s.models.size(), true);
    m_vrmlTransforms.clear();
    m_model_source_keys.clear();

    // ---- Hintergrundfarbe ----
    s.background_color = scm::math::vec3(
        covise::coCoviseConfig::getFloat("r", "COVER.Background", 0.0f),
        covise::coCoviseConfig::getFloat("g", "COVER.Background", 0.0f),
        covise::coCoviseConfig::getFloat("b", "COVER.Background", 0.0f)
    );
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
    if (!m_measurement) return; // only accumulate when measurement is active

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
    // Ensure shader type is determined consistently even if s.shader_type is overridden later:
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

    // 1) Highest stem number from ANY stem_####*.* file (includes Frames/Summary)
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

    // 2) Avoid collisions by checking the stem_#### prefix against every file
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

    // Target directory (non-fatal on failure)
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
    add_str("primitive",   determinePrimitive); // <<<< final primitive derivation
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

    // Visibility toggles
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

    // Per-model data
    {
        std::map<uint32_t, scm::math::mat4d> cfg;
        const auto &modelInfo = m_model_info;
        for (uint32_t i = 0; i < modelInfo.config_transforms.size(); ++i) {
            cfg[i] = modelInfo.config_transforms[i];
        }
        add_raw("transforms", mapTransforms(cfg));
    }

    // Measurement paths
    add_raw("measurement_segments", segs(s.measurement_segments));

    // Measurement modes/flags
    add_bool("measure_off",    s.measure_off);
    add_bool("measure_light",  s.measure_light);
    add_bool("measure_full",   s.measure_full);
    add_i   ("measure_sample", s.measure_sample);

    // Modelle (Meta)
    add_i("num_models", s.num_models);


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





std::vector<std::string> Lamure::resolveAndNormalizeModels()
{
    const char* root = "COVER.Plugin.LamurePointCloud";
    auto normalize = [](const std::string &p)->std::string{
        std::string s = std::filesystem::absolute(std::filesystem::path(p)).string();
#ifdef _WIN32
        std::replace(s.begin(), s.end(), '\\', '/');
#endif
        return s;
    };
    auto make_key = [](std::string p){
#ifdef _WIN32
        for (auto &c : p) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
#endif
        return p;
    };

    std::vector<std::string> out;

    // 1) CLI/Bootstrap files have priority
    if (!m_bootstrap_files.empty()) {
        std::unordered_set<std::string> seen;
        out.reserve(m_bootstrap_files.size());
        for (const auto &mp : m_bootstrap_files) {
            std::string abs = normalize(mp);
            if (abs.empty()) continue;
            const std::string k = make_key(abs);
            if (seen.insert(k).second)
                out.push_back(std::move(abs));
        }
        m_models_from_config = false;
        return out;
    }

    // 2) Fall back to config if plugin is enabled
    if (!getOn(root, false)) {
        m_models_from_config = false;
        return out;
    }

    const std::string models_list = getStr((std::string(root) + ".models").c_str(), "");
    const std::string data_dir    = getStr((std::string(root) + ".data_dir").c_str(), "");

    std::vector<std::string> collected;
    for (const auto &m : LamureUtil::splitSemicolons(models_list)) {
        if (!m.empty()) collected.push_back(m);
    }
    if (!data_dir.empty() && std::filesystem::is_directory(data_dir)) {
        for (auto &e : std::filesystem::recursive_directory_iterator(data_dir)) {
            if (!e.is_regular_file()) continue;
            std::string ext = e.path().extension().string();
            for (auto &ch : ext) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
            if (ext == ".bvh") collected.push_back(e.path().string());
        }
    }

    std::unordered_set<std::string> seen;
    for (const auto &mp : collected) {
        std::string abs = normalize(mp);
        if (abs.empty()) continue;
        if (!std::filesystem::exists(abs)) {
            std::cerr << "[Lamure] Warning: Model path from config not found, skipping: " << abs << std::endl;
            continue;
        }
        const std::string k = make_key(abs);
        if (seen.insert(k).second)
            out.push_back(std::move(abs));
    }

    // Apply optional initial_selection after building the list
    const std::string sel = getStr((std::string(root) + ".initial_selection").c_str(), "");
    if (!sel.empty() && !out.empty()) {
        auto parse = [](const std::string &str, size_t N){
            std::vector<uint32_t> idx; if (str.empty()) return idx;
            std::istringstream ss(str); std::string part;
            auto trim=[&](std::string t){ auto b=t.find_first_not_of(" \t"); auto e=t.find_last_not_of(" \t");
                return (b==std::string::npos)?std::string():t.substr(b,e-b+1); };
            while (std::getline(ss, part, ',')) {
                part = trim(part); auto dash = part.find('-');
                if (dash != std::string::npos) {
                    int a = std::stoi(part.substr(0,dash)), b = std::stoi(part.substr(dash+1)); if (a>b) std::swap(a,b);
                    for (int i=a;i<=b;++i) if (i>=0 && (size_t)i<N) idx.push_back((uint32_t)i);
                } else {
                    int v = std::stoi(part); if (v>=0 && (size_t)v<N) idx.push_back((uint32_t)v);
                }
            }
            std::sort(idx.begin(), idx.end()); idx.erase(std::unique(idx.begin(), idx.end()), idx.end()); return idx;
        };
        const auto indices = parse(sel, out.size());
        if (!indices.empty()) {
            std::vector<std::string> filtered;
            filtered.reserve(indices.size());
            for (auto i : indices) if (i < out.size()) filtered.push_back(out[i]);
            out.swap(filtered);
        }
    }

    m_models_from_config = !out.empty();
    return out;
}

void Lamure::updateModelDependentSettings()
{
    auto &s = m_settings;
    // JSON path preference
    s.json = getStr("COVER.Plugin.LamurePointCloud.json", s.json);
    if (!s.json.empty() && !std::filesystem::exists(s.json)) {
        std::cerr << "[Lamure] config json not found: " << s.json << " -> ignore\n";
        s.json.clear();
    }

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

    // Reset transforms to identity for each model
    m_model_info.config_transforms.clear();
    m_model_info.config_transforms.resize(s.models.size(), scm::math::mat4d::identity());
    m_model_info.model_visible.clear();
    m_model_info.model_visible.resize(s.models.size(), true);
    m_vrmlTransforms.clear();
    m_model_source_keys.clear();
}

void Lamure::ensureFileMenuEntry(const std::string& path)
{
    if (path.empty()) return;
    if (m_registeredFiles.count(path)) return;

    if (auto *fm = opencover::coVRFileManager::instance()) {
        fm->loadFile(path.c_str(), nullptr, m_lamure_grp.get(), "");
        m_registeredFiles.insert(path);
    }
}
