#ifdef WIN32
#include <winbase.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#endif
//local
#include "LamurePointCloud.h"
#include "gl_state.h"
#include "osg_util.h"
#include "LamurePointCloudInteractor.h"

// std
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <vector>
#include <algorithm>
#include <list>
#include <iosfwd>
#include <sstream>
#include <inttypes.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <math.h>
#include <mutex>
#include <filesystem>

//boost
#include <boost/assign/list_of.hpp>
#include <boost/regex.hpp>
#include <boost/thread.hpp>

//schism
#include <scm/time.h>
//#include <scm/core.h>
#include <scm/core/math.h>
#include <scm/core/io/tools.h>
#include <scm/core/pointer_types.h>
//#include <scm/core/platform/platform.h>
//#include <scm/core/utilities/platform_warning_disable.h>
#include <scm/gl_util/primitives/primitives_fwd.h>
#include <scm/gl_util/primitives.h>
#include <scm/gl_core/buffer_objects/scoped_buffer_map.h>

//lamure
#include <lamure/pvs/pvs_database.h>
#include <lamure/prov/prov_aux.h>
#include <lamure/vt/pre/AtlasFile.h>
#include <lamure/prov/octree.h>
#include <lamure/vt/VTConfig.h>
#include <lamure/vt/ren/CutDatabase.h>
#include <lamure/vt/ren/CutUpdate.h>
#include <lamure/utils.h>
#include "lamure/ren/data_provenance.h"
#include "lamure/ren/controller.h"
#include <lamure/config.h>
#include <lamure/ren/cut.h>

#include <config/coConfigConstants.h>
#include <config/coConfigLog.h>
#include <config/coConfig.h>
#include <config/coConfigString.h>
#include <config/coConfigEntryString.h>

#include <cover/ui/SelectionList.h>
#include <cover/coVRStatsDisplay.h>
#include <cover/VRSceneGraph.h>
#include "cover/OpenCOVER.h"
#include <cover/VRWindow.h>
#include <cover/VRViewer.h>
#include <cover/coHud.h>
#include <cover/coVRTui.h>
#include <cover/ui/Menu.h>
#include "cover/coVRCollaboration.h"
#include "cover/coIntersection.h"

#include <osgViewer/GraphicsWindow>
#include <osgViewer/Renderer>
#include <osgGA/EventQueue>
#include <osg/PolygonMode>
#include <osg/StateSet>

#include <util/coExport.h>
#include <PluginUtil/FeedbackManager.h>
#include <PluginUtil/ModuleInteraction.h>
#include <OpenVRUI/coButtonInteraction.h>
#include <config/CoviseConfig.h>

#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>


int CheckGLError(char* file, int line)
{
	GLenum glErr;
	int    retCode = 0;
	glErr = glGetError();
	while (glErr != GL_NO_ERROR) {
		const GLubyte* sError = gluErrorString(glErr);
		if (sError) { cerr << "GL Error #" << glErr << "(" << gluErrorString(glErr) << ") " << " in File " << file << " at line: " << line << endl; }
		else { cerr << "GL Error #" << glErr << " (no message available)" << " in File " << file << " at line: " << line << endl; }
		retCode = 1;
		glErr = glGetError();
	}
	return retCode;
}
#define CHECK_GL_ERROR() CheckGLError(__FILE__, __LINE__)


#ifdef __cplusplus
extern "C" {
#endif
	__declspec(dllexport) DWORD NvOptimusEnablement = 1;
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
#ifdef __cplusplus
}
#endif

COVERPLUGIN(LamurePointCloudPlugin)
LamurePointCloudPlugin* LamurePointCloudPlugin::plugin = nullptr;

static FileHandler handler =
{ NULL,
  LamurePointCloudPlugin::loadLMR,
  LamurePointCloudPlugin::unloadLMR,
  "lmr"
};

LamurePointCloudPlugin::LamurePointCloudPlugin() : ui::Owner("LamurePointCloud", cover->ui)
{
	coVRFileManager::instance()->registerFileHandler(&handler);
	plugin = this;
}

LamurePointCloudPlugin* LamurePointCloudPlugin::instance()
{
	return plugin;
}

LamurePointCloudPlugin::~LamurePointCloudPlugin()
{
	fprintf(stderr, "LamurePlugin::~LamurePlugin\n");
	coVRFileManager::instance()->unregisterFileHandler(&handler);
	cover->getObjectsRoot()->removeChild(LamureGroup);
}

FT_Library ft_;
FT_Face face_;
static GLuint       g_FontTexture = 0;
static coVRConfig* coco = coVRConfig::instance();
static const osg::GraphicsContext::Traits* traits = coVRConfig::instance()->windows[0].context->getTraits();
static lamure::context_t lmr_ctx;
boost::mutex m;
std::mutex gl_state_mutex;
uint32_t render_width_;
uint32_t render_height_;
lamure::ren::Data_Provenance data_provenance_;
bool prov_valid;
float height_divided_by_top_minus_bottom_ = 0.0f;
uint32_t num_models_ = 0;

scm::gl::render_device_ptr      device_;
scm::gl::render_context_ptr     context_;
scm::gl::quad_geometry_ptr      screen_quad_;
scm::gl::text_renderer_ptr      text_renderer_;
scm::gl::text_ptr               renderable_text_;

lmr_camera* lamure_camera_;
lamure::ren::camera* scm_camera_;
osg::ref_ptr<osg::Camera>   osg_camera_;
osg::ref_ptr<osg::Camera>   rtt_camera_;

scm::gl::program_ptr vis_point_shader_;
scm::gl::program_ptr vis_point_prov_shader_;
scm::gl::program_ptr vis_surfel_shader_;
scm::gl::program_ptr vis_surfel_prov_shader_;

scm::gl::program_ptr vis_line_bb_shader_; 
scm::gl::program_ptr vis_text_shader_;
scm::gl::program_ptr vis_box_shader_;
scm::gl::program_ptr vis_xyz_shader_;
scm::gl::program_ptr vis_xyz_pass1_shader_;
scm::gl::program_ptr vis_xyz_pass2_shader_;
scm::gl::program_ptr vis_xyz_pass3_shader_;
scm::gl::program_ptr vis_xyz_lighting_shader_;
scm::gl::program_ptr vis_xyz_pass2_lighting_shader_;
scm::gl::program_ptr vis_xyz_pass3_lighting_shader_;
scm::gl::program_ptr vis_xyz_qz_shader_;
scm::gl::program_ptr vis_xyz_qz_pass1_shader_;
scm::gl::program_ptr vis_xyz_qz_pass2_shader_;
scm::gl::program_ptr vis_quad_shader_;
scm::gl::program_ptr vis_plane_shader_;
scm::gl::program_ptr vis_line_shader_;
scm::gl::program_ptr vis_triangle_shader_;
scm::gl::program_ptr vis_vt_shader_;

scm::gl::frame_buffer_ptr fbo_;
scm::gl::texture_2d_ptr fbo_color_buffer_;
scm::gl::texture_2d_ptr fbo_depth_buffer_;
scm::gl::frame_buffer_ptr pass1_fbo_;
scm::gl::frame_buffer_ptr pass2_fbo_;
scm::gl::frame_buffer_ptr pass3_fbo_;
scm::gl::texture_2d_ptr pass1_depth_buffer_;
scm::gl::texture_2d_ptr pass2_color_buffer_;
scm::gl::texture_2d_ptr pass2_normal_buffer_;
scm::gl::texture_2d_ptr pass2_view_space_pos_buffer_;
scm::gl::texture_2d_ptr pass2_depth_buffer_;
scm::gl::depth_stencil_state_ptr depth_state_disable_;
scm::gl::depth_stencil_state_ptr depth_state_less_;
scm::gl::depth_stencil_state_ptr depth_state_without_writing_;
scm::gl::rasterizer_state_ptr no_backface_culling_rasterizer_state_;
scm::gl::blend_state_ptr color_blending_state_;
scm::gl::blend_state_ptr color_no_blending_state_;
scm::gl::sampler_state_ptr filter_linear_;
scm::gl::sampler_state_ptr filter_nearest_;
scm::gl::sampler_state_ptr vt_filter_linear_;
scm::gl::sampler_state_ptr vt_filter_nearest_;
scm::gl::texture_2d_ptr bg_texture_;

scm::time::accum_timer<scm::time::high_res_timer> frame_time_;

std::string vis_point_vs_source;
std::string vis_point_fs_source;
std::string vis_point_prov_vs_source;
std::string vis_point_prov_fs_source;
std::string vis_surfel_vs_source;
std::string vis_surfel_gs_source;
std::string vis_surfel_fs_source;
std::string vis_surfel_prov_vs_source;
std::string vis_surfel_prov_fs_source;

std::string vis_line_bb_vs_source;
std::string vis_line_bb_fs_source;
std::string vis_quad_vs_source;
std::string vis_quad_fs_source;
std::string vis_line_vs_source;
std::string vis_line_fs_source;
std::string vis_triangle_vs_source;
std::string vis_triangle_fs_source;
std::string vis_plane_vs_source;
std::string vis_plane_fs_source;
std::string vis_text_vs_source;
std::string vis_text_fs_source;
std::string vis_vt_vs_source;
std::string vis_vt_fs_source;
std::string vis_xyz_vs_source;
std::string vis_xyz_gs_source;
std::string vis_xyz_fs_source;
std::string vis_xyz_pass1_vs_source;
std::string vis_xyz_pass1_gs_source;
std::string vis_xyz_pass1_fs_source;
std::string vis_xyz_pass2_vs_source;
std::string vis_xyz_pass2_gs_source;
std::string vis_xyz_pass2_fs_source;
std::string vis_xyz_pass3_vs_source;
std::string vis_xyz_pass3_fs_source;
std::string vis_xyz_qz_vs_source;
std::string vis_xyz_qz_pass1_vs_source;
std::string vis_xyz_qz_pass2_vs_source;
std::string vis_box_vs_source;
std::string vis_box_gs_source;
std::string vis_box_fs_source;

std::string vis_xyz_vs_lighting_source;
std::string vis_xyz_gs_lighting_source;
std::string vis_xyz_fs_lighting_source;
std::string vis_xyz_pass2_vs_lighting_source;
std::string vis_xyz_pass2_gs_lighting_source;
std::string vis_xyz_pass2_fs_lighting_source;
std::string vis_xyz_pass3_vs_lighting_source;
std::string vis_xyz_pass3_fs_lighting_source;

std::string shader_root_path = LAMURE_SHADERS_DIR;
std::string font_root_path = LAMURE_FONTS_DIR;

static osg::Vec3f vecConv3F(scm::math::vec3f& v);
static osg::Vec3d vecConv3D(scm::math::vec3d& v);
static osg::Vec4f vecConv4F(scm::math::vec4f& v);
static osg::Vec4d vecConv4D(scm::math::vec4d& v);

static scm::math::vec3f vecConv3F(osg::Vec3f& v);
static scm::math::vec3d vecConv3D(osg::Vec3d& v);
static scm::math::vec4f vecConv4F(osg::Vec4f& v);
static scm::math::vec4d vecConv4D(osg::Vec4d& v);

static osg::Matrixf matConv4F(scm::math::mat4f& m);
static osg::Matrixd matConv4D(scm::math::mat4d& m);
static scm::math::mat4f matConv4F(osg::Matrixd& m);
static scm::math::mat4d matConv4D(osg::Matrixd& m);

osg::Vec3f vecConv3F(scm::math::vec3f& v) {
	osg::Vec3f vec_osg = osg::Vec3f(v[0], v[1], v[2]);
	return vec_osg;
}
osg::Vec3d vecConv3D(scm::math::vec3d& v) {
	osg::Vec3d vec_osg = osg::Vec3d(v[0], v[1], v[2]);
	return vec_osg;
}
osg::Vec4f vecConv4F(scm::math::vec4f& v) {
	osg::Vec4f vec_osg = osg::Vec4f(v[0], v[1], v[2], v[3]);
	return vec_osg;
}
osg::Vec4d vecConv4D(scm::math::vec4d& v) {
	osg::Vec4d vec_osg = osg::Vec4d(v[0], v[1], v[2], v[3]);
	return vec_osg;
}
scm::math::vec3f vecConv3F(osg::Vec3f& v) {
	scm::math::vec3f vec_scm = scm::math::vec3f(v[0], v[1], v[2]);
	return vec_scm;
}
scm::math::vec3d vecConv3D(osg::Vec3d& v) {
	scm::math::vec3d vec_scm = scm::math::vec3d(v[0], v[1], v[2]);
	return vec_scm;
}
scm::math::vec4f vecConv4F(osg::Vec4f& v) {
	scm::math::vec4f vec_scm = scm::math::vec4f(v[0], v[1], v[2], v[3]);
	return vec_scm;
}
scm::math::vec4d vecConv4D(osg::Vec4d& v) {
	scm::math::vec4d vec_scm = scm::math::vec4d(v[0], v[1], v[2], v[3]);
	return vec_scm;
}
osg::Matrixf matConv4F(scm::math::mat4f& m) {
	osg::Matrix mat_osg = osg::Matrixf(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
	return mat_osg;
}
osg::Matrixd matConv4D(scm::math::mat4d& m) {
	osg::Matrixd mat_osg = osg::Matrixd(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11], m[12], m[13], m[14], m[15]);
	return mat_osg;
}
scm::math::mat4f matConv4F(osg::Matrixd& m) {
	scm::math::mat4f mat_scm = scm::math::mat4f(m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
	return mat_scm;
}
scm::math::mat4d matConv4D(osg::Matrixd& m) {
	scm::math::mat4d mat_scm = scm::math::mat4d(m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
	return mat_scm;
}
scm::math::mat4f matConv4F(const osg::Matrixd& m) {
	scm::math::mat4f mat_scm = scm::math::mat4f(m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
	return mat_scm;
}
scm::math::mat4d matConv4D(const osg::Matrixd& m) {
	scm::math::mat4d mat_scm = scm::math::mat4d(m(0, 0), m(0, 1), m(0, 2), m(0, 3), m(1, 0), m(1, 1), m(1, 2), m(1, 3), m(2, 0), m(2, 1), m(2, 2), m(2, 3), m(3, 0), m(3, 1), m(3, 2), m(3, 3));
	return mat_scm;
}

struct settings {
	int32_t frame_div_{ 1 };
	int32_t vram_{ 1024 };
	int32_t ram_{ 4096 };
	int32_t upload_{ 32 };
	bool provenance_{ 1 };
	bool create_aux_resources_{ 1 };
	bool gamma_correction_{ 0 };
	bool surfel_shader_{ 1 };
	bool face_eye_{ 0 };
	int32_t gui_{ 1 };
	int32_t travel_{ 2 };
	float travel_speed_{ 20.5f };
	int32_t max_brush_size_{ 4096 };
	bool lod_update_{ 1 };
	float lod_error_{ 1.0f };
	bool use_pvs_{ 0 };
	bool pvs_culling_{ 0 };
	float point_size_factor_{ 1.0f };
	float surfel_size_factor_{ 1.0f };
	float aux_point_size_{ 1.0f };
	float aux_point_distance_{ 0.5f };
	float aux_point_scale_{ 1.0f };
	float aux_focal_length_{ 1.0f };
	int32_t vis_{ 0 };
	int32_t show_normals_{ 0 };
	bool show_accuracy_{ 0 };
	bool show_radius_deviation_{ 0 };
	bool show_output_sensitivity_{ 0 };
	bool show_sparse_{ 0 };
	bool show_views_{ 0 };
	bool show_photos_{ 0 };
	bool show_octrees_{ 0 };
	bool show_bvhs_{ 0 };
	bool show_pvs_{ 0 };
	int32_t channel_{ 0 };
	bool enable_lighting_{ 0 };
	bool use_material_color_{ 1 };
	scm::math::vec3f material_diffuse_{ 0.6f, 0.6f, 0.6f };
	scm::math::vec4f material_specular_{ 0.4f, 0.4f, 0.4f, 1000.0f };
	scm::math::vec3f ambient_light_color_{ 0.1f, 0.1f, 0.1f };
	scm::math::vec4f point_light_color_{ 1.0f, 1.0f, 1.0f, 1.2f };
	bool heatmap_{ 0 };
	float heatmap_min_{ 0.0f };
	float heatmap_max_{ 0.05f };
	scm::math::vec3f background_color_{ LAMURE_DEFAULT_COLOR_R, LAMURE_DEFAULT_COLOR_G, LAMURE_DEFAULT_COLOR_B };
	scm::math::vec3f heatmap_color_min_{ 68.0f / 255.0f, 0.0f, 84.0f / 255.0f };
	scm::math::vec3f heatmap_color_max_{ 251.f / 255.f, 231.f / 255.f, 35.f / 255.f };
	std::string atlas_file_{ "" };
	std::string json_{ "" };
	std::string pvs_{ "" };
	std::string background_image_{ "" };
	std::vector<std::string> models_;
	std::vector<uint32_t> selection_;
	std::map<uint32_t, scm::math::mat4d> transforms_;
	std::map<uint32_t, std::shared_ptr<lamure::prov::octree>> octrees_;
	std::map<uint32_t, std::vector<lamure::prov::aux::view>> views_;
	std::map<uint32_t, std::string> aux_;
	float max_radius_{ std::min(std::numeric_limits<float>::max(), 0.1f) };
	float scale_radius_{ 1.5f };
	std::vector<float> bvh_color_{ 1.0f, 1.0f, 0.0f, 1.0f };
	std::vector<float> frustum_color_{ 0.0f, 0.0f, 0.0f, 1.0f };
};
settings settings_;

int LamurePointCloudPlugin::unloadLMR(const char* filename, const char* covise_key)
{
	return 1;
}

void LamurePointCloudPlugin::load_settings(std::string const& filename) {
	using namespace std::filesystem;
	std::cout << "load_settings()" << std::endl;
	auto parseIndices = [](const std::string &s, size_t max_index) {
		std::vector<uint32_t> out;
		if (s.empty()) {
			for (uint32_t i = 0; i < max_index; ++i) out.push_back(i);
			return out;
		}
		std::istringstream ss(s);
		std::string part;
		while (std::getline(ss, part, ',')) {
			size_t dash = part.find('-');
			if (dash != std::string::npos) {
				int32_t a = std::stoi(part.substr(0, dash));
				int32_t b = std::stoi(part.substr(dash + 1));
				for (int32_t i = a; i <= b; ++i)
					if (i >= 0 && static_cast<size_t>(i) < max_index) out.push_back(i);
			} else {
				int32_t val = std::stoi(part);
				if (val >= 0 && static_cast<size_t>(val) < max_index) out.push_back(val);
			}
		}
		std::sort(out.begin(), out.end());
		out.erase(std::unique(out.begin(), out.end()), out.end());
		return out;
	};

	auto strip_ws = [](std::string s) {
		auto not_ws = [](char c) { return !std::isspace(c); };
		s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_ws));
		s.erase(std::find_if(s.rbegin(), s.rend(), not_ws).base(), s.end());
		return s;
	};

	std::ifstream file(filename);
	if (!file.is_open()) {
		std::cerr << "could not open lmr file: " << filename << std::endl;
		std::exit(EXIT_FAILURE);
	}
	std::vector<std::string> lines;
	for (std::string line; std::getline(file, line); ) lines.push_back(line);
	file.close();

	settings_.models_.clear();
	settings_.transforms_.clear();
	settings_.json_.clear();

	std::set<std::string> unique_models;
	std::string data_dir;
	for (const auto& raw : lines) {
		std::string l = strip_ws(raw);
		if (l.empty() || l[0] == '#') continue;
		size_t colon = l.find(':');
		if (colon <= 1) {
			if (exists(l)) {
				if (is_directory(l)) {
					data_dir = l;
				} else {
					unique_models.insert(absolute(l).string());
				}
			}
		}
	}

	if (!data_dir.empty()) {
		for (const auto& e : recursive_directory_iterator(data_dir)) {
			if (e.is_regular_file() && e.path().extension() == ".bvh") {
				unique_models.insert(absolute(e.path()).string());
			}
		}
	}

	for (const auto& path : unique_models) { settings_.models_.push_back(path); }

	prov_valid = true;
	std::string first_json;
	for (const auto& model_path : settings_.models_) {
		std::filesystem::path p(model_path);
		std::filesystem::path prov_file = p;
		prov_file.replace_extension(".prov");
		std::filesystem::path json_file = p;
		json_file.replace_extension(".json");
		if (!exists(prov_file) || !exists(json_file)) {
			prov_valid = false;
			break;
		} else {
			if (first_json.empty()) {
				first_json = json_file.string();
			}
		}
	}

	lamure::model_t model_id = 0;
	for (const auto& model_path : settings_.models_) {
		if (settings_.transforms_.find(model_id) == settings_.transforms_.end()) {
			settings_.transforms_[model_id] = scm::math::mat4d::identity();
		}
		++model_id;
	}

	for (const auto& raw : lines) {
		auto l = strip_ws(raw);
		if (l.empty() || l[0] == '#') continue;
		auto colon = l.find(':');
		if (colon <= 1 || (colon == std::string::npos)) continue;
		auto key = strip_ws(l.substr(0, colon));
		auto value = strip_ws(l.substr(colon + 1));

		if (!key.empty() && key[0] == '@') {
			auto ws = l.find_first_of(' ');
			uint32_t addr = std::atoi(strip_ws(l.substr(1, ws - 1)).c_str());
			key = strip_ws(l.substr(ws + 1, colon - (ws + 1)));
			if (key == "tf") settings_.transforms_[addr] = load_matrix(value);
			else {
				std::cerr << "unrecognized @-key: " << key << std::endl;
				std::exit(EXIT_FAILURE);
			}
		}
		else if (key == "surfel_shader")        settings_.surfel_shader_ = std::atoi(value.c_str());
		else if (key == "frame_div")            settings_.frame_div_ = std::max(std::atoi(value.c_str()), 1);
		else if (key == "vram")                 settings_.vram_ = std::max(std::atoi(value.c_str()), 8);
		else if (key == "ram")                  settings_.ram_ = std::max(std::atoi(value.c_str()), 8);
		else if (key == "upload")               settings_.upload_ = std::max(std::atoi(value.c_str()), 8);
		else if (key == "face_eye")             settings_.face_eye_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "gamma_correction")     settings_.gamma_correction_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "pvs_culling")          settings_.pvs_culling_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "use_pvs")              settings_.use_pvs_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "point_size_factor")    settings_.point_size_factor_ =  std::atoi(value.c_str());
		else if (key == "surfel_size_factor")   settings_.surfel_size_factor_ =  std::atoi(value.c_str());
		else if (key == "aux_point_size")       settings_.aux_point_size_ = std::clamp(std::atof(value.c_str()), 0.00001, 1.0);
		else if (key == "aux_point_distance")   settings_.aux_point_distance_ = std::clamp(std::atof(value.c_str()), 0.00001, 1.0);
		else if (key == "aux_focal_length")     settings_.aux_focal_length_ = std::clamp(std::atof(value.c_str()), 0.001, 10.0);
		else if (key == "max_brush_size")       settings_.max_brush_size_ = std::clamp(std::atoi(value.c_str()), 64, 1024 * 1024);
		else if (key == "lod_error")            settings_.lod_error_ = std::clamp(std::atof(value.c_str()), 1.0, 10.0);
		else if (key == "provenance")			settings_.provenance_ = (std::max(std::atoi(value.c_str()), 0) != 0) && prov_valid;
		else if (key == "create_aux_resources") settings_.create_aux_resources_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_normals")         settings_.show_normals_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_accuracy")        settings_.show_accuracy_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_radius_deviation")settings_.show_radius_deviation_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_output_sensitivity") settings_.show_output_sensitivity_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_sparse")          settings_.show_sparse_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_views")           settings_.show_views_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_photos")          settings_.show_photos_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_octrees")         settings_.show_octrees_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_bvhs")            settings_.show_bvhs_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "show_pvs")             settings_.show_pvs_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "channel")              settings_.channel_ = std::max(std::atoi(value.c_str()), 0);
		else if (key == "enable_lighting")      settings_.enable_lighting_ = std::clamp(std::atoi(value.c_str()), 0, 1) != 0;
		else if (key == "use_material_color")   settings_.use_material_color_ = std::clamp(std::atoi(value.c_str()), 0, 1) != 0;
		else if (key == "material_diffuse_r")   settings_.material_diffuse_.x = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "material_diffuse_g")   settings_.material_diffuse_.y = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "material_diffuse_b")   settings_.material_diffuse_.z = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "material_specular_r")  settings_.material_specular_.x = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "material_specular_g")  settings_.material_specular_.y = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "material_specular_b")  settings_.material_specular_.z = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "material_specular_exponent") settings_.material_specular_.w = std::clamp(std::atof(value.c_str()), 0.0, 10000.0);
		else if (key == "ambient_light_color_r")settings_.ambient_light_color_.r = std::clamp<float>(std::atof(value.c_str()), 0.0f, 1.0f);
		else if (key == "ambient_light_color_g")settings_.ambient_light_color_.g = std::clamp<float>(std::atof(value.c_str()), 0.0f, 1.0f);
		else if (key == "ambient_light_color_b")settings_.ambient_light_color_.b = std::clamp<float>(std::atof(value.c_str()), 0.0f, 1.0f);
		else if (key == "point_light_color_r")  settings_.point_light_color_.r = std::clamp<float>(std::atof(value.c_str()), 0.0f, 1.0f);
		else if (key == "point_light_color_g")  settings_.point_light_color_.g = std::clamp<float>(std::atof(value.c_str()), 0.0f, 1.0f);
		else if (key == "point_light_color_b")  settings_.point_light_color_.b = std::clamp<float>(std::atof(value.c_str()), 0.0f, 1.0f);
		else if (key == "point_light_intensity")settings_.point_light_color_.w = std::clamp<float>(std::atof(value.c_str()), 0.0f, 10000.0);
		else if (key == "background_color_r")   settings_.background_color_.x = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "background_color_g")   settings_.background_color_.y = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "background_color_b")   settings_.background_color_.z = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "heatmap")              settings_.heatmap_ = std::max(std::atoi(value.c_str()), 0) != 0;
		else if (key == "heatmap_min")          settings_.heatmap_min_ = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "heatmap_max")          settings_.heatmap_max_ = std::max<float>(std::atof(value.c_str()), 0.0f);
		else if (key == "heatmap_min_r")        settings_.heatmap_color_min_.x = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "heatmap_min_g")        settings_.heatmap_color_min_.y = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "heatmap_min_b")        settings_.heatmap_color_min_.z = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "heatmap_max_r")        settings_.heatmap_color_max_.x = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "heatmap_max_g")        settings_.heatmap_color_max_.y = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "heatmap_max_b")        settings_.heatmap_color_max_.z = std::min(std::max(std::atoi(value.c_str()), 0), 255) / 255.0f;
		else if (key == "json")					settings_.json_ = !value.empty() ? value : (!first_json.empty() ? first_json : settings_.json_);
		else if (key == "selection")            settings_.selection_ = parseIndices(value, settings_.models_.size());
		else if (key == "pvs")                  settings_.pvs_ = value;
		else if (key == "background_image")     settings_.background_image_ = value;
		else if (key == "max_radius")           settings_.max_radius_ = std::max(std::atof(value.c_str()), 0.1);
		else if (key == "scale_radius")         settings_.scale_radius_ = std::max(std::atof(value.c_str()), 0.1);
		else { std::cerr << "unrecognized key: " << key << std::endl; std::exit(EXIT_FAILURE); }
	}
	for (auto const& m : settings_.models_) { std::cout << "Loaded model: " << m << std::endl; }
	for (auto const& m : settings_.selection_) { std::cout << "Selected models: " << m << std::endl; }
}

struct Character {
	int SizeX;
	int SizeY;
	int BearingX;
	int BearingY;
	GLuint Advance;
	float TexCoordX;
	float TexCoordY;
	float TexCoordWidth;
	float TexCoordHeight;
};
std::map<char, Character> characters_;

struct resource {
	uint64_t num_primitives_{ 0 };
	scm::gl::buffer_ptr buffer_;
	scm::gl::vertex_array_ptr array_;
	std::vector<std::vector<float>> corners_;
};
resource brush_resource_;
resource pvs_resource_;

std::map<uint32_t, resource> bvh_res_;
std::map<uint32_t, resource> octree_res_;
std::map<uint32_t, resource> sparse_res_;
std::map<uint32_t, resource> image_plane_res_;

struct model_info {
	std::vector<scm::math::mat4d> model_transformations_;
	std::vector<scm::math::vec3f> root_bb_min;
	std::vector<scm::math::vec3f> root_bb_max;
	std::vector<scm::math::vec3f> root_center;
	scm::math::vec3f models_min;
	scm::math::vec3f models_max;
	scm::math::vec3d models_center;
};
model_info model_info_;

struct gui {
	bool selection_settings_{ true };
	bool view_settings_{ true };
	bool visual_settings_{ true };
	bool provenance_settings_{ true };
	scm::math::mat4f ortho_matrix_;
};
gui gui_;

struct xyz {
	scm::math::vec3f pos_;
	uint8_t r_;
	uint8_t g_;
	uint8_t b_;
	uint8_t a_;
	float rad_;
	scm::math::vec3f nml_;
};

struct vertex {
	scm::math::vec3f pos_;
	scm::math::vec2f uv_;
};

struct selection {
	int32_t selected_model_ = -1;
	int32_t selected_view_ = -1;
	std::vector<xyz> brush_;
	std::set<uint32_t> selected_views_;
	int64_t brush_end_{ 0 };
};
selection selection_;

struct trackball {
	float dist_ = 0.0;
	float size_ = 0.0;
	osg::Vec3 initial_pos_;
	osg::Vec3 pos_;
};
trackball trackball_;

struct input {
	float trackball_x_ = 0.f;
	float trackball_y_ = 0.f;
	scm::math::vec2i mouse_;
	scm::math::vec2i prev_mouse_;
	bool brush_mode_ = 0;
	bool brush_clear_ = 0;
	bool gui_lock_ = false;
	lamure::ren::camera::mouse_state mouse_state_;
	bool keys_[3] = { 0, 0, 0 };
};
input input_;

struct provenance {
	uint32_t num_views_{ 0 };
};
std::map<uint32_t, provenance> provenance_;

struct vt_info {
	uint32_t texture_id_;
	uint16_t view_id_;
	uint16_t context_id_;
	uint64_t cut_id_;
	vt::CutUpdate* cut_update_;
	std::vector<scm::gl::texture_2d_ptr> index_texture_hierarchy_;
	scm::gl::texture_2d_ptr physical_texture_;
	scm::math::vec2ui physical_texture_size_;
	scm::math::vec2ui physical_texture_tile_size_;
	size_t size_feedback_;
	int32_t* feedback_lod_cpu_buffer_;
	uint32_t* feedback_count_cpu_buffer_;
	scm::gl::buffer_ptr feedback_lod_storage_;
	scm::gl::buffer_ptr feedback_count_storage_;
	int toggle_visualization_;
	bool enable_hierarchy_;
};
vt_info vt_;

struct render_info {
	uint64_t rendered_splats_{ 0 };
	uint64_t rendered_nodes_{ 0 };
	uint64_t rendered_bounding_boxes_{ 0 };
	float fps_{0.0f};
};
render_info render_info_;

struct point_shader {
	GLuint program{0};
	GLint mvp_matrix_loc{-1};
	GLint max_radius_loc{-1};
	GLint scale_radius_loc{-1};
	GLint point_size_factor_loc{-1};
	GLint proj_scale_loc{-1};
};
point_shader point_shader_;

struct surfel_shader {
	GLuint program;
	GLint max_radius_loc;
	GLint scale_radius_loc;
	GLint surfel_size_factor_loc;
	GLint mvp_matrix_loc;
	GLint model_view_matrix_loc;  // Neue Uniform Location
	GLint proj_scale_loc;
	GLint viewport_loc; 
};
surfel_shader surfel_shader_;

struct point_prov_shader {
	GLuint program_ = 0;
	// Core-Matrizen
	GLint mvp_matrix_location = -1;
	GLint view_matrix_location = -1;
	GLint projection_matrix_location = -1;
	GLint model_matrix_location = -1;
	GLint model_view_matrix_location = -1;
	GLint inverse_mv_matrix_location = -1;
	GLint model_to_screen_matrix_location = -1;
	GLint viewport_location = -1;
	GLint height_divided_by_top_minus_bottom_location = -1;
	// Point-Cloud-Parameter
	GLint scale_radius_location = -1;
	GLint max_radius_location = -1;
	GLint window_size_location = -1;
	GLint near_plane_location = -1;
	GLint far_plane_location = -1;
	GLint point_size_factor_location = -1;
	// Anzeige-Flags
	GLint show_normals_location = -1;
	GLint show_accuracy_location = -1;
	GLint show_output_sensitivity_location = -1;
	GLint channel_location = -1;
	GLint heatmap_enabled_location = -1;
	// Heatmap-Bereiche
	GLint heatmap_min_location = -1;
	GLint heatmap_max_location = -1;
	GLint heatmap_min_color_location = -1;
	GLint heatmap_max_color_location = -1;
	// Beleuchtung
	GLint use_material_color_location = -1;
	GLint material_diffuse_location = -1;
	GLint material_specular_location = -1;
	GLint ambient_light_color_location = -1;
	GLint point_light_color_location = -1;
};
point_prov_shader point_prov_shader_;

struct line_shader {
	GLuint program_ = 0;

	GLint in_color_location = -1;
	GLint mvp_matrix_location = -1;
};
line_shader line_shader_;

struct pcl_resource {
	GLuint program_ = 0;
	GLuint vao_ = 0;

};
pcl_resource pcl_resource_;

struct box_resource {
	GLuint vbo_ = 0;
	GLuint ibo_ = 0;
	GLuint vao_ = 0;
	GLuint program_ = 0;
	std::vector<std::vector<float>> vertices_;
	std::array<unsigned short, 24> idx_ = {
		0, 1, 2, 3, 4, 5, 6, 7,
		0, 2, 1, 3, 4, 6, 5, 7,
		0, 4, 1, 5, 2, 6, 3, 7,
	};
};
box_resource box_resource_;

struct plane_resource {
	GLuint vbo_{ 0 };
	GLuint ibo_{ 0 };
	GLuint vao_{ 0 };
	GLuint program_{ 0 };
	std::array<unsigned short, 6> idx_ = {
		1,3,7,5,1,7
	};
};
plane_resource plane_resource_;

struct sphere_resource {
	GLuint vbo_{ 0 };
	GLuint vao_{ 0 };
	GLuint ibo_{ 0 };
	std::array<float, 3> points = {
		0.0f,0.0f,0.0f
	};
};
sphere_resource sphere_resource_;

struct coord_recourse {
	GLuint vao_ = 0;
	GLuint vbo_ = 0;
	GLuint ibo_ = 0;
	GLuint program_ = 0;
	std::array<float, 12> vertices_ = {
		0.f,   0.f,   0.f,
		50.f,   0.f,   0.f,  // X-Achse
		0.f,  50.f,   0.f,  // Y-Achse
		0.f,   0.f,   50.f  // Z-Achse
	};
	std::array<unsigned short, 6> idx_ = {
		0, 1,
		0, 2,
		0, 3,
	};
};
coord_recourse coord_resource_;

struct frustum_resource {
	GLuint vao_ = 0;
	GLuint vbo_ = 0;
	GLuint ibo_ = 0;
	GLuint program_ = 0;
	std::array<float, 24> vertices_;
	std::array<unsigned short, 24> idx_ = {
		0, 1, 2, 3, 4, 5, 6, 7,
		0, 2, 1, 3, 4, 6, 5, 7,
		0, 4, 1, 5, 2, 6, 3, 7,
	};
};
frustum_resource frustum_resource_;

struct text_resource {
	GLuint vao_{ 0 };
	GLuint vbo_{ 0 };
	GLuint program_{ 0 };
	GLuint atlas_texture_{ 0 };
	std::string text_;
	size_t num_vertices_{ 0 };
};
text_resource text_resource_;

void printChildNodes(osg::Node * node, int depth = 0) 
{
	if (!node) return;
	for (int i = 0; i < depth; ++i) { std::cout << "  "; }
	std::cout << "- " << node->className();
	if (node->getName().empty()) { std::cout << " (unnamed)"; }
	else { std::cout << " (" << node->getName() << ")"; }
	std::cout << std::endl;
	osg::Group* group = node->asGroup();
	if (group) {
		for (unsigned int i = 0; i < group->getNumChildren(); ++i) {
			printChildNodes(group->getChild(i), depth + 1);
		}
	}
}

void LamurePointCloudPlugin::printNodePath(osg::ref_ptr<osg::Node> pointer) 
{
	osg::NodePathList npl = pointer->getParentalNodePaths();
	int path_size = npl.size();
	std::cout << pointer->className() << " at level " << path_size << std::endl;
	if (path_size > 0) {
		for (int j = 0; j < npl[0].size(); j++) {
			std::cout << "[" << j << "] " << npl[0][j]->className() << ":  " << npl[0][j]->getName() << std::endl;
		}
		std::cout << "" << std::endl;
	}
	std::cout << "" << std::endl;
}

void APIENTRY openglCallbackFunction(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	std::cerr << "---------------------" << std::endl;
	std::cerr << "Debug message (" << id << "): " << message << std::endl;
	std::cerr << "Source: " << source << ", Type: " << type << ", Severity: " << severity << std::endl;
	std::cerr << "---------------------" << std::endl;
}

float* gl_mat_to_array(GLdouble mat[16]) {
	scm::math::mat4d gl_mat = scm::math::mat4d(mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
	float* gl_array = scm::math::mat4f(gl_mat).data_array;
	return gl_array;
}

scm::math::mat4d gl_mat(GLdouble mat[16]) {
	scm::math::mat4d gl_mat = scm::math::mat4d(mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
	return gl_mat;
}

std::vector<float> LamurePointCloudPlugin::getBoxCorners(scm::gl::boxf bbv) {
	std::vector<float> corners_ = {
		bbv.corner(0).data_array[0], bbv.corner(0).data_array[1], bbv.corner(0).data_array[2],
		bbv.corner(1).data_array[0], bbv.corner(1).data_array[1], bbv.corner(1).data_array[2],
		bbv.corner(2).data_array[0], bbv.corner(2).data_array[1], bbv.corner(2).data_array[2],
		bbv.corner(3).data_array[0], bbv.corner(3).data_array[1], bbv.corner(3).data_array[2],
		bbv.corner(4).data_array[0], bbv.corner(4).data_array[1], bbv.corner(4).data_array[2],
		bbv.corner(5).data_array[0], bbv.corner(5).data_array[1], bbv.corner(5).data_array[2],
		bbv.corner(6).data_array[0], bbv.corner(6).data_array[1], bbv.corner(6).data_array[2],
		bbv.corner(7).data_array[0], bbv.corner(7).data_array[1], bbv.corner(7).data_array[2],
	};
	return corners_;
}

std::vector<vector<float>> LamurePointCloudPlugin::getSerializedBvhMinMax(const std::vector<scm::gl::boxf> bounding_boxes) {
	std::vector<vector<float>> vecOfVec;
	for (uint64_t node_id = 0; node_id < bounding_boxes.size(); ++node_id) {
		scm::math::vec3f min_vertex = bounding_boxes[node_id].min_vertex();
		scm::math::vec3f max_vertex = bounding_boxes[node_id].max_vertex();
		vector<float> elements{
			min_vertex.x, min_vertex.y, min_vertex.z,
			max_vertex.x, max_vertex.y, max_vertex.z };
		vecOfVec.push_back(elements);
	}
	return vecOfVec;
}

double roundToDecimal(double value, int decimals) {
	if (decimals < 0) {
		return value;
	}
	double factor = std::pow(10.0, decimals);
	return std::round(value * factor) / factor;
}

int LamurePointCloudPlugin::loadLMR(const char* filename, osg::Group* parent, const char* covise_key) {
	std::printf("loadLMR()\n");
	assert(plugin);
	std::string lmr_file = std::string(filename);
	plugin->load_settings(lmr_file);
	settings_.vis_ = settings_.show_normals_ ? 1
		: settings_.show_accuracy_ ? 2
		: settings_.show_output_sensitivity_ ? 3
		: settings_.channel_ > 0 ? 3 + settings_.channel_
		: 0;
	if (settings_.provenance_ && settings_.json_ != "") {
		std::cout << "json: " << settings_.json_ << std::endl;
		data_provenance_ = lamure::ren::Data_Provenance::parse_json(settings_.json_);
		std::cout << "size of provenance: " << data_provenance_.get_size_in_bytes() << std::endl;
	}
	render_width_ = traits->width / settings_.frame_div_;
	render_height_ = traits->height / settings_.frame_div_;
	char str[200];
	sprintf(str, "COVER.WindowConfig.Window:%d", 0);
	std::printf("render_width_: %03" PRId32 "\n", render_width_);
	std::printf("render_height_: %03" PRId32 "\n", render_height_);
	lamure::ren::policy* policy = lamure::ren::policy::get_instance();
	policy->set_max_upload_budget_in_mb(settings_.upload_);
	policy->set_render_budget_in_mb(settings_.vram_);
	policy->set_out_of_core_budget_in_mb(settings_.ram_);
	policy->set_window_width(render_width_);
	policy->set_window_height(render_height_);

	lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
	lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
	lamure::ren::controller* controller = lamure::ren::controller::get_instance();
	for (const auto& input_file : settings_.models_) {
		lamure::model_t model_id = database->add_model(input_file, std::to_string(num_models_));
		model_info_.model_transformations_.push_back(settings_.transforms_[num_models_] * scm::math::mat4d(scm::math::make_translation(database->get_model(num_models_)->get_bvh()->get_translation())));
		++num_models_;
	}
	return 1;
}

using namespace scm::gl;

struct TextCullCallback : public osg::Drawable::CullCallback {
	TextCullCallback(LamurePointCloudPlugin* plugin, osgText::Text* values, render_info* render_info)
		: _plugin(plugin),
		_values(values),
		_render_info(render_info)
	{
		_lastUpdateTime = std::chrono::steady_clock::now();
		_minInterval = std::chrono::milliseconds(100);
	}

	virtual bool cull(osg::NodeVisitor* nv, osg::Drawable* drawable, osg::RenderInfo* renderInfo) const override {
		auto now = std::chrono::steady_clock::now();
		if (now - _lastUpdateTime >= _minInterval) {
			scm::math::vec3d camPos = scm_camera_->get_cam_pos();

			//osg::NodePathList npl = _plugin->LamureGroup->getParentalNodePaths();
			//const osg::NodePath& path = npl.front();
			//osg::Matrix world_matrix;
			//for (osg::Node* n : path) {
			//	osg::Matrix localMat;
			//	if (auto mt = dynamic_cast<osg::MatrixTransform*>(n)) {
			//		localMat = mt->getMatrix();
			//	}
			//	world_matrix = localMat * world_matrix;
			//}

			osg::Matrix baseMatrix = VRSceneGraph::instance()->getScaleTransform()->getMatrix();
			osg::Matrix transformMatrix = VRSceneGraph::instance()->getTransform()->getMatrix();
			baseMatrix.postMult(transformMatrix);

			scm::math::mat4d osg_base = matConv4D(baseMatrix);
			scm::math::mat4d osg_view = matConv4D(osg_camera_->getViewMatrix());
			scm::math::mat4d osg_projection = matConv4D(osg_camera_->getProjectionMatrix());

			std::stringstream osg_base_ss;
			std::stringstream osg_projection_ss;
			std::stringstream osg_mvp_ss;
			std::stringstream gl_modelview_ss;
			std::stringstream gl_projection_ss;
			std::stringstream gl_mvp_ss;
			std::stringstream scm_modelview_ss;
			std::stringstream scm_projection_ss;
			std::stringstream scm_mvp_ss;
			std::stringstream value_ss;

			osg_base_ss << osg_view * osg_base;
			osg_projection_ss << osg_projection;
			osg_mvp_ss << osg_projection * osg_view * osg_base;
			gl_modelview_ss << _plugin->gl_modelview_matrix;
			gl_projection_ss << _plugin->gl_projection_matrix;
			gl_mvp_ss << _plugin->gl_projection_matrix * _plugin->gl_modelview_matrix;
			scm_modelview_ss << scm_camera_->get_view_matrix();
			scm_projection_ss << scm_camera_->get_projection_matrix();
			scm_mvp_ss << scm_camera_->get_projection_matrix() * scm_camera_->get_view_matrix();
			value_ss << "\n"
				<< std::fixed << std::setprecision(2)
				<< 1.0f / cover->frameDuration() << "\n"
				<< _render_info->rendered_nodes_ << "\n"
				<< _render_info->rendered_splats_ << "\n"
				<< _render_info->rendered_bounding_boxes_ << "\n\n\n"
				<< camPos.x << "\n"
				<< camPos.y << "\n"
				<< camPos.z << "\n\n\n\n"
				<< osg_base_ss.str() << "\n\n\n"
				<< osg_projection_ss.str() << "\n\n\n"
				<< osg_mvp_ss.str() << "\n\n\n"
				<< gl_modelview_ss.str() << "\n\n\n"
				<< gl_projection_ss.str() << "\n\n\n"
				<< gl_mvp_ss.str() << "\n\n\n"
				<< scm_modelview_ss.str() << "\n\n\n"
				<< scm_projection_ss.str() << "\n\n\n"
				<< scm_mvp_ss.str() << "\n";
			_values->setText(value_ss.str(), osgText::String::ENCODING_UTF8);
			_lastUpdateTime = now;
		}
		return false;
	}
	LamurePointCloudPlugin* _plugin;
	osg::ref_ptr<osgText::Text> _values;
	render_info* _render_info;
	mutable std::chrono::steady_clock::time_point _lastUpdateTime;
	std::chrono::milliseconds _minInterval;
};


struct TextGeode : public osg::Geode {
	TextGeode(LamurePointCloudPlugin* plugin) :
		_plugin(plugin) {
		if (plugin->notify_button->state()) { std::cout << "[Notify] TextGeode()" << std::endl; }
		osg::Quat rotation(osg::DegreesToRadians(90.0f), osg::Vec3(1.0f, 0.0f, 0.0f));
		osg::Vec4 color(1.0f, 1.0f, 1.0f, 1.0f);
		std::string font = coVRFileManager::instance()->getFontFile(NULL);
		float characterSize = 20.0f;
		osg::Vec3 pos_label(+traits->width * 0.5f, 0.0f, traits->height * 0.7f);
		osg::Vec3 pos_value = pos_label + osg::Vec3(100.0f, 0.0f, 0.0f);
		osg::ref_ptr<osgText::Text> label = new osgText::Text();
		label->setRotation(rotation);
		label->setColor(color);
		label->setFont(font);
		label->setCharacterSize(characterSize);
		label->setPosition(pos_label);
		std::stringstream label_ss;
		label_ss << "Rendering" << "\n"
			<< "FPS:" << "\n"
			<< "Nodes:" << "\n"
			<< "Splats:" << "\n"
			<< "Boxes:" << "\n\n"
			<< "Frustum Position" << "\n"
			<< "X:" << "\n"
			<< "Y:" << "\n"
			<< "Z:" << "\n\n\n"
			<< "OSG BASE:" << "\n\n\n\n\n\n"
			<< "OSG Projection:" << "\n\n\n\n\n\n"
			<< "OSG MVP:" << "\n\n\n\n\n\n"
			<< "GL ModelView:" << "\n\n\n\n\n\n"
			<< "GL Projection:" << "\n\n\n\n\n\n"
			<< "GL MVP:" << "\n\n\n\n\n\n"
			<< "SCM ModelView:" << "\n\n\n\n\n\n"
			<< "SCM Projection:" << "\n\n\n\n\n\n"
			<< "SCM MVP:" << "\n";
		label->setText(label_ss.str(), osgText::String::ENCODING_UTF8);

		osg::ref_ptr<osgText::Text> value = new osgText::Text();
		value->setRotation(rotation);
		value->setColor(color);
		value->setFont(font);
		value->setCharacterSize(characterSize);
		value->setPosition(pos_value);
		std::stringstream value_ss;
		value_ss << "\n"
			<< "0.00:" << "\n"
			<< "0.00" << "\n"
			<< "0.00" << "\n"
			<< "0.00:" << "\n\n\n"
			<< "0.00" << "\n"
			<< "0.00" << "\n"
			<< "0.00" << "\n\n\n\n\n"
			<< "0.00" << "\n\n\n\n"
			<< "0.00" << "\n\n\n\n"
			<< "0.00" << "\n\n\n\n"
			<< "0.00" << "\n\n\n\n"
			<< "0.00" << "\n\n\n\n"
			<< "0.00" << "\n\n\n\n"
			<< "0.00" << "\n";
		value->setText(value_ss.str(), osgText::String::ENCODING_UTF8);
		this->addDrawable(label.get());
		this->addDrawable(value.get());
		value->setCullCallback(new TextCullCallback(_plugin, value.get(), &render_info_));
	}
	LamurePointCloudPlugin* _plugin;
};


struct CoordDrawCallback : public osg::Drawable::DrawCallback
{
	CoordDrawCallback(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
		: _stateset(stateset),
		_plugin(plugin),
		_initialized(false) {
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override
	{
		GLState before = GLState::capture();
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		glBindVertexArray(coord_resource_.vao_);
		glBindBuffer(GL_ARRAY_BUFFER, coord_resource_.vbo_);
		glUseProgram(line_shader_.program_);

		scm::math::mat4 view_matrix_ = matConv4F(osg::Matrix(renderInfo.getState()->getModelViewMatrix()));
		scm::math::mat4 projection_matrix_ = matConv4F(osg::Matrix(renderInfo.getState()->getProjectionMatrix()));
		scm::math::mat4 mvp_matrix = projection_matrix_ * view_matrix_;

		glUniformMatrix4fv(line_shader_.mvp_matrix_location, 1, GL_FALSE, mvp_matrix.data_array);
		glUniform4f(line_shader_.in_color_location, settings_.frustum_color_[0], settings_.frustum_color_[1], settings_.frustum_color_[2], settings_.frustum_color_[3]);
		glDrawElements(GL_LINES, coord_resource_.idx_.size(), GL_UNSIGNED_SHORT, nullptr);

		glPopAttrib();
		before.restore();
	}
	osg::ref_ptr<osg::StateSet> _stateset;
	LamurePointCloudPlugin* _plugin;
	mutable bool _initialized;
};


struct CoordGeometry : public osg::Geometry
{
	CoordGeometry(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
		: _stateset(stateset), _plugin(plugin)
	{
		if (plugin->notify_button->state()) { std::cout << "[Notify] CoordGeometry()" << std::endl; }
		setUseDisplayList(false);
		setUseVertexBufferObjects(true);
		setUseVertexArrayObject(false);
		setDrawCallback(new CoordDrawCallback(stateset, plugin));
	}
	osg::ref_ptr<osg::StateSet> _stateset;
	LamurePointCloudPlugin* _plugin;
};


struct FrustumDrawCallback : public osg::Drawable::DrawCallback
{
	FrustumDrawCallback(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
		: _stateset(stateset),
		_plugin(plugin),
		_initialized(false) {
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override
	{
		GLState before = GLState::capture();

		if (!_initialized)
		{
			_plugin->init_frustum_resources();
			_initialized = true;
		}

		std::vector<scm::math::vec3d> corner_values = scm_camera_->get_frustum_corners();
		for (size_t i = 0; i < corner_values.size(); ++i) {
			auto vv = scm::math::vec3f(corner_values[i]);
			frustum_resource_.vertices_[i * 3 + 0] = vv.x;
			frustum_resource_.vertices_[i * 3 + 1] = vv.y;
			frustum_resource_.vertices_[i * 3 + 2] = vv.z;
		}

		glBindVertexArray(frustum_resource_.vao_);
		glBindBuffer(GL_ARRAY_BUFFER, frustum_resource_.vbo_);
		glBufferData(GL_ARRAY_BUFFER, sizeof(float) * frustum_resource_.vertices_.size(), frustum_resource_.vertices_.data(), GL_STATIC_DRAW);
		glUseProgram(line_shader_.program_);

		scm::math::mat4 view_matrix_ = matConv4F(osg::Matrix(renderInfo.getState()->getModelViewMatrix()));
		scm::math::mat4 projection_matrix_ = matConv4F(osg::Matrix(renderInfo.getState()->getProjectionMatrix()));
		scm::math::mat4 mvp_matrix = projection_matrix_ * view_matrix_;

		glUniformMatrix4fv(line_shader_.mvp_matrix_location, 1, GL_FALSE, mvp_matrix.data_array);
		glUniform4f(line_shader_.in_color_location, settings_.frustum_color_[0], settings_.frustum_color_[1], settings_.frustum_color_[2], settings_.frustum_color_[3]);
		glDrawElements(GL_LINES, frustum_resource_.idx_.size(), GL_UNSIGNED_SHORT, nullptr);

		before.restore();
	}
	osg::ref_ptr<osg::StateSet> _stateset;
	LamurePointCloudPlugin* _plugin;
	mutable bool _initialized;
};

struct FrustumGeometry : public osg::Geometry
{
	FrustumGeometry(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
	{
		if (plugin->notify_button->state()) { std::cout << "[Notify] FrustumGeometryGL()" << std::endl; }
		setUseDisplayList(false);
		setUseVertexBufferObjects(true);
		setUseVertexArrayObject(false);
		setDrawCallback(new FrustumDrawCallback(stateset, plugin));
	}
};


struct BoundingBoxDrawCallback : public virtual osg::Drawable::DrawCallback
{
	BoundingBoxDrawCallback(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
		: _stateset(stateset),
		_plugin(plugin)
	{ 
		if (plugin->notify_button->state()) { std::cout << "[Notify] BoundingBoxDrawCallback()" << std::endl; } 
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const {

		GLState before = GLState::capture();
		glPushAttrib(GL_ALL_ATTRIB_BITS);

		osg::State* state = renderInfo.getState();
		//state->setCheckForGLErrors(osg::State::CheckForGLErrors::ONCE_PER_ATTRIBUTE);
		scm::math::mat4 view_matrix_ = matConv4F(state->getModelViewMatrix());
		scm::math::mat4 projection_matrix_ = matConv4F(state->getProjectionMatrix());
		scm::math::mat4 osg_scale = matConv4F(cover->getObjectsScale()->getMatrix());

		lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
		lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
		lamure::ren::controller* controller = lamure::ren::controller::get_instance();
		lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();
		if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) { controller->reset_system(data_provenance_); }
		else { controller->reset_system(); }

		lamure::context_t context_id = controller->deduce_context_id(lmr_ctx);
		for (lamure::model_t model_id = 0; model_id < num_models_; ++model_id) {
			lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
			cuts->send_transform(context_id, m_id, osg_scale * scm::math::mat4(model_info_.model_transformations_[m_id]));
			cuts->send_threshold(context_id, m_id, settings_.lod_error_);
			cuts->send_rendered(context_id, m_id);
			database->get_model(m_id)->set_transform(osg_scale * scm::math::mat4(model_info_.model_transformations_[m_id]));
		}

		lamure::view_t view_id = controller->deduce_view_id(context_id, scm_camera_->view_id());
		cuts->send_camera(context_id, view_id, *scm_camera_);
		std::vector<scm::math::vec3d> corner_values = scm_camera_->get_frustum_corners();
		double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
		height_divided_by_top_minus_bottom_ = traits->height / top_minus_bottom;
		cuts->send_height_divided_by_top_minus_bottom(context_id, view_id, height_divided_by_top_minus_bottom_);

		if (settings_.use_pvs_) {
			scm::math::vec3d cam_pos = scm_camera_->get_cam_pos();
			pvs->set_viewer_position(cam_pos);
		}

		if (settings_.lod_update_) {
			if (lamure::ren::policy::get_instance()->size_of_provenance() > 0)
			{ controller->dispatch(context_id, device_, data_provenance_); }
			else { controller->dispatch(context_id, device_); }
		}

		glBindVertexArray(box_resource_.vao_);
		glUseProgram(line_shader_.program_);
		glUniform4f(line_shader_.in_color_location, settings_.bvh_color_[0], settings_.bvh_color_[1], settings_.bvh_color_[2], settings_.bvh_color_[3]);

		uint64_t rendered_bounding_boxes = 0;
		for (uint16_t model_id = 0; model_id < num_models_; ++model_id) {
			if (!_plugin->model_visible_[model_id]) { continue; }
			lamure::ren::cut& cut = cuts->get_cut(context_id, lmr_ctx, model_id);
			std::vector<lamure::ren::cut::node_slot_aggregate> renderable = cut.complete_set();
			const lamure::ren::bvh* bvh = database->get_model(model_id)->get_bvh();
			std::vector<scm::gl::boxf>const& bbv = bvh->get_bounding_boxes();
			scm::math::mat4 model_matrix_ = scm::math::mat4(model_info_.model_transformations_[model_id]);
			scm::math::mat4 mvp_matrix_ = projection_matrix_ * view_matrix_ * model_matrix_;
			scm::gl::frustum frustum_ = scm_camera_->get_frustum_by_model(model_matrix_);
			glUniformMatrix4fv(line_shader_.mvp_matrix_location, 1, GL_FALSE, mvp_matrix_.data_array);
			for (auto const& node_slot_aggregate : renderable) {
				uint32_t node_culling_result = scm_camera_->cull_against_frustum(frustum_, bbv[node_slot_aggregate.node_id_]);
				if (node_culling_result != 1) {
					const std::vector<float>& corners_ = bvh_res_[model_id].corners_[node_slot_aggregate.node_id_];
					glBindBuffer(GL_ARRAY_BUFFER, box_resource_.vbo_);
					glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(float) * corners_.size(), corners_.data());
					glDrawElements(GL_LINES, 24, GL_UNSIGNED_SHORT, nullptr);
					rendered_bounding_boxes++;
				}
			}
		}
		render_info_.rendered_bounding_boxes_ = rendered_bounding_boxes;
		glPopAttrib();
		before.restore();
	};
	osg::ref_ptr<osg::StateSet> _stateset;
	LamurePointCloudPlugin* _plugin;
};


struct BoundingBoxGeometry : public osg::Geometry
{
	BoundingBoxGeometry(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
	{
		if (plugin->notify_button->state()) { std::cout << "[Notify] BoundingBoxGeometry()" << std::endl; }
		setUseDisplayList(false);
		setUseVertexBufferObjects(true);
		setUseVertexArrayObject(false);
		setDrawCallback(new BoundingBoxDrawCallback(stateset, plugin));
	}
};


struct PointsDrawCallback : public virtual osg::Drawable::DrawCallback
{
	PointsDrawCallback(osg::ref_ptr<osg::StateSet> pointcloud_stateset, LamurePointCloudPlugin* plugin)
		: _stateset(pointcloud_stateset),
		_plugin(plugin),
		_initialized(false)
	{ if (_plugin->notify_button->state()) { std::cout << "[Notify] PointsDrawCallback()" << std::endl; } }

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const
	{
		if (_plugin->rendering_) { return; }
		_plugin->rendering_ = true;

		GLState before = GLState::capture();
		glDisable(GL_CULL_FACE);

		osg::State* state = renderInfo.getState();
		state->setCheckForGLErrors(osg::State::CheckForGLErrors::ONCE_PER_ATTRIBUTE);

		scm::math::mat4 view_matrix_ = matConv4F(osg::Matrix(renderInfo.getState()->getModelViewMatrix()));
		scm::math::mat4 projection_matrix_ = matConv4F(osg::Matrix(renderInfo.getState()->getProjectionMatrix()));
		scm::math::mat4 osg_scale = matConv4F(cover->getObjectsScale()->getMatrix());

		lamure::ren::model_database* database = lamure::ren::model_database::get_instance();
		lamure::ren::cut_database* cuts = lamure::ren::cut_database::get_instance();
		lamure::ren::controller* controller = lamure::ren::controller::get_instance();
		lamure::pvs::pvs_database* pvs = lamure::pvs::pvs_database::get_instance();

		if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) { controller->reset_system(data_provenance_); }
		else { controller->reset_system(); }

		lamure::context_t context_id = controller->deduce_context_id(lmr_ctx);
		lamure::view_t    view_id = controller->deduce_view_id(context_id, scm_camera_->view_id());
		size_t surfels_per_node = database->get_primitives_per_node();

		for (lamure::model_t model_id = 0; model_id < num_models_; ++model_id) {
			lamure::model_t m_id = controller->deduce_model_id(std::to_string(model_id));
			cuts->send_transform(context_id, m_id, scm::math::mat4(model_info_.model_transformations_[m_id]));
			cuts->send_threshold(context_id, m_id, settings_.lod_error_);
			cuts->send_rendered(context_id, m_id);
			database->get_model(m_id)->set_transform(scm::math::mat4(model_info_.model_transformations_[m_id]));
		}
		cuts->send_camera(context_id, view_id, *scm_camera_);
		std::vector<scm::math::vec3d> corner_values = scm_camera_->get_frustum_corners();
		double top_minus_bottom = scm::math::length((corner_values[2]) - (corner_values[0]));
		height_divided_by_top_minus_bottom_ = traits->height / top_minus_bottom;
		cuts->send_height_divided_by_top_minus_bottom(context_id, view_id, height_divided_by_top_minus_bottom_);

		if (settings_.use_pvs_) {
			scm::math::vec3d cam_pos = scm_camera_->get_cam_pos();
			pvs->set_viewer_position(cam_pos);
		}
		if (settings_.lod_update_) {
			if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) 
			{ controller->dispatch(context_id, device_, data_provenance_); }
			else { controller->dispatch(context_id, device_); }
		}

		if (_initialized) { glBindVertexArray(pcl_resource_.vao_); }
		context_->apply_vertex_input();
		if (lamure::ren::policy::get_instance()->size_of_provenance() > 0) {
			context_->bind_vertex_array(controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, device_, data_provenance_));
		}
		else { context_->bind_vertex_array(controller->get_context_memory(context_id, lamure::ren::bvh::primitive_type::POINTCLOUD, device_)); }

		const scm::math::mat4d viewport_scale = scm::math::make_scale(traits->width * 0.5, traits->height * 0.5, 0.5);
		const scm::math::mat4d viewport_translate = scm::math::make_translation(1.0, 1.0, 1.0);
		const scm::math::mat4 model_to_screen_matrix_ = scm::math::mat4(viewport_scale * viewport_translate);
		scm::math::vec3 eye_ = scm::math::vec3f(scm_camera_->get_cam_pos());
		scm::math::vec2 viewport_ = scm::math::vec2f(traits->width, traits->height);

		if (!settings_.surfel_shader_ && !settings_.provenance_) {
			glUseProgram(point_shader_.program);
			glEnable(GL_POINT_SMOOTH);
			glEnable(GL_PROGRAM_POINT_SIZE);
			glUniform1f(point_shader_.max_radius_loc, settings_.max_radius_);
			glUniform1f(point_shader_.scale_radius_loc, settings_.scale_radius_);
			glUniform1f(point_shader_.point_size_factor_loc,  settings_.point_size_factor_ * cover->getScale());
			glUniform1f(point_shader_.proj_scale_loc, viewport_.y * 0.5f * projection_matrix_.data_array[5]);
		}
		else if (settings_.surfel_shader_ && !settings_.provenance_) {
			glUseProgram(surfel_shader_.program);
			glUniform1f(surfel_shader_.max_radius_loc, settings_.max_radius_);
			glUniform1f(surfel_shader_.scale_radius_loc,settings_.scale_radius_);
			glUniform1f(surfel_shader_.surfel_size_factor_loc,settings_.surfel_size_factor_);
			glUniform1f(surfel_shader_.proj_scale_loc, viewport_.y * 0.5f * projection_matrix_.data_array[5]);
			glUniform2f(surfel_shader_.viewport_loc, viewport_.x, viewport_.y);
		}

		uint64_t rendered_splats_ = 0;
		uint64_t rendered_nodes_ = 0;

		for (uint16_t model_id = 0; model_id < num_models_; ++model_id) {
			if (!_plugin->model_visible_[model_id]) { continue; }
			lamure::ren::cut& cut = cuts->get_cut(context_id, lmr_ctx, model_id);
			std::vector<lamure::ren::cut::node_slot_aggregate> renderable = cut.complete_set();
			const lamure::ren::bvh* bvh = database->get_model(model_id)->get_bvh();
			std::vector<scm::gl::boxf>const& bounding_box_vector = bvh->get_bounding_boxes();

			scm::math::mat4 model_matrix_ = scm::math::mat4(model_info_.model_transformations_[model_id]);
			scm::math::mat4 model_view_matrix_ = view_matrix_ * model_matrix_;
			scm::math::mat4 mvp_matrix_ = projection_matrix_ * model_view_matrix_;
			scm::gl::frustum frustum_ = scm_camera_->get_frustum_by_model(model_matrix_);

			if (!settings_.surfel_shader_ && !settings_.provenance_) {
				glUniformMatrix4fv(point_shader_.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix_.data_array);
			}
			else if (settings_.surfel_shader_ && !settings_.provenance_) {
				glUniformMatrix4fv(surfel_shader_.model_view_matrix_loc,     1, GL_FALSE, model_view_matrix_.data_array);
				glUniformMatrix4fv(surfel_shader_.mvp_matrix_loc, 1, GL_FALSE, mvp_matrix_.data_array);
			}

			for (auto const& node_slot_aggregate : renderable) {
				if (scm_camera_->cull_against_frustum(frustum_, bounding_box_vector[node_slot_aggregate.node_id_]) != 1) {
					glDrawArrays(scm::gl::PRIMITIVE_POINT_LIST, (node_slot_aggregate.slot_id_) * (GLsizei)surfels_per_node, surfels_per_node);
					rendered_splats_ += surfels_per_node;
					++rendered_nodes_;
				}
			}
		}

		if (_plugin->dump_button->state()) { 
			_plugin->debug_print_settings();
			_plugin->dump_button->setState(0);
		}

		render_info_.rendered_splats_ = rendered_splats_;
		render_info_.rendered_nodes_ = rendered_nodes_;
		_plugin->rendering_ = false;

		if (!_initialized) {
			GLState after = GLState::capture();
			if (after.getVertexArrayBinding()
				!= before.getVertexArrayBinding())
			{
				pcl_resource_.vao_ = after.getVertexArrayBinding();
				_initialized = true;
			}
		}

		before.restore();
		if (_plugin->notify_button->state()) {
			GLState after = GLState::capture();
			GLState::compare(before, after, "[Notify] PointsDrawCallback::drawImplementation()");
		}
	}
	osg::ref_ptr<osg::StateSet> _stateset;
	LamurePointCloudPlugin* _plugin;
	mutable bool _initialized;
};


struct PointsGeometry : public osg::Geometry
{
	PointsGeometry(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin) :
		_plugin(plugin)
	{
		if (plugin->notify_button->state()) { std::cout << "[Notify] PointsGeometry()" << std::endl; }
		setUseDisplayList(false);
		setUseVertexBufferObjects(true);
		setUseVertexArrayObject(false);
		setDrawCallback(new PointsDrawCallback(stateset, plugin));

		osg::Vec3 minPt = vecConv3F(model_info_.models_min);
		osg::Vec3 maxPt = vecConv3F(model_info_.models_max);
		osg::Vec3 halfExtents(std::max(fabs(minPt.x()), fabs(maxPt.x())),
			std::max(fabs(minPt.y()), fabs(maxPt.y())),
			std::max(fabs(minPt.z()), fabs(maxPt.z())));
		_bbox = osg::BoundingBox(-halfExtents, halfExtents);
		_bsphere = osg::BoundingSphere(_bbox.center(), _bbox.radius());

		setInitialBound(_bbox);
	}
	LamurePointCloudPlugin* _plugin;
	osg::BoundingSphere _bsphere;
	osg::BoundingBox _bbox;
};


struct InitDrawCallback : public osg::Drawable::DrawCallback {
	InitDrawCallback(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin)
		: _stateset(stateset),
		_plugin(plugin),
		_initialized(false)
	{
		if (plugin->notify_button->state()) { std::cout << "[Notify] InitDrawCallback()" << std::endl; }
	}

	virtual void drawImplementation(osg::RenderInfo& renderInfo, const osg::Drawable* drawable) const override
	{
		_plugin->gl_modelview_matrix = matConv4D(osg::Matrixd(renderInfo.getState()->getModelViewMatrix()));
		_plugin->gl_projection_matrix = matConv4D(osg::Matrixd(renderInfo.getState()->getProjectionMatrix()));

		if (_plugin->sync_button->state() == 1) {
			scm_camera_->set_view_matrix(_plugin->gl_modelview_matrix);
			scm_camera_->set_projection_matrix(_plugin->gl_projection_matrix);
		}

		if (!_initialized) {
			GLState before = GLState::capture();

			_plugin->init_schism_objects();
			_plugin->HDC_draw = wglGetCurrentDC();
			_plugin->HGLRC_draw = wglGetCurrentContext();

			_plugin->create_framebuffers();
			_plugin->init_render_states();
			_plugin->init_lamure_shader();
			_plugin->init_uniforms();
			_plugin->init_frustum_resources();
			_plugin->init_box_resources();
			_plugin->init_coord_resources();

			_plugin->pointcloud_geode->setNodeMask(_plugin->pointcloud_button->state() ? 0xFFFFFFFF : 0x0);
			_plugin->boundingbox_geode->setNodeMask(_plugin->boundingbox_button->state() ? 0xFFFFFFFF : 0x0);
			_plugin->frustum_geode->setNodeMask(_plugin->frustum_button->state() ? 0xFFFFFFFF : 0x0);
			_plugin->coord_geode->setNodeMask(_plugin->coord_button->state() ? 0xFFFFFFFF : 0x0);
			_plugin->text_geode->setNodeMask(_plugin->text_button->state() ? 0xFFFFFFFF : 0x0);

			_initialized = true;
			before.restore();
		}
	}
	osg::ref_ptr<osg::StateSet> _stateset;
	LamurePointCloudPlugin* _plugin;
	mutable bool _initialized;
};


struct InitGeometry : public osg::Geometry {
	InitGeometry(osg::ref_ptr<osg::StateSet> stateset, LamurePointCloudPlugin* plugin):
		_plugin(plugin)
	{
		if (plugin->notify_button->state()) { std::cout << "[Notify] InitGeometry()" << std::endl; }
		osg::ref_ptr<osg::StateSet> stateSet = new osg::StateSet();
		setUseDisplayList(false);
		setUseVertexBufferObjects(true);
		setUseVertexArrayObject(false);
		setDrawCallback(new InitDrawCallback(stateset, plugin));
	}
	LamurePointCloudPlugin* _plugin;
};


void updateFrustumTransform(osg::ref_ptr<osg::MatrixTransform> matrixTransform, const osg::Vec3& translation) {
	osg::Matrix transMatrix = osg::Matrix::translate(translation);
	matrixTransform->setMatrix(transMatrix);
};


bool LamurePointCloudPlugin::init2() {
	std::cout << "init2()" << std::endl;
	std::cout << "getConfigEntry(COVER.Plugin.LamurePointCloud).c_str(): " << getConfigEntry("COVER.Plugin.LamurePointCloud").c_str() << std::endl;
	file = coVRFileManager::instance()->loadFile(getConfigEntry("COVER.Plugin.LamurePointCloud").c_str());
	std::cerr << "hostname: " << covise::coConfigConstants::getHostname() << std::endl;

	LamureGroup = new osg::Group();
	LamureGroup->setName("LamureGroup");
	cover->getObjectsRoot()->addChild(plugin->LamureGroup);
	osg::ref_ptr<osg::MatrixTransform> frustumTransform = new osg::MatrixTransform;

	lamure_menu = new ui::Menu("Lamure", this);
	lamure_menu->setText("Lamure");
	lamure_menu->allowRelayout(true);

	selection_group = new ui::Group(lamure_menu, "Selection");

	//plugin->debug_print_settings();

	pointcloud_button = new ui::Button(selection_group, "pointcloud");
	boundingbox_button = new ui::Button(selection_group, "boundingboxes");
	frustum_button = new ui::Button(selection_group, "frustum");
	coord_button = new ui::Button(selection_group, "coordinates");
	sync_button = new ui::Button(selection_group, "sync");
	notify_button = new ui::Button(selection_group, "notify");
	text_button = new ui::Button(selection_group, "text");
	dump_button = new ui::Button(selection_group, "dump");

	pointcloud_button->setShared(true);
	boundingbox_button->setShared(true);
	frustum_button->setShared(true);
	coord_button->setShared(true);
	sync_button->setShared(true);
	notify_button->setShared(true);
	text_button->setShared(true);
	dump_button->setShared(true);

	model_group = new ui::Group(lamure_menu, "Modelle");

	model_buttons_.clear();
	model_visible_.clear();
	for (uint16_t m_id = 0; m_id < num_models_; m_id++) {
		std::filesystem::path pathObj(settings_.models_[m_id]);
		std::string filename = pathObj.filename().string();
		std::string filename_strip = pathObj.stem().string();
		opencover::ui::Button* btn = new ui::Button(model_group, filename_strip, nullptr, m_id);
		model_group->add(btn);
		btn->setShared(true);
		bool checked = settings_.selection_.empty() ||
			(std::find(settings_.selection_.begin(), settings_.selection_.end(), m_id) != settings_.selection_.end());
		btn->setState(checked);
		model_visible_.push_back(checked);
		model_buttons_.push_back(btn);
		btn->setCallback([this, m_id](bool state) { model_visible_[m_id] = state; });
	}

	osg_util::waitForOpenGLContext();

	plugin->HGLRC_init = wglGetCurrentContext();
	plugin->HDC_init = wglGetCurrentDC();
	plugin->HWND_init = WindowFromDC(plugin->HDC_init);
	plugin->HWND_cover = FindWindow(NULL, "COVER");
	plugin->HWND_opencover = FindWindow(NULL, "OpenCOVER");
	plugin->HDC_cover = GetDC(plugin->HWND_cover);
	plugin->HDC_opencover = GetDC(plugin->HWND_opencover);

	std::cout << "HGLRC_init: " << wglGetCurrentContext() << std::endl;
	std::cout << "HDC_init: " << wglGetCurrentDC() << std::endl;
	std::cout << "HDC_cover: " << GetDC(FindWindow(NULL, "COVER")) << std::endl;
	std::cout << "HDC_opencover: " << GetDC(FindWindow(NULL, "OpenCOVER")) << std::endl;
	std::cout << "HWND_init: " << WindowFromDC(wglGetCurrentDC()) << std::endl;
	std::cout << "HWND_cover: " << FindWindow(NULL, "COVER") << std::endl;
	std::cout << "HWND_opencover: " << FindWindow(NULL, "OpenCOVER") << std::endl;

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageCallback(openglCallbackFunction, nullptr);

	plugin->init_camera();

	adaption_group = new ui::Group(lamure_menu, "Adaption");

	maxRadiusSlider = new ui::Slider(adaption_group, "max_radius");
	maxRadiusSlider->setText("max. radius");
	maxRadiusSlider->setBounds(0.0, settings_.max_radius_ * 5.0f);
	maxRadiusSlider->setValue(settings_.max_radius_);
	maxRadiusSlider->setShared(true);
	maxRadiusSlider->setCallback([this](double value, bool released) { settings_.max_radius_ = static_cast<float>(value); });

	scaleRadiusSlider = new ui::Slider(adaption_group, "scale_radius");
	scaleRadiusSlider->setText("scale radius");
	scaleRadiusSlider->setBounds(0.0001, settings_.scale_radius_ * 5.0f);
	scaleRadiusSlider->setValue(settings_.scale_radius_);
	scaleRadiusSlider->setScale(ui::Slider::Logarithmic);
	scaleRadiusSlider->setShared(true);
	scaleRadiusSlider->setCallback([this](double value, bool released) { settings_.scale_radius_ = static_cast<float>(value); });

	lodErrorSlider_ = new ui::Slider(adaption_group, "lod_error");
	lodErrorSlider_->setText("LOD Error");
	lodErrorSlider_->setBounds(LAMURE_MIN_THRESHOLD, LAMURE_MAX_THRESHOLD);
	lodErrorSlider_->setValue(settings_.lod_error_);
	lodErrorSlider_->setShared(true);
	lodErrorSlider_->setCallback([this](double value, bool released) { settings_.lod_error_ = static_cast<float>(value); });

	uploadBudgetSlider_ = new ui::Slider(adaption_group, "upload_budget");
	uploadBudgetSlider_->setText("Upload Budget (MB)");
	uploadBudgetSlider_->setBounds(LAMURE_MIN_UPLOAD_BUDGET, 256);
	uploadBudgetSlider_->setValue(lamure::ren::policy::get_instance()->max_upload_budget_in_mb());
	uploadBudgetSlider_->setShared(true);
	uploadBudgetSlider_->setCallback([this](double v, bool) {
		lamure::ren::policy::get_instance()->set_max_upload_budget_in_mb(static_cast<size_t>(v));
		});

	//maxUpdatesSlider_ = new ui::Slider(adaption_group, "max_cut_updates");
	//maxUpdatesSlider_->setText("Max Cut-Updates/Frame");
	//maxUpdatesSlider_->setBounds(1.0, 32.0);
	//maxUpdatesSlider_->setValue(static_cast<double>(LAMURE_CUT_UPDATE_MAX_NUM_UPDATES_PER_FRAME));
	//maxUpdatesSlider_->setShared(true);
	//maxUpdatesSlider_->setCallback([this](double v, bool) {
	//	lamure::ren::controller::get_instance()->set_max_updates_per_frame(static_cast<uint32_t>(v));
	//});

	rendering_group = new ui::Group(lamure_menu, "Rendering");

	_surfelButton = new ui::Button(rendering_group, "surfel_shader");
	_surfelButton->setText("surfel shader");
	_surfelButton->setShared(true);
	_surfelButton->setState(settings_.surfel_shader_);
	_surfelButton->setCallback([this](bool state) { settings_.surfel_shader_ = state; });
	rendering_group->add(_surfelButton);

	if (prov_valid) { 
		_provButton = new ui::Button(rendering_group, "provenance");
		_provButton->setText("provenance");
		_provButton->setShared(true);
		_provButton->setState(settings_.provenance_);
		_provButton->setCallback([this](bool state) { settings_.provenance_ = (state);  });
		rendering_group->add(_provButton); 
	}

	_measureButton = new ui::Button(rendering_group, "measurement");
	_measureButton->setShared(true);
	_measureButton->setState(false);
	rendering_group->add(_measureButton);

	_measureCB = [this](bool state) {
		if (state) startMeasurement();
		else       stopMeasurement();
		};
	_measureButton->setCallback(_measureCB);

	init_stateset = new osg::StateSet();
	pointcloud_stateset = new osg::StateSet();
	boundingbox_stateset = new osg::StateSet();
	frustum_stateset = new osg::StateSet();
	coord_stateset = new osg::StateSet();
	text_stateset = new osg::StateSet();

	init_geode = new osg::Geode();
	pointcloud_geode = new osg::Geode();
	boundingbox_geode = new osg::Geode();
	frustum_geode = new osg::Geode();
	coord_geode = new osg::Geode();
	text_geode = new TextGeode(plugin);

	pointcloud_button->setState(true);
	boundingbox_button->setState(false);
	frustum_button->setState(false);
	coord_button->setState(false);
	text_button->setState(true);
	sync_button->setState(true);
	notify_button->setState(true);

	pointcloud_geode->setNodeMask(0);
	boundingbox_geode->setNodeMask(0);
	frustum_geode->setNodeMask(0);
	coord_geode->setNodeMask(0);
	text_geode->setNodeMask(0);

	pointcloud_button->setCallback([this](bool state) { pointcloud_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
	boundingbox_button->setCallback([this](bool state) { boundingbox_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
	frustum_button->setCallback([this](bool state) { frustum_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
	coord_button->setCallback([this](bool state) { coord_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
	text_button->setCallback([this](bool state) { text_geode->setNodeMask(state ? 0xFFFFFFFF : 0x0); });
	dump_button->setCallback([this](bool state) {});

	text_stateset->setRenderBinDetails(10, "RenderBin");
	init_geode->setName("init_geode");
	init_geode->setStateSet(init_stateset.get());
	pointcloud_geode->setStateSet(pointcloud_stateset.get());
	boundingbox_geode->setStateSet(boundingbox_stateset.get());
	frustum_geode->setStateSet(frustum_stateset.get());
	coord_geode->setStateSet(coord_stateset.get());
	text_geode->setStateSet(text_stateset.get());

	updateFrustumTransform(frustumTransform, osg::Vec3(scm_camera_->get_cam_pos()[0], scm_camera_->get_cam_pos()[1], scm_camera_->get_cam_pos()[2]));
	LamureGroup->addChild(frustumTransform);
	LamureGroup->addChild(frustum_geode);
	LamureGroup->addChild(coord_geode);
	LamureGroup->addChild(boundingbox_geode);
	LamureGroup->addChild(pointcloud_geode);
	LamureGroup->addChild(init_geode);
	hud_camera->addChild(text_geode);

	init_geometry = new InitGeometry(init_stateset, plugin);
	pointcloud_geometry = new PointsGeometry(pointcloud_stateset, plugin);
	boundingbox_geometry = new BoundingBoxGeometry(boundingbox_stateset, plugin);
	frustum_geometry = new FrustumGeometry(frustum_stateset, plugin);
	coord_geometry = new CoordGeometry(coord_stateset, plugin);

	init_geode->addDrawable(init_geometry);
	pointcloud_geode->addDrawable(pointcloud_geometry);
	boundingbox_geode->addDrawable(boundingbox_geometry);
	frustum_geode->addDrawable(frustum_geometry);
	coord_geode->addDrawable(coord_geometry);

	//pointcloud_geode->dirtyBound();
	//LamureGroup->dirtyBound();
	//cover->getObjectsRoot()->dirtyBound();

	if (covise::coCoviseConfig::isOn("COVER.showRotationPoint", false)) {
		coVRNavigationManager::instance()->setRotationPointVisible(true);
		trackball_.initial_pos_ = coVRNavigationManager::instance()->getRotationPoint();
		trackball_.pos_ = coVRNavigationManager::instance()->getRotationPoint();
		trackball_.size_ = covise::coCoviseConfig::getFloat("COVER.rotationPointSize", 1.0f);
		trackball_.dist_ = (trackball_.initial_pos_ - osg_camera_->getViewMatrix().getTrans()).length();
	}

	interactor = new LamurePointCloudInteractor();
	osg::ref_ptr<IntersectionHandler> handler = interactor;
	coIntersection::instance()->addHandler(handler);
	coVRNavigationManager::instance()->setNavMode("Point");

	if (plugin->notify_button->state()) {
		//std::cout << "[Notify] === SceneGraph ===" << std::endl;
		//printChildNodes(osg_camera_, 5);
	}
	return 1;
}

int counter = 0;

void LamurePointCloudPlugin::preFrame() {

	float deltaTime = std::clamp(float(cover->frameDuration()), 1.0f / 60.0f, 1.0f / 15.0f);
	float moveAmount = 1000.0f * deltaTime;

	//const float fixedFps = 60.0f;
	//float moveAmount = 1000.0f / fixedFps;

	osg::Matrix oldMat = VRSceneGraph::instance()->getTransform()->getMatrix();

	if (GetAsyncKeyState(VK_NUMPAD4) & 0x8000)
	{
		oldMat.postMult(osg::Matrix::translate(moveAmount, 0.0, 0.0));
	}
	//    VK_NUMPAD6 → translate X‐  
	if (GetAsyncKeyState(VK_NUMPAD6) & 0x8000)
	{
		oldMat.postMult(osg::Matrix::translate(-moveAmount, 0.0, 0.0));
	}
	//    VK_NUMPAD8 → translate Y+ (oder Z‐, je nach Achsen‐Konvention)
	if (GetAsyncKeyState(VK_NUMPAD8) & 0x8000)
	{
		oldMat.postMult(osg::Matrix::translate(0.0, -moveAmount, 0.0));
	}
	//    VK_NUMPAD5 → translate Y‐ (oder Z+, wenn Y‐Achse nach oben)
	if (GetAsyncKeyState(VK_NUMPAD5) & 0x8000)
	{
		oldMat.postMult(osg::Matrix::translate(0.0, moveAmount, 0.0));
	}

	VRSceneGraph::instance()->getTransform()->setMatrix(oldMat);

	if ((cover->getMouseButton()->getState() == 1) && (counter == 0)) {
		//std::cout << matConv4F(cover->getXformMat()) << std::endl;
		//coVRTui* tui = coVRTui::instance();
		//auto mainFolder = tui->mainFolder;
		//for (auto* child : mainFolder->children())
		//	std::cerr << "  " << child->objectName().toStdString() << std::endl;
		//auto* lamureTab = mainFolder->findChild<coTUITab*>("Lamure");
		//if (!lamureTab)
		//{
		//	std::cerr << "[ERROR] TUI-Tab 'Lamure' nicht gefunden! Prüfe den exacten objectName aus Schritt 2." << std::endl;
		//}
		//else {
		//	int tabID = lamureTab->getID();
		//	std::cout << "[INFO] Lamure-Tab gefunden, ID = " << tabID << std::endl;
		//}
		//int n = mainFolder->getNumChildren();
		//std::cerr << "Found " << n << " TUI tabs under COVERMainFolder\n";
		//for (int i = 0; i < n; ++i) {
		//	int childID = mainFolder->getChild(i);
		//	coTUIElement* elem = tui->getElement(childID);
		//	std::string name = elem->getName().toStdString();
		//	std::cerr << "  Tab[" << i << "] = " << name << "\n";
		//}
		//coVRNavigationManager::instance()->enableViewerPosRotation(true);
		//pointcloud_geode->setNodeMask(Isect::Visible);
		//pointcloud_geometry->setNodeMask(Isect::Visible);
		//osg::BoundingSphere bs = pointcloud_geometry->computeBound();
		//osg::BoundingBox bb = pointcloud_geometry->computeBoundingBox();

		counter = counter + 1;
	}

	if ((cover->getMouseButton()->getState() == 1) && (counter != 0)) { }
}


void LamurePointCloudPlugin::setViewerPos(float x, float y, float z)
{
	osg::Matrix dcs_mat;
	osg::Matrix doTrans;
	osg::Matrix tmp;
	osg::Vec3 viewerPos = cover->getViewerMat().getTrans();
	dcs_mat.postMult(osg::Matrix::translate(-viewerPos[0], -viewerPos[1], -viewerPos[2]));
	dcs_mat.postMult(osg::Matrix::translate(x, y, z));
	tmp.makeTranslate(viewerPos[0], viewerPos[1], viewerPos[2]);
	dcs_mat.postMult(tmp);
	VRSceneGraph::instance()->getTransform()->setMatrix(dcs_mat);
	coVRCollaboration::instance()->SyncXform();
}


void LamurePointCloudPlugin::startMeasurement() {
	std::cout << "startMeasurement(): " << _measureButton->state() << std::endl;
	std::vector<Measurement::Segment> _segments = {
		{ {0,-500,0},	{0,0,360},		200.0f, 30.0f },
		{ {0,0,0},		{45,0,360},		200.0f,	30.0f },
		{ {0,-400,0},	{0,0,0},		200.0f,	30.0f },
	};
	auto rendering_scheme = VRViewer::instance()->getRunFrameScheme();
	VRViewer::instance()->setRunFrameScheme(osgViewer::Viewer::CONTINUOUS);
	//_measurement = new Measurement(VRViewer::instance(), 1.0f, "", _measureButton, _measureCB);
	_measurement = new Measurement(VRViewer::instance(), _segments, "C:/Users/Daniel/Documents/Studium/Forschungsarbeit/Measurement/measurement1.txt", _measureButton, _measureCB);
}


void LamurePointCloudPlugin::stopMeasurement() {
	std::cout << "stopMeasurement(): " << _measureButton->state() << std::endl;
	VRViewer::instance()->setRunFrameScheme(rendering_scheme);
	_measurement->writeLogAndStop();
	_measureButton->setState(false);
	delete _measurement;
	_measurement = nullptr;
}


void LamurePointCloudPlugin::init_schism_objects() {
	if (!device_) {
		device_.reset(new scm::gl::render_device());
		if (!device_) { std::cout << "error creating device" << std::endl; }
		if (plugin->notify_button->state()) {
			std::cout << "[Notify] init_schism_objects()" << std::endl;
			std::ostringstream oss;
			device_->dump_memory_info(oss);
			std::cout << oss.str() << std::endl;
			//std::cout << (*device_);
			scm::gl::render_device::device_capabilities capa = device_->capabilities();
		}
	}
	if (!context_) {
		context_ = device_->main_context();
		if (!context_) { std::cout << "error creating context" << std::endl; }
	}
}


void LamurePointCloudPlugin::init_camera() {
	osg_camera_ = VRViewer::instance()->getCamera();
	lmr_ctx = osg_camera_->getGraphicsContext()->getState()->getContextID();

	double look_dist = 1.0;
	double left, right, bottom, top, zNear, zFar;
	osg::Vec3d eye, center, up;
	osg_camera_->getProjectionMatrixAsFrustum(left, right, bottom, top, zNear, zFar);
	osg_camera_->getViewMatrixAsLookAt(eye, center, up, look_dist);
	double fovy = 2.0 * std::atan((top - bottom) / (2.0 * zNear));
	double aspect = (right - left) / (top - bottom);

	std::cout << "OSG Projection Matrix as Frustum:" << std::endl;
	std::cout << "Left:  " << left << std::endl;
	std::cout << "Right: " << right << std::endl;
	std::cout << "Bottom:" << bottom << std::endl;
	std::cout << "Top:   " << top << std::endl;
	std::cout << "Near:  " << zNear << std::endl;
	std::cout << "Far:   " << zFar << std::endl;
	std::cout << "fovy (in Radiant): " << fovy << std::endl;
	std::cout << "fovy (in Grad): " << fovy * (180.0 / M_PI) << std::endl;
	std::cout << "Aspect Ratio: " << aspect << std::endl;
	std::cout << "OSG View Matrix as LookAt:" << std::endl;
	std::cout << "Eye:    (" << eye.x() << ", " << eye.y() << ", " << eye.z() << ")" << std::endl;
	std::cout << "Center: (" << center.x() << ", " << center.y() << ", " << center.z() << ")" << std::endl;
	std::cout << "Up:     (" << up.x() << ", " << up.y() << ", " << up.z() << ")" << std::endl;
	std::cout << "lookDistance: " << look_dist << std::endl;

	osg::Matrixd view = osg_camera_->getViewMatrix();
	osg::Matrixd proj = osg_camera_->getProjectionMatrix();

	scm_camera_ = new lamure::ren::camera((lamure::view_t)lmr_ctx, zNear, zFar, matConv4D(view), matConv4D(proj));
	//scm_camera_ = new lamure::ren::camera((lamure::view_t)lmr_ctx, left, right, bottom, top, zNear, zFar, vecConv3D(eye), vecConv3D(center), vecConv3D(up), look_dist);

	osgViewer::Viewer::Windows windows;
	VRViewer::instance()->getWindows(windows);
	osgViewer::GraphicsWindow* window = windows.front();
	hud_camera = new osg::Camera();
	hud_camera->setName("hud_camera");
	hud_camera->setGraphicsContext(window);
	hud_camera->setReferenceFrame(osg::Transform::ABSOLUTE_RF);
	hud_camera->setProjectionResizePolicy(osg::Camera::FIXED);
	hud_camera->setViewMatrix(osg_camera_->getViewMatrix());
	hud_camera->setProjectionMatrix(osg_camera_->getProjectionMatrix());
	hud_camera->setViewport(0, 0, traits->width, traits->height);
	hud_camera->setRenderOrder(osg::Camera::POST_RENDER, 2);
	hud_camera->setClearMask(0);
	hud_camera->setRenderer(new osgViewer::Renderer(hud_camera.get()));
	osg_camera_->addChild(hud_camera.get());

	scm::math::vec3f temp_center = scm::math::vec3f::zero();
	scm::math::vec3f root_min_temp = scm::math::vec3f::zero();
	scm::math::vec3f root_max_temp = scm::math::vec3f::zero();

	for (lamure::model_t model_id = 0; model_id < num_models_; ++model_id) {
		lamure::model_t m_id = lamure::ren::controller::get_instance()->deduce_model_id(std::to_string(model_id));

		auto root_bb = lamure::ren::model_database::get_instance()->get_model(model_id)->get_bvh()->get_bounding_boxes()[0];

		model_info_.root_bb_min.push_back(scm::math::mat4f(model_info_.model_transformations_[model_id]) * scm::math::vec4f(root_bb.min_vertex()[0], root_bb.min_vertex()[1], root_bb.min_vertex()[2], 1));
		model_info_.root_bb_max.push_back(scm::math::mat4f(model_info_.model_transformations_[model_id]) * scm::math::vec4f(root_bb.max_vertex()[0], root_bb.max_vertex()[1], root_bb.max_vertex()[2], 1));
		model_info_.root_center.push_back(scm::math::mat4f(model_info_.model_transformations_[model_id]) * scm::math::vec4f(root_bb.center()[0], root_bb.center()[1], root_bb.center()[2], 1));

		temp_center += model_info_.root_center.back();
		if (model_info_.root_bb_min[model_id][0] < root_min_temp[0]) { root_min_temp[0] = model_info_.root_bb_min[model_id][0]; }
		if (model_info_.root_bb_min[model_id][1] < root_min_temp[1]) { root_min_temp[1] = model_info_.root_bb_min[model_id][1]; }
		if (model_info_.root_bb_min[model_id][2] < root_min_temp[2]) { root_min_temp[2] = model_info_.root_bb_min[model_id][2]; }
		if (model_info_.root_bb_max[model_id][0] > root_max_temp[0]) { root_max_temp[0] = model_info_.root_bb_max[model_id][0]; }
		if (model_info_.root_bb_max[model_id][1] > root_max_temp[1]) { root_max_temp[1] = model_info_.root_bb_max[model_id][1]; }
		if (model_info_.root_bb_max[model_id][2] > root_max_temp[2]) { root_max_temp[2] = model_info_.root_bb_max[model_id][2]; }
	}
	model_info_.models_center = temp_center / num_models_;
	model_info_.models_min = root_min_temp;
	model_info_.models_max = root_max_temp;
}


void LamurePointCloudPlugin::init_frustum_resources() {
	if (plugin->notify_button->state()) {
		std::cout << "[Notify] create_frustum_resources() " << std::endl;
	}
	std::vector<scm::math::vec3d> corner_values = scm_camera_->get_frustum_corners();
	for (size_t i = 0; i < corner_values.size(); ++i) {
		auto vv = scm::math::vec3f(corner_values[i]);
		frustum_resource_.vertices_[i * 3 + 0] = vv.x;
		frustum_resource_.vertices_[i * 3 + 1] = vv.y;
		frustum_resource_.vertices_[i * 3 + 2] = vv.z;
	}
	GLuint vao_;
	glGenVertexArrays(1, &vao_);
	glBindVertexArray(vao_);
	GLuint ibo_;
	glGenBuffers(1, &ibo_);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, frustum_resource_.idx_.size() * sizeof(unsigned short), frustum_resource_.idx_.data(), GL_STATIC_DRAW);
	GLuint vbo_;
	glGenBuffers(1, &vbo_);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * frustum_resource_.vertices_.size(), frustum_resource_.vertices_.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
	frustum_resource_.vao_ = vao_;
	frustum_resource_.vbo_ = vbo_;
	frustum_resource_.ibo_ = ibo_;
	glBindVertexArray(0);
}

void LamurePointCloudPlugin::init_coord_resources() {
	if (plugin->notify_button->state()) { 
		std::cout << "[Notify] init_coord_resources() " << std::endl;
	}
	GLuint vao_;
	glGenVertexArrays(1, &vao_);
	glBindVertexArray(vao_);
	GLuint ibo_;
	glGenBuffers(1, &ibo_);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, coord_resource_.idx_.size() * sizeof(unsigned short), coord_resource_.idx_.data(), GL_STATIC_DRAW);
	GLuint vbo_;
	glGenBuffers(1, &vbo_);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * coord_resource_.vertices_.size(), coord_resource_.vertices_.data(), GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	coord_resource_.vao_ = vao_;
	coord_resource_.vbo_ = vbo_;
	coord_resource_.ibo_ = ibo_;
	glBindVertexArray(0);
}

void LamurePointCloudPlugin::init_box_resources() {
	if (plugin->notify_button->state()) { std::cout << "[Notify] init_box_resources() " << std::endl; }
	for (uint32_t model_id = 0; model_id < num_models_; ++model_id) {
		std::vector<vector<float>> corners_;
		const auto& bvh_ = lamure::ren::model_database::get_instance()->get_model(model_id)->get_bvh();
		const auto& bounding_boxes = bvh_->get_bounding_boxes();
		for (uint64_t node_id = 0; node_id < bounding_boxes.size(); ++node_id) {
			corners_.push_back(plugin->getBoxCorners(bounding_boxes[node_id]));
		}
		bvh_res_[model_id].corners_ = corners_;
	}
	vector<float> vertices_ = plugin->getBoxCorners(lamure::ren::model_database::get_instance()->get_model(0)->get_bvh()->get_bounding_boxes()[0]);
	GLuint vao_;
	glGenVertexArrays(1, &vao_);
	glBindVertexArray(vao_);
	GLuint ibo_;
	glGenBuffers(1, &ibo_);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, box_resource_.idx_.size() * sizeof(unsigned short), box_resource_.idx_.data(), GL_STATIC_DRAW);
	GLuint vbo_;
	glGenBuffers(1, &vbo_);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * vertices_.size(), vertices_.data(), GL_STREAM_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
	box_resource_.vao_ = vao_;
	box_resource_.vbo_ = vbo_;
	box_resource_.ibo_ = ibo_;
	glBindVertexArray(0);
}

void LamurePointCloudPlugin::create_aux_resources() {
	if (!settings_.create_aux_resources_) {
		return;
	}
	GLuint vbo_s;
	glGenBuffers(1, &vbo_s);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_s);
	glBufferData(GL_ARRAY_BUFFER, sphere_resource_.points.size() * sizeof(float), sphere_resource_.points.data(), GL_STREAM_DRAW);
	GLuint vao_s;
	glGenVertexArrays(1, &vao_s);
	glBindVertexArray(vao_s);
	glBindVertexArray(0);
	sphere_resource_.vbo_ = vbo_s;
	sphere_resource_.vao_ = vao_s;
	GLuint ibo_p;
	glGenBuffers(1, &ibo_p);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo_p);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, plane_resource_.idx_.size() * sizeof(unsigned short), plane_resource_.idx_.data(), GL_STATIC_DRAW);
	plane_resource_.ibo_ = ibo_p;
}

scm::math::mat4d LamurePointCloudPlugin::swapMiddleColumns(const scm::math::mat4d& m)
{
	scm::math::mat4d swapMatrix = scm::math::mat4d::identity();
	swapMatrix[5] = 0.0f;
	swapMatrix[6] = -1.0f;
	swapMatrix[9] = 1.0f;
	swapMatrix[10] = 0.0f;
	scm::math::mat4d result(m * swapMatrix);
	return result;
}

scm::math::mat4d LamurePointCloudPlugin::swapMiddleColumns(scm::math::mat4d& m)
{
	scm::math::mat4d swapMatrix = scm::math::mat4d::identity();
	swapMatrix[5] = 0.0f;
	swapMatrix[6] = -1.0f;
	swapMatrix[9] = 1.0f;
	swapMatrix[10] = 0.0f;
	scm::math::mat4d result(m * swapMatrix);
	return result;
}

scm::math::mat4d LamurePointCloudPlugin::swapMiddleRows(scm::math::mat4d& m)
{
	scm::math::mat4d swapMatrix = scm::math::mat4d::identity();
	swapMatrix[5] = 0.0f;
	swapMatrix[6] = 1.0f;
	swapMatrix[9] = -1.0f;
	swapMatrix[10] = 0.0f;
	scm::math::mat4d result(swapMatrix * m);
	return result;
}

scm::math::mat4d LamurePointCloudPlugin::swapMiddleRows(const scm::math::mat4d& m)
{
	scm::math::mat4d swapMatrix = scm::math::mat4d::identity();
	swapMatrix[5] = 0.0f;
	swapMatrix[6] = 1.0f;
	swapMatrix[9] = -1.0f;
	swapMatrix[10] = 0.0f;
	scm::math::mat4d result(swapMatrix * m);
	return result;
}

void LamurePointCloudPlugin::init_uniforms() {
	cout << "[Notify] init_uniforms()" << endl;

	glUseProgram(point_shader_.program);
	point_shader_.mvp_matrix_loc         = glGetUniformLocation(point_shader_.program, "mvp_matrix");
	point_shader_.max_radius_loc         = glGetUniformLocation(point_shader_.program, "max_radius");
	point_shader_.scale_radius_loc		 = glGetUniformLocation(point_shader_.program, "scale_radius");
	point_shader_.point_size_factor_loc  = glGetUniformLocation(point_shader_.program, "point_size_factor");
	point_shader_.proj_scale_loc         = glGetUniformLocation(point_shader_.program, "proj_scale");

	glUseProgram(surfel_shader_.program);
	surfel_shader_.mvp_matrix_loc           = glGetUniformLocation(surfel_shader_.program, "mvp_matrix");
	surfel_shader_.max_radius_loc           = glGetUniformLocation(surfel_shader_.program, "max_radius");
	surfel_shader_.scale_radius_loc			= glGetUniformLocation(surfel_shader_.program, "scale_radius");
	surfel_shader_.surfel_size_factor_loc   = glGetUniformLocation(surfel_shader_.program, "surfel_size_factor");
	surfel_shader_.proj_scale_loc           = glGetUniformLocation(surfel_shader_.program, "proj_scale");
	surfel_shader_.model_view_matrix_loc    = glGetUniformLocation(surfel_shader_.program, "model_view_matrix");
	surfel_shader_.viewport_loc             = glGetUniformLocation(surfel_shader_.program, "viewport");

	//glUseProgram(surfel_prov_shader_.program_);
	//surfel_shader_.mvp_matrix_location = glGetUniformLocation(surfel_shader_.program_, "mvp_matrix");
	//surfel_shader_.model_matrix_location = glGetUniformLocation(surfel_shader_.program_, "model_matrix");
	//surfel_shader_.view_matrix_location = glGetUniformLocation(surfel_shader_.program_, "view_matrix");
	//surfel_shader_.projection_matrix_location = glGetUniformLocation(surfel_shader_.program_, "projection_matrix");
	//surfel_shader_.model_view_matrix_location = glGetUniformLocation(surfel_shader_.program_, "model_view_matrix");
	//surfel_shader_.inverse_mv_matrix_location = glGetUniformLocation(surfel_shader_.program_, "inv_mv_matrix");
	//surfel_shader_.model_to_screen_matrix_location = glGetUniformLocation(surfel_shader_.program_, "model_to_screen_matrix");
	//surfel_shader_.scale_radius_location = glGetUniformLocation(surfel_shader_.program_, "scale_radius");
	//surfel_shader_.max_radius_location = glGetUniformLocation(surfel_shader_.program_, "max_radius");
	//surfel_shader_.window_size_location = glGetUniformLocation(surfel_shader_.program_, "win_size");
	//surfel_shader_.near_plane_location = glGetUniformLocation(surfel_shader_.program_, "near_plane");
	//surfel_shader_.far_plane_location = glGetUniformLocation(surfel_shader_.program_, "far_plane");
	//surfel_shader_.surfel_size_factor_location = glGetUniformLocation(surfel_shader_.program_, "surfel_size_factor");
	//surfel_shader_.show_normals_location = glGetUniformLocation(surfel_shader_.program_, "show_normals");
	//surfel_shader_.show_accuracy_location = glGetUniformLocation(surfel_shader_.program_, "show_accuracy");
	//surfel_shader_.show_output_sensitivity_location = glGetUniformLocation(surfel_shader_.program_, "show_output_sensitivity");
	//surfel_shader_.channel_location = glGetUniformLocation(surfel_shader_.program_, "channel");
	//surfel_shader_.heatmap_enabled_location = glGetUniformLocation(surfel_shader_.program_, "heatmap");
	//surfel_shader_.heatmap_min_location = glGetUniformLocation(surfel_shader_.program_, "heatmap_min");
	//surfel_shader_.heatmap_max_location = glGetUniformLocation(surfel_shader_.program_, "heatmap_max");
	//surfel_shader_.heatmap_min_color_location = glGetUniformLocation(surfel_shader_.program_, "heatmap_min_color");
	//surfel_shader_.heatmap_max_color_location = glGetUniformLocation(surfel_shader_.program_, "heatmap_max_color");
	//surfel_shader_.use_material_color_location = glGetUniformLocation(surfel_shader_.program_, "use_material_color");
	//surfel_shader_.material_diffuse_location = glGetUniformLocation(surfel_shader_.program_, "material_diffuse");
	//surfel_shader_.material_specular_location = glGetUniformLocation(surfel_shader_.program_, "material_specular");
	//surfel_shader_.ambient_light_color_location = glGetUniformLocation(surfel_shader_.program_, "ambient_light_color");
	//surfel_shader_.point_light_color_location = glGetUniformLocation(surfel_shader_.program_, "point_light_color");
	//surfel_shader_.eye_location = glGetUniformLocation(surfel_shader_.program_, "eye");
	//surfel_shader_.face_eye_location = glGetUniformLocation(surfel_shader_.program_, "face_eye");

	//glUseProgram(point_prov_shader_.program_);
	//point_prov_shader_.mvp_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "mvp_matrix");
	//point_prov_shader_.model_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "model_matrix");
	//point_prov_shader_.view_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "view_matrix");
	//point_prov_shader_.projection_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "projection_matrix");
	//point_prov_shader_.model_view_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "model_view_matrix");
	//point_prov_shader_.inverse_mv_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "inv_mv_matrix");
	//point_prov_shader_.model_to_screen_matrix_location = glGetUniformLocation(point_prov_shader_.program_, "model_to_screen_matrix");
	//point_prov_shader_.viewport_location = glGetUniformLocation(point_prov_shader_.program_, "viewport");
	//point_prov_shader_.scale_radius_location = glGetUniformLocation(point_prov_shader_.program_, "scale_radius");
	//point_prov_shader_.max_radius_location = glGetUniformLocation(point_prov_shader_.program_, "max_radius");
	//point_prov_shader_.point_size_factor_location = glGetUniformLocation(point_prov_shader_.program_, "point_size_factor");
	//point_prov_shader_.near_plane_location = glGetUniformLocation(point_prov_shader_.program_, "near_plane");
	//point_prov_shader_.far_plane_location = glGetUniformLocation(point_prov_shader_.program_, "far_plane");
	//point_prov_shader_.height_divided_by_top_minus_bottom_location = glGetUniformLocation(point_prov_shader_.program_, "height_divided_by_top_minus_bottom");

	glUseProgram(line_shader_.program_);
	line_shader_.mvp_matrix_location = glGetUniformLocation(line_shader_.program_, "mvp_matrix");
	line_shader_.in_color_location = glGetUniformLocation(line_shader_.program_, "in_color");

	glUseProgram(0);
}

unsigned int LamurePointCloudPlugin::create_shader(const std::string& vertexShader, const std::string& fragmentShader, uint8_t ctx_id)
{
	osg::GLExtensions* gl_api = new osg::GLExtensions(ctx_id);
	unsigned int program = gl_api->glCreateProgram();
	unsigned int vs = compile_shader(GL_VERTEX_SHADER, vertexShader, ctx_id);
	unsigned int fs = compile_shader(GL_FRAGMENT_SHADER, fragmentShader, ctx_id);
	gl_api->glAttachShader(program, vs);
	gl_api->glAttachShader(program, fs);
	gl_api->glLinkProgram(program);
	gl_api->glValidateProgram(program);
	gl_api->glDeleteProgram(vs);
	gl_api->glDeleteProgram(fs);
	return 1;
}

unsigned int LamurePointCloudPlugin::compile_shader(unsigned int type, const std::string& source, uint8_t ctx_id)
{
	osg::GLExtensions* gl_api = new osg::GLExtensions(ctx_id);
	unsigned int id = gl_api->glCreateShader(type);
	const char* src = source.c_str();
	gl_api->glShaderSource(id, 1, &src, nullptr);
	gl_api->glCompileShader(id);
	int result;
	gl_api->glGetShaderiv(id, GL_COMPILE_STATUS, &result);
	if (result == false)
	{
		int length;
		gl_api->glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);
		char* message = (char*)alloca(length * sizeof(char));
		gl_api->glGetShaderInfoLog(id, length, &length, message);
		std::cout << "Failed to compile " << (type == GL_VERTEX_SHADER ? "vertex" : "fragment") << " shader!" << std::endl;
		std::cout << message << std::endl;
		gl_api->glDeleteShader(id);
		return 0;
	};
	return id;
}

GLuint LamurePointCloudPlugin::compile_and_link_shaders(std::string vs_source, std::string fs_source)
{
	GLuint program = glCreateProgram();
	GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source, 0);
	GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source, 0);
	glAttachShader(program, vs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);
	glDeleteShader(vs);
	glDeleteShader(fs);
	return program;
}

GLuint LamurePointCloudPlugin::compile_and_link_shaders(std::string vs_source, std::string gs_source, std::string fs_source)
{
	GLuint program = glCreateProgram();
	GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_source, 0);
	GLuint gs = compile_shader(GL_GEOMETRY_SHADER, gs_source, 0);
	GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_source, 0);
	glAttachShader(program, vs);
	glAttachShader(program, gs);
	glAttachShader(program, fs);
	glLinkProgram(program);
	glValidateProgram(program);
	glDeleteShader(vs);
	glDeleteShader(gs);
	glDeleteShader(fs);
	return program;
}

bool LamurePointCloudPlugin::read_shader(std::string const& path_string, std::string& shader_string, bool keep_optional_shader_code = false) 
{
	if (!boost::filesystem::exists(path_string)) {
		std::cout << "WARNING: File " << path_string << "does not exist." << std::endl;
		return false;
	}
	std::ifstream shader_source(path_string, std::ios::in);
	std::string line_buffer;
	std::string include_prefix("INCLUDE");
	std::string optional_begin_prefix("OPTIONAL_BEGIN");
	std::string optional_end_prefix("OPTIONAL_END");
	std::size_t slash_position = path_string.find_last_of("/\\");
	std::string const base_path = path_string.substr(0, slash_position + 1);

	bool disregard_code = false;
	while (std::getline(shader_source, line_buffer)) {
		line_buffer = strip_whitespace(line_buffer);
		if (parse_prefix(line_buffer, include_prefix)) {
			if (!disregard_code || keep_optional_shader_code) {
				std::string filename_string = line_buffer;
				read_shader(base_path + filename_string, shader_string);
			}
		}
		else if (parse_prefix(line_buffer, optional_begin_prefix)) {
			disregard_code = true;
		}
		else if (parse_prefix(line_buffer, optional_end_prefix)) {
			disregard_code = false;
		}
		else {
			if ((!disregard_code) || keep_optional_shader_code) {
				shader_string += line_buffer + "\n";
			}
		}
	}
	return true;
}

void LamurePointCloudPlugin::init_lamure_shader()
{
	if (plugin->notify_button->state() == 1) { std::cout << "[Notify] init_lamure_shader()" << std::endl; }
	try
	{
		if (   !read_shader(shader_root_path + "/vis/vis_point.glslv", vis_point_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_point.glslf", vis_point_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_point_prov.glslv", vis_point_prov_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_point_prov.glslf", vis_point_prov_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_surfel.glslv", vis_surfel_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_surfel.glslg", vis_surfel_gs_source)
			|| !read_shader(shader_root_path + "/vis/vis_surfel.glslf", vis_surfel_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_surfel_prov.glslv", vis_surfel_prov_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_surfel_prov.glslf", vis_surfel_prov_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_line_bb.glslv", vis_line_bb_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_line_bb.glslf", vis_line_bb_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_quad.glslv", vis_quad_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_quad.glslf", vis_quad_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_line.glslv", vis_line_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_line.glslf", vis_line_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_triangle.glslv", vis_triangle_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_triangle.glslf", vis_triangle_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_plane.glslv", vis_plane_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_plane.glslf", vis_plane_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_text.glslv", vis_text_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_text.glslf", vis_text_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_box.glslv", vis_box_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_box.glslg", vis_box_gs_source)
			|| !read_shader(shader_root_path + "/vis/vis_box.glslf", vis_box_fs_source)
			|| !read_shader(shader_root_path + "/vt/virtual_texturing.glslv", vis_vt_vs_source)
			|| !read_shader(shader_root_path + "/vt/virtual_texturing_hierarchical.glslf", vis_vt_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz.glslv", vis_xyz_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz.glslg", vis_xyz_gs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz.glslf", vis_xyz_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass1.glslv", vis_xyz_pass1_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass1.glslg", vis_xyz_pass1_gs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass1.glslf", vis_xyz_pass1_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass2.glslv", vis_xyz_pass2_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass2.glslg", vis_xyz_pass2_gs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass2.glslf", vis_xyz_pass2_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass3.glslv", vis_xyz_pass3_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass3.glslf", vis_xyz_pass3_fs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_qz.glslv", vis_xyz_qz_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_qz_pass1.glslv", vis_xyz_qz_pass1_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_qz_pass2.glslv", vis_xyz_qz_pass2_vs_source)
			|| !read_shader(shader_root_path + "/vis/vis_xyz.glslv", vis_xyz_vs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz.glslg", vis_xyz_gs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz.glslf", vis_xyz_fs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass2.glslv", vis_xyz_pass2_vs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass2.glslg", vis_xyz_pass2_gs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass2.glslf", vis_xyz_pass2_fs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass3.glslv", vis_xyz_pass3_vs_lighting_source, true)
			|| !read_shader(shader_root_path + "/vis/vis_xyz_pass3.glslf", vis_xyz_pass3_fs_lighting_source, true)
			) {
			std::cout << "error reading shader files" << std::endl; exit(1);
		}

		point_shader_.program = compile_and_link_shaders(vis_point_vs_source, vis_point_fs_source);
		surfel_shader_.program = compile_and_link_shaders(vis_surfel_vs_source, vis_surfel_gs_source, vis_surfel_fs_source);
		line_shader_.program_ = compile_and_link_shaders(vis_line_bb_vs_source, vis_line_bb_fs_source);
		//std::cout << "[Notify] point_shader_" << std::endl;
		//point_prov_shader_.program_ = compile_and_link_shaders(vis_point_prov_vs_source, vis_point_prov_fs_source);
		//std::cout << "[Notify] line_shader_" << std::endl;

		//vis_point_shader_ = device_->create_program(
		//	boost::assign::list_of
		//	(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_point_shader_vs_source))
		//	(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_point_shader_fs_source)));
		//if (!vis_point_shader_) { std::cout << "error creating shader vis_point_shader_ program" << std::endl; exit(1); }

		vis_quad_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_quad_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_quad_fs_source)));
		if (!vis_quad_shader_) { std::cout << "error creating shader vis_quad_shader_ program" << std::endl; exit(1); }

		vis_plane_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_plane_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_plane_fs_source)));
		if (!vis_quad_shader_) { std::cout << "error creating shader vis_plane_shader_ program" << std::endl; exit(1); }

		vis_line_bb_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_line_bb_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_line_bb_fs_source)));
		if (!vis_line_bb_shader_) {
			std::cout << "error creating shader vis_line_bb_shader_ program" << std::endl; exit(1); }

		vis_text_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_text_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_text_fs_source)));
		if (!vis_text_shader_) { std::cout << "error creating shader vis_text_shader_ program" << std::endl; exit(1); }

		vis_line_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_line_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_line_fs_source)));
		if (!vis_line_shader_) { std::cout << "error creating shader vis_line_shader_ program" << std::endl; exit(1); }

		vis_triangle_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_triangle_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_triangle_fs_source)));
		if (!vis_triangle_shader_) { std::cout << "error creating shader vis_triangle_shader_ program" << std::endl; std::exit(1); }

		vis_vt_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_vt_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_vt_fs_source)));
		if (!vis_vt_shader_) { std::cout << "error creating shader vis_vt_shader_program" << std::endl; std::exit(1); }

		vis_xyz_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_vs_source))
			(device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, vis_xyz_gs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_fs_source)));
		if (!vis_xyz_shader_) { std::cout << "error creating shader vis_xyz_shader_ program" << std::endl; exit(1); }

		vis_xyz_pass1_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_pass1_vs_source))
			(device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, vis_xyz_pass1_gs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_pass1_fs_source)));
		if (!vis_xyz_pass1_shader_) { std::cout << "error creating vis_xyz_pass1_shader_ program" << std::endl; exit(1); }

		vis_xyz_pass2_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_pass2_vs_source))
			(device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, vis_xyz_pass2_gs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_pass2_fs_source)));
		if (!vis_xyz_pass2_shader_) { std::cout << "error creating vis_xyz_pass2_shader_ program" << std::endl; exit(1); }

		vis_xyz_pass3_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_pass3_vs_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_pass3_fs_source)));
		if (!vis_xyz_pass3_shader_) { std::cout << "error creating vis_xyz_pass3_shader_ program" << std::endl; exit(1); }

		vis_xyz_lighting_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_vs_lighting_source))
			(device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, vis_xyz_gs_lighting_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_fs_lighting_source)));
		if (!vis_xyz_lighting_shader_) { std::cout << "error creating vis_xyz_lighting_shader_ program" << std::endl; exit(1); }

		vis_xyz_pass2_lighting_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_pass2_vs_lighting_source))
			(device_->create_shader(scm::gl::STAGE_GEOMETRY_SHADER, vis_xyz_pass2_gs_lighting_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_pass2_fs_lighting_source)));
		if (!vis_xyz_pass2_lighting_shader_) { std::cout << "error creating vis_xyz_pass2_lighting_shader_ program" << std::endl; exit(1); }

		vis_xyz_pass3_lighting_shader_ = device_->create_program(
			boost::assign::list_of
			(device_->create_shader(scm::gl::STAGE_VERTEX_SHADER, vis_xyz_pass3_vs_lighting_source))
			(device_->create_shader(scm::gl::STAGE_FRAGMENT_SHADER, vis_xyz_pass3_fs_lighting_source)));
		if (!vis_xyz_pass3_lighting_shader_) { std::cout << "error creating vis_xyz_pass3_lighting_shader_ program" << std::endl; exit(1); }
	}
	catch (std::exception& e) { std::cout << e.what() << std::endl; }
}


void LamurePointCloudPlugin::debug_print_settings() const {
	using std::cout;
	using std::endl;
	cout << "=== Settings Dump ===" << endl;
	cout << "frame_div: " << settings_.frame_div_ << endl;
	cout << "vram: " << settings_.vram_ << endl;
	cout << "ram: " << settings_.ram_ << endl;
	cout << "upload: " << settings_.upload_ << endl;
	cout << "provenance: " << settings_.provenance_ << endl;
	cout << "surfel_shader: " << settings_.surfel_shader_ << endl;
	cout << "create_aux_resources: " << settings_.create_aux_resources_ << endl;
	cout << "gamma_correction: " << settings_.gamma_correction_ << endl;
	cout << "max_brush_size: " << settings_.max_brush_size_ << endl;
	cout << "lod_update: " << settings_.lod_update_ << endl;
	cout << "use_pvs: " << settings_.use_pvs_ << endl;
	cout << "pvs_culling: " << settings_.pvs_culling_ << endl;
	cout << "point_size_factor: " << settings_.point_size_factor_ << endl;
	cout << "surfel_size_factor: " << settings_.surfel_size_factor_ << endl;
	cout << "face_eye: " << settings_.face_eye_ << endl;
	cout << "aux_point_size: " << settings_.aux_point_size_ << endl;
	cout << "aux_point_distance: " << settings_.aux_point_distance_ << endl;
	cout << "aux_point_scale: " << settings_.aux_point_scale_ << endl;
	cout << "aux_focal_length: " << settings_.aux_focal_length_ << endl;
	cout << "vis: " << settings_.vis_ << endl;
	cout << "show_normals: " << settings_.show_normals_ << endl;
	cout << "show_accuracy: " << settings_.show_accuracy_ << endl;
	cout << "show_radius_deviation: " << settings_.show_radius_deviation_ << endl;
	cout << "show_output_sensitivity: " << settings_.show_output_sensitivity_ << endl;
	cout << "show_sparse: " << settings_.show_sparse_ << endl;
	cout << "show_views: " << settings_.show_views_ << endl;
	cout << "show_photos: " << settings_.show_photos_ << endl;
	cout << "show_octrees: " << settings_.show_octrees_ << endl;
	cout << "show_bvhs: " << settings_.show_bvhs_ << endl;
	cout << "show_pvs: " << settings_.show_pvs_ << endl;
	cout << "channel: " << settings_.channel_ << endl;
	cout << "enable_lighting: " << settings_.enable_lighting_ << endl;
	cout << "use_material_color: " << settings_.use_material_color_ << endl;
	cout << std::fixed << std::setprecision(3);
	cout << "material_diffuse: (" << settings_.material_diffuse_.x << ", " << settings_.material_diffuse_.y << ", " << settings_.material_diffuse_.z << ")" << endl;
	cout << "material_specular: (" << settings_.material_specular_.x << ", " << settings_.material_specular_.y << ", " << settings_.material_specular_.z << ", " << settings_.material_specular_.w << ")" << endl;
	cout << "ambient_light_color: (" << settings_.ambient_light_color_.r << ", " << settings_.ambient_light_color_.g << ", " << settings_.ambient_light_color_.b << ")" << endl;
	cout << "point_light_color: (" << settings_.point_light_color_.r << ", " << settings_.point_light_color_.g << ", " << settings_.point_light_color_.b << ", " << settings_.point_light_color_.w << ")" << endl;
	cout << "heatmap: " << settings_.heatmap_ << endl;
	cout << "heatmap_min: " << settings_.heatmap_min_ << endl;
	cout << "heatmap_max: " << settings_.heatmap_max_ << endl;
	cout << "background_color: (" << settings_.background_color_.x << ", " << settings_.background_color_.y << ", " << settings_.background_color_.z << ")" << endl;
	cout << "heatmap_color_min: ("<< settings_.heatmap_color_min_.x << ", " << settings_.heatmap_color_min_.y << ", " << settings_.heatmap_color_min_.z << ")" << endl;
	cout << "heatmap_color_max: ("<< settings_.heatmap_color_max_.x << ", " << settings_.heatmap_color_max_.y << ", " << settings_.heatmap_color_max_.z << ")" << endl;
	cout << std::defaultfloat;
	cout << "atlas_file: " << settings_.atlas_file_ << endl;
	cout << "json: " << settings_.json_ << endl;
	cout << "pvs: " << settings_.pvs_ << endl;
	cout << "background_image: " << settings_.background_image_ << endl;
	cout << "max_radius: " << settings_.max_radius_ << endl;
	cout << "scale_radius: " << settings_.scale_radius_ << endl;
	cout << "bvh_color: (" << settings_.bvh_color_[0] << ", " << settings_.bvh_color_[1] << ", " << settings_.bvh_color_[2] << ", " << settings_.bvh_color_[3] << ")" << endl;
	cout << "frustum_color: (" << settings_.frustum_color_[0] << ", " << settings_.frustum_color_[1] << ", " << settings_.frustum_color_[2] << ", " << settings_.frustum_color_[3] << ")" << endl;
	cout << "models (" << settings_.models_.size() << "):" << endl;
	for (size_t i = 0; i < settings_.models_.size(); ++i) {
		cout << "  [" << i << "] " << settings_.models_[i] << std::endl; 
	}
	cout << "transforms (" << settings_.transforms_.size() << "):" << endl;
	for (auto const& [id, mat] : settings_.transforms_) {
		cout << "  id=" << id << ":";
		cout << mat << endl;
	}
	cout << "aux data (" << settings_.aux_.size() << "):" << endl;
	for (auto const& [id, aux_path] : settings_.aux_) {
		cout << "  id=" << id << " => " << aux_path << endl;
	}
	cout << "=== End Settings Dump ===" << endl;
}


void LamurePointCloudPlugin::create_framebuffers()
{
	if (plugin->notify_button->state()) { std::cout << "[Notify] create_framebuffers() " << std::endl; }
	fbo_.reset();
	fbo_color_buffer_.reset();
	fbo_depth_buffer_.reset();
	pass1_fbo_.reset();
	pass1_depth_buffer_.reset();
	pass2_fbo_.reset();
	pass2_color_buffer_.reset();
	pass2_normal_buffer_.reset();
	pass2_view_space_pos_buffer_.reset();

	fbo_ = device_->create_frame_buffer();
	fbo_color_buffer_ = device_->create_texture_2d(scm::math::vec2ui(traits->width, traits->height), scm::gl::FORMAT_RGBA_32F, 1, 1, 1);
	fbo_depth_buffer_ = device_->create_texture_2d(scm::math::vec2ui(traits->width, traits->height), scm::gl::FORMAT_D24, 1, 1, 1);
	fbo_->attach_color_buffer(0, fbo_color_buffer_);
	fbo_->attach_depth_stencil_buffer(fbo_depth_buffer_);
	pass1_fbo_ = device_->create_frame_buffer();
	pass1_depth_buffer_ = device_->create_texture_2d(scm::math::vec2ui(traits->width, render_height_), scm::gl::FORMAT_D24, 1, 1, 1);
	pass1_fbo_->attach_depth_stencil_buffer(pass1_depth_buffer_);
	//pass2_fbo_ = device_->create_frame_buffer();
	//pass2_color_buffer_ = device_->create_texture_2d(scm::math::vec2ui(traits->width, render_height_), scm::gl::FORMAT_RGBA_32F, 1, 1, 1);
	//pass2_fbo_->attach_color_buffer(0, pass2_color_buffer_);
	//pass2_fbo_->attach_depth_stencil_buffer(pass1_depth_buffer_);
	//pass2_normal_buffer_ = device_->create_texture_2d(scm::math::vec2ui(traits->width, render_height_), scm::gl::FORMAT_RGB_32F, 1, 1, 1);
	//pass2_fbo_->attach_color_buffer(1, pass2_normal_buffer_);
	//pass2_view_space_pos_buffer_ = device_->create_texture_2d(scm::math::vec2ui(traits->width, render_height_), scm::gl::FORMAT_RGB_32F, 1, 1, 1);
	//pass2_fbo_->attach_color_buffer(2, pass2_view_space_pos_buffer_);
}


void LamurePointCloudPlugin::init_render_states() 
{
	if (plugin->notify_button->state()) { std::cout << "[Notify] init_render_states() " << std::endl; }
	color_blending_state_ = device_->create_blend_state(true, scm::gl::FUNC_ONE, scm::gl::FUNC_ONE, scm::gl::FUNC_ONE, scm::gl::FUNC_ONE, scm::gl::EQ_FUNC_ADD, scm::gl::EQ_FUNC_ADD);
	color_no_blending_state_ = device_->create_blend_state(false);
	depth_state_less_ = device_->create_depth_stencil_state(true, true, scm::gl::COMPARISON_LESS);
	auto no_depth_test_descriptor = depth_state_less_->descriptor();
	no_depth_test_descriptor._depth_test = false;
	depth_state_disable_ = device_->create_depth_stencil_state(no_depth_test_descriptor);
	depth_state_without_writing_ = device_->create_depth_stencil_state(true, false, scm::gl::COMPARISON_LESS_EQUAL);
	no_backface_culling_rasterizer_state_ = device_->create_rasterizer_state(scm::gl::FILL_SOLID, scm::gl::CULL_NONE, scm::gl::ORIENT_CCW, false, false, 0.0, false, false);
	filter_linear_ = device_->create_sampler_state(scm::gl::FILTER_ANISOTROPIC, scm::gl::WRAP_CLAMP_TO_EDGE, 16u);
	filter_nearest_ = device_->create_sampler_state(scm::gl::FILTER_MIN_MAG_LINEAR, scm::gl::WRAP_CLAMP_TO_EDGE);
	//vt_filter_linear_ = device_->create_sampler_state(scm::gl::FILTER_MIN_MAG_LINEAR, scm::gl::WRAP_CLAMP_TO_EDGE);
	//vt_filter_nearest_ = device_->create_sampler_state(scm::gl::FILTER_MIN_MAG_NEAREST, scm::gl::WRAP_CLAMP_TO_EDGE);
}


void LamurePointCloudPlugin::set_lamure_uniforms(scm::gl::program_ptr shader) 
{
	shader->uniform("win_size", scm::math::vec2f(traits->width, traits->height));
	shader->uniform("near_plane", coco->nearClip());
	shader->uniform("far_plane", coco->farClip());
	shader->uniform("point_size_factor", settings_.point_size_factor_);
	shader->uniform("show_normals", (bool)settings_.show_normals_);
	shader->uniform("show_accuracy", (bool)settings_.show_accuracy_);
	shader->uniform("show_radius_deviation", (bool)settings_.show_radius_deviation_);
	shader->uniform("show_output_sensitivity", (bool)settings_.show_output_sensitivity_);
	shader->uniform("channel", settings_.channel_);
	shader->uniform("heatmap", (bool)settings_.heatmap_);
	shader->uniform("face_eye", false);
	shader->uniform("scale_radius", settings_.scale_radius_);
	shader->uniform("max_radius", settings_.max_radius_);
	shader->uniform("heatmap_min", settings_.heatmap_min_);
	shader->uniform("heatmap_max", settings_.heatmap_max_);
	shader->uniform("heatmap_min_color", settings_.heatmap_color_min_);
	shader->uniform("heatmap_max_color", settings_.heatmap_color_max_);
	if (settings_.enable_lighting_) {
		shader->uniform("use_material_color", settings_.use_material_color_);
		shader->uniform("material_diffuse", settings_.material_diffuse_);
		shader->uniform("material_specular", settings_.material_specular_);
		shader->uniform("ambient_light_color", settings_.ambient_light_color_);
		shader->uniform("point_light_color", settings_.point_light_color_);
	}
}


std::string const LamurePointCloudPlugin::strip_whitespace(std::string const& in_string) {
	return boost::regex_replace(in_string, boost::regex("^ +| +$|( ) +"), "$1");
}


string LamurePointCloudPlugin::getConfigEntry(string scope) {
	std::cout << "getConfigEntry(scope): ";
	covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries(scope);
	for (const auto& entry : entries)
	{ return entry.second; }
	return "";
}


string LamurePointCloudPlugin::getConfigEntry(string scope, string name) {
	std::cout << "getConfigEntry(scope, name): ";
	covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries(scope);
	for (const auto& entry : entries) {
		if (name == entry.first)
		{ return entry.second; }
	}
	return "";
}


size_t LamurePointCloudPlugin::query_video_memory_in_mb() {
	int size_in_kb;
	glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &size_in_kb);
	return size_t(size_in_kb) / 1024;
}


scm::math::mat4d LamurePointCloudPlugin::load_matrix(const std::string& filename) {
	std::ifstream file(filename.c_str());
	if (!file.is_open()) {
		std::cerr << "Unable to open transformation file: \""
			<< filename << "\"\n";
		return scm::math::mat4d::identity();
	}
	scm::math::mat4d mat = scm::math::mat4d::identity();
	std::string matrix_values_string;
	std::getline(file, matrix_values_string);
	std::stringstream sstr(matrix_values_string);
	for (int i = 0; i < 16; ++i)
		sstr >> std::setprecision(16) >> mat[i];
	file.close();
	return scm::math::transpose(mat);
}


bool LamurePointCloudPlugin::parse_prefix(std::string& in_string, std::string const& prefix) {
	uint32_t num_prefix_characters = prefix.size();
	bool prefix_found
		= (!(in_string.size() < num_prefix_characters)
			&& strncmp(in_string.c_str(), prefix.c_str(), num_prefix_characters) == 0);
	if (prefix_found) {
		in_string = in_string.substr(num_prefix_characters);
		in_string = strip_whitespace(in_string);
	}
	return prefix_found;
}


void LamurePointCloudPlugin::readMenuConfigData(const char* menu, vector<ImageFileEntry>& menulist, ui::Group* subMenu)
{
	covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries(menu);
	for (const auto& entry : entries)
	{
		ui::Button* temp = new ui::Button(subMenu, entry.second);
		temp->setCallback([this, entry](bool state)
			{
				if (state)
					std::printf("createGeodes(planetTrans, entry.second)");
			});
		menulist.push_back(ImageFileEntry(entry.first.c_str(), entry.second.c_str(), (ui::Element*)temp));
	}
}
