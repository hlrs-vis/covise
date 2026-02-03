#include "LamureUtil.h"
#include <iostream>
#include <scm/gl_core/primitives/box.h>
#include <config/CoviseConfig.h>

//boost
#include <boost/regex.hpp>

#include <string> // For std::string
#include <fstream> // For std::ifstream
#include <sstream> // For std::stringstream
#include <iomanip> // For std::setprecision
#include <cmath> // For std::pow, std::round
#include <algorithm> // For std::max
#include <cstring> // For std::strcmp, std::strstr, std::strlen
#include <GL/glu.h> // For gluErrorString
#include <unordered_map>

namespace {
#ifndef GL_DEVICE_UUID_EXT
#define GL_DEVICE_UUID_EXT 0x9597
#define GL_DRIVER_UUID_EXT 0x9598
#define GL_UUID_SIZE_EXT 16
#endif
typedef void (*GetUnsignedBytevExtProc)(GLenum pname, GLubyte* data);
#ifndef GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX
#define GL_GPU_MEMORY_INFO_DEDICATED_VIDMEM_NVX 0x9047
#define GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX 0x9048
#define GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX 0x9049
#define GL_GPU_MEMORY_INFO_EVICTION_COUNT_NVX 0x904A
#define GL_GPU_MEMORY_INFO_EVICTED_MEMORY_NVX 0x904B
#endif
#ifndef GL_TEXTURE_FREE_MEMORY_ATI
#define GL_TEXTURE_FREE_MEMORY_ATI 0x87FC
#define GL_VBO_FREE_MEMORY_ATI 0x87FB
#define GL_RENDERBUFFER_FREE_MEMORY_ATI 0x87FD
#endif

#ifdef _WIN32
#ifndef WGL_GPU_VENDOR_AMD
#define WGL_GPU_VENDOR_AMD 0x1F00
#define WGL_GPU_RENDERER_STRING_AMD 0x1F01
#define WGL_GPU_OPENGL_VERSION_STRING_AMD 0x1F02
#define WGL_GPU_RAM_AMD 0x21A3
#endif
typedef UINT(WINAPI* PFNWGLGETCONTEXTGPUIDAMDPROC)(HGLRC hglrc);
typedef UINT(WINAPI* PFNWGLGETGPUIDSAMDPROC)(UINT maxCount, UINT* ids);
typedef int (WINAPI* PFNWGLGETGPUINFOAMDPROC)(UINT id, int property, GLenum dataType, UINT size, void* data);
#endif

bool hasExtension(const char* ext)
{
    if (!ext || !*ext) return false;
    if (GLEW_VERSION_3_0) {
        GLint count = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &count);
        for (GLint i = 0; i < count; ++i) {
            const char* name = reinterpret_cast<const char*>(glGetStringi(GL_EXTENSIONS, i));
            if (name && std::strcmp(name, ext) == 0) {
                return true;
            }
        }
        return false;
    }
    const char* ext_list = reinterpret_cast<const char*>(glGetString(GL_EXTENSIONS));
    if (!ext_list) return false;
    const char* pos = std::strstr(ext_list, ext);
    if (!pos) return false;
    const char* end = pos + std::strlen(ext);
    const bool start_ok = (pos == ext_list) || (*(pos - 1) == ' ');
    const bool end_ok = (*end == ' ' || *end == '\0');
    return start_ok && end_ok;
}

std::string bytesToHex(const GLubyte* data, std::size_t len)
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (std::size_t i = 0; i < len; ++i) {
        oss << std::setw(2) << static_cast<unsigned int>(data[i]);
    }
    return oss.str();
}

struct UuidInfo {
    std::string device;
    std::string driver;
};

UuidInfo queryUuidInfo()
{
    UuidInfo info;
    if (!hasExtension("GL_EXT_memory_object") &&
        !hasExtension("GL_EXT_memory_object_win32") &&
        !hasExtension("GL_EXT_memory_object_fd")) {
        return info;
    }
    GetUnsignedBytevExtProc getUnsignedBytevExt = nullptr;
#ifdef _WIN32
    getUnsignedBytevExt = reinterpret_cast<GetUnsignedBytevExtProc>(wglGetProcAddress("glGetUnsignedBytevEXT"));
#else
    getUnsignedBytevExt = reinterpret_cast<GetUnsignedBytevExtProc>(
        glXGetProcAddress(reinterpret_cast<const GLubyte*>("glGetUnsignedBytevEXT")));
#endif
    if (!getUnsignedBytevExt) {
        return info;
    }
    GLubyte device_uuid[GL_UUID_SIZE_EXT] = {};
    GLubyte driver_uuid[GL_UUID_SIZE_EXT] = {};
    getUnsignedBytevExt(GL_DEVICE_UUID_EXT, device_uuid);
    getUnsignedBytevExt(GL_DRIVER_UUID_EXT, driver_uuid);
    info.device = bytesToHex(device_uuid, GL_UUID_SIZE_EXT);
    info.driver = bytesToHex(driver_uuid, GL_UUID_SIZE_EXT);
    return info;
}

#ifdef _WIN32
void appendAmdGpuAssociationInfo(std::ostream& os)
{
    auto get_ctx_gpu_id = reinterpret_cast<PFNWGLGETCONTEXTGPUIDAMDPROC>(wglGetProcAddress("wglGetContextGPUIDAMD"));
    auto get_gpu_info = reinterpret_cast<PFNWGLGETGPUINFOAMDPROC>(wglGetProcAddress("wglGetGPUInfoAMD"));
    if (!get_ctx_gpu_id || !get_gpu_info) {
        return;
    }
    HGLRC hglrc = wglGetCurrentContext();
    if (!hglrc) {
        return;
    }
    UINT gpu_id = get_ctx_gpu_id(hglrc);
    if (gpu_id == 0) {
        return;
    }
    char renderer[256] = {};
    char vendor[256] = {};
    GLuint ram_mb = 0;
    get_gpu_info(gpu_id, WGL_GPU_RENDERER_STRING_AMD, GL_UNSIGNED_BYTE, sizeof(renderer), renderer);
    get_gpu_info(gpu_id, WGL_GPU_VENDOR_AMD, GL_UNSIGNED_BYTE, sizeof(vendor), vendor);
    get_gpu_info(gpu_id, WGL_GPU_RAM_AMD, GL_UNSIGNED_INT, sizeof(ram_mb), &ram_mb);
    os << " amd_gpu_id=" << gpu_id
       << " amd_vendor=" << (vendor[0] ? vendor : "unknown")
       << " amd_renderer=" << (renderer[0] ? renderer : "unknown")
       << " amd_vram_mb=" << ram_mb;
}
#endif

void appendVendorSpecificInfo(std::ostream& os, const char* vendor)
{
    if (!vendor) {
        return;
    }
    if (std::strstr(vendor, "NVIDIA")) {
        if (hasExtension("GL_NVX_gpu_memory_info")) {
            GLint total_kb = 0;
            GLint avail_kb = 0;
            glGetIntegerv(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX, &total_kb);
            glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX, &avail_kb);
            os << " nvx_total_mb=" << (total_kb / 1024)
               << " nvx_avail_mb=" << (avail_kb / 1024);
        }
        return;
    }
    if (std::strstr(vendor, "AMD") || std::strstr(vendor, "ATI")) {
#ifdef _WIN32
        appendAmdGpuAssociationInfo(os);
#endif
        if (hasExtension("GL_ATI_meminfo")) {
            GLint tex_mem[4] = {0,0,0,0};
            glGetIntegerv(GL_TEXTURE_FREE_MEMORY_ATI, tex_mem);
            os << " ati_tex_free_mb=" << (tex_mem[0] / 1024);
        }
        return;
    }
}
} // namespace

namespace LamureUtil {

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
scm::math::vec3f vecConv3F(const osg::Vec3f& v) {
	scm::math::vec3f vec_scm = scm::math::vec3f(v[0], v[1], v[2]);
	return vec_scm;
}
scm::math::vec3d vecConv3D(const osg::Vec3d& v) {
	scm::math::vec3d vec_scm = scm::math::vec3d(v[0], v[1], v[2]);
	return vec_scm;
}
scm::math::vec3f vecConv3F(const osg::Vec4f& v) {
	scm::math::vec4f vec_scm = scm::math::vec4f(v[0], v[1], v[2]);
	return vec_scm;
}
scm::math::vec3d vecConv3D(const osg::Vec4d& v) {
	scm::math::vec4d vec_scm = scm::math::vec4d(v[0], v[1], v[2]);
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
scm::math::vec4f vecConv4F(const osg::Vec4f& v) {
	scm::math::vec4f vec_scm = scm::math::vec4f(v[0], v[1], v[2], v[3]);
	return vec_scm;
}
scm::math::vec4d vecConv4D(const osg::Vec4d& v) {
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

scm::math::mat3f matConv4to3F(scm::math::mat4f& m) {
	const auto& d = m.data_array;
	return scm::math::mat3f(
		d[0], d[1], d[2],
		d[4], d[5], d[6],
		d[8], d[9], d[10]
	);
}
scm::math::mat3d matConv4to3D(scm::math::mat4d& m) {
	const auto& d = m.data_array;
	return scm::math::mat3d(
		d[0], d[1], d[2],
		d[4], d[5], d[6],
		d[8], d[9], d[10]
	);
}

scm::math::mat3f matConv4to3F(scm::math::mat4d& m) {
	const auto& d = scm::math::mat4f(m).data_array;
	return scm::math::mat3f(
		d[0], d[1], d[2],
		d[4], d[5], d[6],
		d[8], d[9], d[10]
	);
}

bool readIndexedMatrix(const std::string& in, osg::Matrixd& M){
	std::string s=in; for(char& c: s) if(c=='['||c==']'||c==','||c==';') c=' ';
	std::istringstream is(s); double v[16]; for(int i=0;i<16;++i){ if(!(is>>v[i])) return false; }
	M=osg::Matrixd(v[0],v[1],v[2],v[3], v[4],v[5],v[6],v[7], v[8],v[9],v[10],v[11], v[12],v[13],v[14],v[15]); return true;
}


std::string getConfigEntry(std::string scope) {
	std::cout << "getConfigEntry(scope): ";
	covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries(scope);
	for (const auto& entry : entries)
	{ return entry.second; }
	return "";
}
std::string getConfigEntry(std::string scope, std::string name) {
	std::cout << "getConfigEntry(scope, name): ";
	covise::coCoviseConfig::ScopeEntries entries = covise::coCoviseConfig::getScopeEntries(scope);
	for (const auto& entry : entries) {
		if (name == entry.first)
		{ return entry.second; }
	}
	return "";
}
void printNodePath(osg::ref_ptr<osg::Node> pointer) 
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
void printChildNodes(osg::Node* node, int depth) 
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
scm::math::mat4d loadMatrixFromFile(const std::string& filename) {
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

osg::Matrixd loadMatrix(const std::string &value) {
	std::string s = value;
	// Entferne Klammern und Semikolons durch Leerzeichen ersetzen
	s.erase(std::remove(s.begin(), s.end(), '['), s.end());
	s.erase(std::remove(s.begin(), s.end(), ']'), s.end());
	std::replace(s.begin(), s.end(), ';', ' ');

	std::istringstream iss(s);
	std::vector<double> numbers{std::istream_iterator<double>(iss), std::istream_iterator<double>()};

	osg::Matrixd mat;
	if(numbers.size() == 16) {
		// OpenSceneGraph erwartet Spalten-major (wie OpenGL)
		mat.set(numbers[0], numbers[1], numbers[2], numbers[3],
			numbers[4], numbers[5], numbers[6], numbers[7],
			numbers[8], numbers[9], numbers[10], numbers[11],
			numbers[12], numbers[13], numbers[14], numbers[15]);
	} else {
		// Fallback: Identität
		mat.makeIdentity();
		std::cerr << "LamureUtil::loadMatrix: expected 16 values, got " << numbers.size() << " -> using identity\n";
	}
	return mat;
}


std::vector<std::string> splitSemicolons(const std::string &s) {
	std::vector<std::string> out;
	std::string cur;
	std::istringstream ss(s);
	while (std::getline(ss, cur, ';')) {
		// trim leading/trailing whitespace
		auto b = cur.find_first_not_of(" \t\r\n");
		auto e = cur.find_last_not_of(" \t\r\n");
		if (b != std::string::npos) {
			out.emplace_back(cur.substr(b, e - b + 1));
		}
	}
	return out;
}

bool parsePrefix(std::string& in_string, std::string const& prefix) {
	uint32_t num_prefix_characters = prefix.size();
	bool prefix_found
		= (!(in_string.size() < num_prefix_characters)
			&& strncmp(in_string.c_str(), prefix.c_str(), num_prefix_characters) == 0);
	if (prefix_found) {
		in_string = in_string.substr(num_prefix_characters);
		in_string = stripWhitespace(in_string);
	}
	return prefix_found;
}
std::string const stripWhitespace(std::string const& in_string) {
	return boost::regex_replace(in_string, boost::regex("^ +| +$|( ) +"), "$1");
}
double roundToDecimal(double value, int decimals) {
	if (decimals < 0) {
		return value;
	}
	double factor = std::pow(10.0, decimals);
	return std::round(value * factor) / factor;
}

void APIENTRY openglCallbackFunction(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	std::cerr << "---------------------" << std::endl;
	std::cerr << "Debug message (" << id << "): " << message << std::endl;
	std::cerr << "Source: " << source << ", Type: " << type << ", Severity: " << severity << std::endl;
	std::cerr << "---------------------" << std::endl;
}

float *gl_mat_to_array(GLdouble mat[16])
{
    scm::math::mat4d gl_mat = scm::math::mat4d(mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
    float *gl_array = scm::math::mat4f(gl_mat).data_array;
    return gl_array;
}

scm::math::mat4d gl_mat(GLdouble mat[16])
{
    scm::math::mat4d gl_mat = scm::math::mat4d(mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9], mat[10], mat[11], mat[12], mat[13], mat[14], mat[15]);
    return gl_mat;
}

std::vector<float> getBoxCorners(scm::gl::boxf box)
{
    std::vector<float> corners = {
        box.corner(0).data_array[0],
        box.corner(0).data_array[1],
        box.corner(0).data_array[2],
        box.corner(1).data_array[0],
        box.corner(1).data_array[1],
        box.corner(1).data_array[2],
        box.corner(2).data_array[0],
        box.corner(2).data_array[1],
        box.corner(2).data_array[2],
        box.corner(3).data_array[0],
        box.corner(3).data_array[1],
        box.corner(3).data_array[2],
        box.corner(4).data_array[0],
        box.corner(4).data_array[1],
        box.corner(4).data_array[2],
        box.corner(5).data_array[0],
        box.corner(5).data_array[1],
        box.corner(5).data_array[2],
        box.corner(6).data_array[0],
        box.corner(6).data_array[1],
        box.corner(6).data_array[2],
        box.corner(7).data_array[0],
        box.corner(7).data_array[1],
        box.corner(7).data_array[2],
    };
    return corners;
}

std::vector<std::vector<float>> getSerializedBvhMinMax(const std::vector<scm::gl::box> &bounding_boxes)
{
    std::vector<std::vector<float>> vecOfVec;
    for (uint64_t node_id = 0; node_id < bounding_boxes.size(); ++node_id)
    {
        scm::math::vec3f min_vertex = bounding_boxes[node_id].min_vertex();
        scm::math::vec3f max_vertex = bounding_boxes[node_id].max_vertex();
        std::vector<float> elements = {
            min_vertex.x, min_vertex.y, min_vertex.z,
            max_vertex.x, max_vertex.y, max_vertex.z};
        vecOfVec.push_back(elements);
    }
    return vecOfVec;
}

void updateFrustumTransform(osg::ref_ptr<osg::MatrixTransform> matrixTransform, const osg::Vec3 &translation)
{
    osg::Matrix transMatrix = osg::Matrix::translate(translation);
    matrixTransform->setMatrix(transMatrix);
};

int CheckGLError(char* file, int line)
{
    GLenum glErr;
    int    retCode = 0;
    glErr = glGetError();
    while (glErr != GL_NO_ERROR) {
        const GLubyte* sError = gluErrorString(glErr);
        if (sError) { std::cerr << "GL Error #" << glErr << "(" << gluErrorString(glErr) << ") " << " in File " << file << " at line: " << line << std::endl; }
        else { std::cerr << "GL Error #" << glErr << " (no message available)" << " in File " << file << " at line: " << line << std::endl; }
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}

#define CHECK_GL_ERROR() CheckGLError(__FILE__, __LINE__)

bool decideUseAniso(const scm::math::mat4& projection_matrix, int anisoMode, float threshold)
{
    // 0=off, 1=auto, 2=on
    if (anisoMode == 2) return true;
    if (anisoMode == 0) return false;
    // Off-axis heuristic: treat small offsets (e.g., stereo eye tiny shifts) as isotropic for performance.
    // Extract the row/column tied to off-axis terms via M * ez (works with our math conversion).
    // Consider anisotropic only if magnitude exceeds a practical threshold.
    const scm::math::vec4 ez(0.0f, 0.0f, 1.0f, 0.0f);
    const scm::math::vec4 v = projection_matrix * ez;
    const float mag = std::max(std::fabs(v[0]), std::fabs(v[1]));
    return mag > std::max(0.0f, threshold);
}

GpuInfo queryGpuInfo()
{
    GpuInfo info;
    const char* vendor = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    const char* renderer = reinterpret_cast<const char*>(glGetString(GL_RENDERER));
    const char* version = reinterpret_cast<const char*>(glGetString(GL_VERSION));

    info.vendor = vendor ? vendor : "unknown";
    info.renderer = renderer ? renderer : "unknown";
    info.version = version ? version : "unknown";

    UuidInfo uuid_info = queryUuidInfo();
    info.device_uuid = uuid_info.device;
    info.driver_uuid = uuid_info.driver;
    if (!info.device_uuid.empty()) {
        info.key = info.device_uuid;
    } else {
        info.key = info.vendor + "|" + info.renderer + "|" + info.version;
    }

    std::ostringstream extra;
    appendVendorSpecificInfo(extra, vendor);
    info.extra = extra.str();
    return info;
}

std::string formatGpuInfoLine(const GpuInfo& info, int ctx, int view_id)
{
    std::ostringstream os;
    os << "[Lamure] ctx=" << ctx
       << " view=" << view_id
       << " gpu_vendor=" << info.vendor
       << " gpu_renderer=" << info.renderer
       << " gpu_version=" << info.version;
    if (!info.device_uuid.empty()) {
        os << " gpu_uuid=" << info.device_uuid;
    }
    if (!info.driver_uuid.empty()) {
        os << " driver_uuid=" << info.driver_uuid;
    }
    if (!info.extra.empty()) {
        os << info.extra;
    }
    return os.str();
}

} // namespace LamureUtil
