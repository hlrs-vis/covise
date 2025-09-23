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
#include <GL/glu.h> // For gluErrorString

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

} // namespace LamureUtil
