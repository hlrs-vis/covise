#ifndef _LAMURE_UTIL_H
#define _LAMURE_UTIL_H

// std
#include <string>
#include <vector>

#include <GL/glew.h>
#include <scm/core/math.h>
#include <scm/gl_core/primitives/primitives_fwd.h>

#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/Matrix>
#include <osg/MatrixTransform>
#include <osg/Node>
#include <osg/Group>

namespace LamureUtil {

// Forward declarations
osg::Vec3f vecConv3F(scm::math::vec3f& v);
osg::Vec3d vecConv3D(scm::math::vec3d& v);
osg::Vec4f vecConv4F(scm::math::vec4f& v);
osg::Vec4d vecConv4D(scm::math::vec4d& v);

scm::math::vec3f vecConv3F(osg::Vec3f& v);
scm::math::vec3d vecConv3D(osg::Vec3d& v);
scm::math::vec3f vecConv3F(const osg::Vec3f& v);
scm::math::vec3d vecConv3D(const osg::Vec3d& v);


scm::math::vec3f vecConv3F(const osg::Vec4f& v);
scm::math::vec3d vecConv3D(const osg::Vec4d& v);

scm::math::vec4f vecConv4F(osg::Vec4f& v);
scm::math::vec4d vecConv4D(osg::Vec4d& v);
scm::math::vec4f vecConv4F(const osg::Vec4f& v);
scm::math::vec4d vecConv4D(const osg::Vec4d& v);

osg::Matrixf matConv4F(scm::math::mat4f& m);
osg::Matrixd matConv4D(scm::math::mat4d& m);
scm::math::mat4f matConv4F(osg::Matrixd& m);
scm::math::mat4d matConv4D(osg::Matrixd& m);
scm::math::mat4f matConv4F(const osg::Matrixd& m);
scm::math::mat4d matConv4D(const osg::Matrixd& m);

scm::math::mat3f matConv4to3F(scm::math::mat4f& m);
scm::math::mat3f matConv4to3F(scm::math::mat4d& m);
scm::math::mat3d matConv4to3D(scm::math::mat4d& m);

std::string getConfigEntry(std::string scope, std::string name);
bool readIndexedMatrix(const std::string& in, osg::Matrixd& M);
std::string getConfigEntry(std::string scope);

std::string const stripWhitespace(std::string const& in_string);
scm::math::mat4d loadMatrixFromFile(const std::string& filename);
osg::Matrixd loadMatrix(const std::string &value);
bool parsePrefix(std::string& in_string, std::string const& prefix);
void printNodePath(osg::ref_ptr<osg::Node> pointer);
void printChildNodes(osg::Node * node, int depth);
double roundToDecimal(double value, int decimals);
int CheckGLError(char* file, int line);
void APIENTRY openglCallbackFunction(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam);
float* gl_mat_to_array(GLdouble mat[16]);
scm::math::mat4d gl_mat(GLdouble mat[16]);
std::vector<float> getBoxCorners(scm::gl::boxf box);
std::vector<std::vector<float>> getSerializedBvhMinMax(const std::vector<scm::gl::boxf>& bounding_boxes);
std::vector<std::string> splitSemicolons(const std::string& s);

bool decideUseAniso(const scm::math::mat4& projection_matrix, int anisoMode, float threshold);

} // namespace LamureUtil

#endif // _LAMURE_UTIL_H
