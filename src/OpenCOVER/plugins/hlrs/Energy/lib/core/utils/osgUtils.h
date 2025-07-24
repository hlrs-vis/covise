#ifndef _CORE_UTILS_OSGUTILS_H
#define _CORE_UTILS_OSGUTILS_H

#include <memory>
#include <osg/BoundingBox>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/StateSet>
#include <osg/Texture1D>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/ref_ptr>
#include <osgDB/Options>
#include <osgText/Text>
#include <vector>

namespace core::utils::osgUtils {
typedef std::vector<osg::ref_ptr<osg::Geode>> Geodes;

namespace visitors {
class NodeNameToggler : public osg::NodeVisitor {
 public:
  NodeNameToggler(const std::string &targetName)
      : osg::NodeVisitor(osg::NodeVisitor::TRAVERSE_ALL_CHILDREN),
        _targetName(targetName) {}

  virtual void apply(osg::Node &node);

 private:
  std::string _targetName;
};
}  // namespace visitors
   //
namespace instancing {
struct GeometryData {
  osg::ref_ptr<osg::Geometry> geometry;
  osg::ref_ptr<osg::StateSet> stateSet;
};

std::vector<GeometryData> extractAllGeometryData(osg::Node *node);
std::vector<GeometryData> extractTexturedGeometryData(osg::Node *node);
osg::ref_ptr<osg::Node> createInstance(
    const std::vector<GeometryData> &masterGeometryData, const osg::Matrix &matrix);
}  // namespace instancing

void switchTo(const osg::ref_ptr<osg::Node> child, osg::ref_ptr<osg::Switch> parent);
bool isActive(osg::ref_ptr<osg::Switch> switchToCheck,
               osg::ref_ptr<osg::Group> group);
std::unique_ptr<Geodes> getGeodes(osg::Group *grp);
osg::BoundingBox getBoundingBox(const Geodes &geodes);
void deleteChildrenFromOtherGroup(osg::Group *grp, osg::Group *anotherGrp);
void deleteChildrenRecursive(osg::Group *grp);
/**
 * @brief Adds a cylinder between two points.
 * Source: http://www.thjsmith.com/40/cylinder-between-two-points-opengl-c
 *
 * @param start The starting point of the cylinder.
 * @param end The ending point of the cylinder.
 * @param radius The radius of the cylinder.
 * @param cylinderColor The color of the cylinder.
 * @param group The group to which the cylinder will be added.
 * @param hints The tessellation hints for the cylinder.
 */
osg::ref_ptr<osg::Geode> createOsgCylinderBetweenPoints(
    osg::Vec3 start, osg::Vec3 end, float radius, osg::Vec4 cylinderColor,
    osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints());

osg::ref_ptr<osg::Geometry> createCylinderBetweenPoints(
    osg::Vec3 start, osg::Vec3 end, float radius, int circleSegments,
    int lengthSegments,
    osg::ref_ptr<osg::TessellationHints> hints = new osg::TessellationHints(),
    bool colorInterpolation = false);

osg::ref_ptr<osg::Geode> createCylinderBetweenPointsColorInterpolation(
    const osg::Vec3 &start, const osg::Vec3 &end, float halfCylinderHalf,
    float radius, int circleSegments, int lengthSegments,
    const osg::Vec4 &startColor, const osg::Vec4 &endColor,
    osg::ref_ptr<osg::TessellationHints> hints);

osg::Vec3 cubicBezier(float t, const osg::Vec3 &p0, const osg::Vec3 &p1,
                      const osg::Vec3 &p2, const osg::Vec3 &p3);
osg::ref_ptr<osg::Geode> createBezierTube(
    const osg::Vec3 &p1, const osg::Vec3 &p2, float midPointOffset, float tubeRadius,
    int numSegments = 50,
    const osg::Vec4 &color = osg::Vec4(1.0f, 1.0f, 0.0f, 1.0f));

osg::ref_ptr<osg::Texture1D> createPointDataTexture(const std::vector<double> &data);
osg::ref_ptr<osg::Texture2D> createValue1DTexture(const std::vector<double> &data);
// osg::ref_ptr<osg::TextureRectangle> createValue1DTexture(const std::vector<double>
// &data); osg::ref_ptr<osg::Texture1D> createValue1DTexture(const
// std::vector<double> &data);
osg::ref_ptr<osg::Texture2D> createValueTexture(const std::vector<double> &fromData,
                                                const std::vector<double> &toData);
osg::ref_ptr<osgText::Text> createTextBox(const std::string &text,
                                          const osg::Vec3 &position, int charSize,
                                          const char *fontFile,
                                          const float &maxWidth,
                                          const float &margin);
void setTransparency(osg::ref_ptr<osg::Geode> geode, float alpha);
osg::ref_ptr<osg::Geometry> createBackgroundGeometryForText(
    osg::ref_ptr<osgText::Text> text, float padding,
    const osg::Vec4 &backgroundColor, float depthOffset = 0.1f);
osg::ref_ptr<osg::Geometry> createBackgroundQuadGeometry(const osg::Vec3 &center,
                                                         float width, float height,
                                                         const osg::Vec4 &color);
void enableLighting(osg::ref_ptr<osg::Geode> geode, bool enable = true);
osg::ref_ptr<osg::Geode> createBoundingBoxVisualization(const osg::BoundingBox &bb);
osg::ref_ptr<osg::Geode> createBoundingSphereVisualization(
    const osg::BoundingSphere &bs);
osg::ref_ptr<osg::Node> readFileViaOSGDB(const std::string &filename,
                                         osg::ref_ptr<osgDB::Options> options,
                                         bool optimize = false);
void addEmissionMaterial(osg::ref_ptr<osg::Geode> geo,
                         const osg::Vec4 &highlightColor);

osg::ref_ptr<osg::Node> createOutline(osg::ref_ptr<osg::Node> originalNode,
                                      float scaleFactor,
                                      const osg::Vec4 &outlineColor);
void applyOutlineShader(osg::ref_ptr<osg::Geode> geo, const osg::Vec4 &outlineColor,
                        float lineWidth = 2.0f);
void createOutlineFX(osg::ref_ptr<osg::Geode> geode, const osg::Vec4 &outlineColor,
                     float lineWidth = 2.0f);

osg::ref_ptr<osg::Geometry> createNormalVisualization(
    osg::Geometry *originalGeometry);

void printNodeInfo(osg::Node *node, int indent = 0);

}  // namespace core::utils::osgUtils
#endif
