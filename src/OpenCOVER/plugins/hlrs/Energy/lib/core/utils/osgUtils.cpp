#include "osgUtils.h"

#include <GL/gl.h>
#include <GL/glext.h>
#include <utils/color.h>

#include <iostream>
#include <memory>
#include <cassert>
#include <osg/Array>
#include <osg/BlendFunc>
#include <osg/BoundingBox>
#include <osg/BoundingSphere>
#include <osg/Depth>
#include <osg/Drawable>
#include <osg/FrameBufferObject>
#include <osg/Geode>
#include <osg/Texture2D>
#include <osg/TextureRectangle>
#include <osg/Geometry>
#include <osg/LineWidth>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/Matrixd>
#include <osg/PolygonMode>
#include <osg/PositionAttitudeTransform>
#include <osg/PrimitiveSet>
#include <osg/Shader>
#include <osg/Shape>
#include <osg/ShapeDrawable>
#include <osg/StateAttribute>
#include <osg/Texture>
#include <osg/Vec3>
#include <osg/Vec4>
#include <osg/ref_ptr>
#include <osg/Texture1D>
#include <osgDB/ReadFile>
#include <osgFX/Outline>
#include <osgText/Text>
#include <osgUtil/Optimizer>
#include <osgUtil/SmoothingVisitor>
#include <utility>
#include <vector>

namespace core::utils::osgUtils {
namespace visitors {
void NodeNameToggler::apply(osg::Node &node) {
  if (node.getName() == _targetName) {
    if (node.getNodeMask() == 0x0) {
      std::cout << "Knoten '" << _targetName << "' aktiviert." << std::endl;
      node.setNodeMask(0xffffffff);
    } else {
      std::cout << "Knoten '" << _targetName << "' deaktiviert." << std::endl;
      node.setNodeMask(0x0);
    }
  }
  traverse(node);
}
}  // namespace visitors
namespace instancing {
std::vector<GeometryData> extractAllGeometryData(osg::Node *node) {
  std::vector<GeometryData> geometryDataList;
  osg::ref_ptr<osg::Geode> geode = node->asGeode();
  if (geode) {
    for (unsigned int i = 0; i < geode->getNumDrawables(); ++i) {
      osg::ref_ptr<osg::Geometry> geometry = geode->getDrawable(i)->asGeometry();
      if (geometry) {
        GeometryData data;
        data.geometry = geometry;
        data.stateSet = geode->getStateSet();  // use state set of the geode
        geometryDataList.push_back(data);
      }
    }
  }

  osg::ref_ptr<osg::Group> group = node->asGroup();
  if (group) {
    for (unsigned int i = 0; i < group->getNumChildren(); ++i) {
      std::vector<GeometryData> childData =
          extractAllGeometryData(group->getChild(i));
      geometryDataList.insert(geometryDataList.end(), childData.begin(),
                              childData.end());
    }
  }
  return geometryDataList;
}

std::vector<GeometryData> extractTexturedGeometryData(osg::Node *node) {
  auto geometryDataList = extractAllGeometryData(node);
  std::vector<GeometryData> texturedGeometryDataList;
  if (geometryDataList.empty()) return texturedGeometryDataList;

  // get only the textured geometry data
  for (int i = 0; i < geometryDataList.size(); ++i) {
    const auto &geometryData = geometryDataList[i];
    auto geom = geometryData.geometry;
    auto geomStateset = geom->getOrCreateStateSet();
    if (!geomStateset) continue;
    auto textureList = geomStateset->getTextureAttributeList();
    bool hasTexture = false;
    for (const auto &attributeMap : textureList) {
      if (hasTexture) break;
      for (const auto &pair : attributeMap) {
        const osg::StateAttribute::TypeMemberPair &typeMemberPair = pair.first;
        const osg::StateSet::RefAttributePair &refAttributePair = pair.second;
        osg::StateAttribute *attribute = refAttributePair.first.get();
        if (attribute && typeMemberPair.first == osg::StateAttribute::TEXTURE) {
          unsigned int textureUnit = typeMemberPair.second;
          osg::Texture *texture = dynamic_cast<osg::Texture *>(attribute);
          if (texture) {
            texturedGeometryDataList.emplace_back(
                instancing::GeometryData{geom, geomStateset});
            hasTexture = true;
            break;
          }
        }
      }
    }
  }
  return texturedGeometryDataList;
}

osg::ref_ptr<osg::Node> createInstance(
    const std::vector<GeometryData> &masterGeometryData, const osg::Matrix &matrix) {
  osg::ref_ptr<osg::Group> instanceRoot = new osg::Group;
  osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform(matrix);
  instanceRoot->addChild(mt);

  for (const auto &data : masterGeometryData) {
    osg::ref_ptr<osg::Geode> geode = new osg::Geode;
    geode->addDrawable(data.geometry);
    geode->setStateSet(data.stateSet);  // share state set
    mt->addChild(geode);
  }

  return instanceRoot.release();
}

}  // namespace instancing
osg::ref_ptr<osgText::Text> createTextBox(const std::string &text,
                                          const osg::Vec3 &position, int charSize,
                                          const char *fontFile,
                                          const float &maxWidth,
                                          const float &margin) {
  osg::ref_ptr<osgText::Text> textBox = new osgText::Text();
  textBox->setAlignment(osgText::Text::LEFT_TOP);
  textBox->setAxisAlignment(osgText::Text::XZ_PLANE);
  textBox->setColor(osg::Vec4(1, 1, 1, 1));
  textBox->setText(text, osgText::String::ENCODING_UTF8);
  textBox->setCharacterSize(charSize);
  textBox->setFont(fontFile);
  textBox->setMaximumWidth(maxWidth);
  textBox->setPosition(position);
  //   textBox->setDrawMode(osgText::Text::FILLEDBOUNDINGBOX | osgText::Text::TEXT);
  textBox->setBoundingBoxMargin(margin);
  textBox->setDrawMode(osgText::Text::TEXT);
  return textBox;
}

void enableLighting(osg::ref_ptr<osg::Geode> geode, bool enable) {
  osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();
  if (enable) {
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    return;
  }
  stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
}

void setTransparency(osg::ref_ptr<osg::Geode> geode, float alpha) {
  osg::ref_ptr<osg::StateSet> stateset = geode->getOrCreateStateSet();

  // Enable blending
  stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
  stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);

  // Set the blend function
  osg::ref_ptr<osg::BlendFunc> blendFunc = new osg::BlendFunc();
  blendFunc->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  stateset->setAttributeAndModes(blendFunc, osg::StateAttribute::ON);

  // Modify the color to include alpha
  for (int i = 0; i < geode->getNumDrawables(); ++i) {
    osg::ref_ptr<osg::Drawable> drawable = geode->getDrawable(i);
    osg::ref_ptr<osg::Geometry> geometry = drawable->asGeometry();
    osg::ref_ptr<osg::Vec4Array> colors =
        dynamic_cast<osg::Vec4Array *>(geometry->getColorArray());
    if (colors) {
      (*colors)[0].a() = alpha;  // Set the alpha value
      geometry->setColorArray(colors, osg::Array::BIND_OVERALL);
      geometry->setColorBinding(
          osg::Geometry::BIND_OVERALL);  // Ensure the color binding is correct.
    }
  }
}

osg::ref_ptr<osg::Geometry> createBackgroundQuadGeometry(const osg::Vec3 &center,
                                                         float width, float height,
                                                         const osg::Vec4 &color) {
  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();

  //
  //         +Z
  //          |  +-----+
  //          |  |     |
  //          |  +-----+
  //          O--------+X
  //         /
  //        /
  //       -Y
  //
  osg::Vec3 topRight(center.x() + width / 2, center.y(), center.z() + height / 2);
  osg::Vec3 bottomRight(center.x() + width / 2, center.y(), center.z() - height / 2);
  osg::Vec3 bottomLeft(center.x() - width / 2, center.y(), center.z() - height / 2);
  osg::Vec3 topLeft(center.x() - width / 2, center.y(), center.z() + height / 2);
  // Vertices of the quad
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array();
  vertices->push_back(topRight);
  vertices->push_back(bottomRight);
  vertices->push_back(bottomLeft);
  vertices->push_back(topLeft);
  geometry->setVertexArray(vertices);

  // Indices for the quad
  osg::ref_ptr<osg::DrawElementsUInt> quad =
      new osg::DrawElementsUInt(osg::PrimitiveSet::QUADS, 0, 4);
  for (auto i = 0; i < 4; ++i) quad->push_back(i);
  geometry->addPrimitiveSet(quad);

  // Set the color
  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array();
  colors->push_back(color);
  geometry->setColorArray(colors, osg::Array::BIND_PER_PRIMITIVE_SET);
  return geometry;
}

osg::ref_ptr<osg::Geometry> createBackgroundGeometryForText(
    osg::ref_ptr<osgText::Text> text, float padding,
    const osg::Vec4 &backgroundColor, float depthOffset) {
  const auto &bb = text->getBoundingBox();
  float width = bb.xMax() - bb.xMin() + 2 * padding;
  float height = bb.zMax() - bb.zMin() + 2 * padding;
  osg::Vec3 center = bb.center();
  center.y() += depthOffset;

  return createBackgroundQuadGeometry(center, width, height, backgroundColor);
}

std::unique_ptr<Geodes> getGeodes(osg::Group *grp) {
  Geodes geodes{};
  for (unsigned int i = 0; i < grp->getNumChildren(); ++i) {
    auto child = grp->getChild(i);
    if (osg::ref_ptr<osg::Geode> child_geode = dynamic_cast<osg::Geode *>(child)) {
      geodes.push_back(child_geode);
      continue;
    }
    if (osg::ref_ptr<osg::Group> child_group = dynamic_cast<osg::Group *>(child)) {
      auto child_geodes = getGeodes(child_group);
      std::move(child_geodes->begin(), child_geodes->end(),
                std::back_inserter(geodes));
    }
  }
  return std::make_unique<Geodes>(geodes);
}

osg::BoundingBox getBoundingBox(
    const std::vector<osg::ref_ptr<osg::Geode>> &geodes) {
  osg::BoundingBox bb;
  for (auto geode : geodes) {
    bb.expandBy(geode->getBoundingBox());
  }
  return bb;
}

void deleteChildrenFromOtherGroup(osg::Group *grp, osg::Group *other) {
  if (!grp || !other) return;

  for (unsigned int i = 0; i < other->getNumChildren(); ++i) {
    auto child = other->getChild(i);
    if (grp->containsNode(child)) grp->removeChild(child);
  }
}

void deleteChildrenRecursive(osg::Group *grp) {
  if (!grp) return;

  for (unsigned int i = 0; i < grp->getNumChildren(); ++i) {
    auto child = grp->getChild(i);
    if (auto child_group = dynamic_cast<osg::Group *>(child))
      deleteChildrenRecursive(child_group);
    grp->removeChild(child);
  }
}

osg::ref_ptr<osg::Geode> createOsgCylinderBetweenPoints(
    osg::Vec3 start, osg::Vec3 end, float radius, osg::Vec4 cylinderColor,
    osg::ref_ptr<osg::TessellationHints> hints) {
  osg::ref_ptr geode = new osg::Geode;
  osg::Vec3 center;
  float height;
  osg::ref_ptr<osg::Cylinder> cylinder;
  osg::ref_ptr<osg::ShapeDrawable> cylinderDrawable;
  osg::ref_ptr<osg::Material> pMaterial;

  height = (start - end).length();
  center = osg::Vec3((start.x() + end.x()) / 2, (start.y() + end.y()) / 2,
                     (start.z() + end.z()) / 2);

  // This is the default direction for the cylinders to face in OpenGL
  osg::Vec3 z = osg::Vec3(0, 0, 1);

  // Get diff between two points you want cylinder along
  osg::Vec3 p = start - end;

  // Get CROSS product (the axis of rotation)
  osg::Vec3 t = z ^ p;

  // Get angle. length is magnitude of the vector
  double angle = acos((z * p) / p.length());

  // Create a cylinder between the two points with the given radius
  cylinder = new osg::Cylinder(center, radius, height);
  cylinder->setRotation(osg::Quat(angle, osg::Vec3(t.x(), t.y(), t.z())));

  cylinderDrawable = new osg::ShapeDrawable(cylinder, hints);
  geode->addDrawable(cylinderDrawable);

  // Set the color of the cylinder that extends between the two points.
  color::overrideGeodeColor(geode, cylinderColor);

  return geode;
}

std::vector<osg::Vec3> calculatePointsAlongLine(const osg::Vec3 &start,
                                                const osg::Vec3 &end,
                                                int lengthSegments) {
  std::vector<osg::Vec3> basePoints;
  basePoints.push_back(start);
  osg::Vec3 direction = start - end;
  direction.normalize();
  direction = -direction;
  float height;
  height = (start - end).length();
  for (int i = 1; i <= lengthSegments; ++i) {
    float segment = static_cast<float>(i) / lengthSegments;
    osg::Vec3 p = start + (direction * segment * height);
    basePoints.push_back(p);
  }
  return basePoints;
}

std::pair<osg::ref_ptr<osg::Vec3Array>, osg::ref_ptr<osg::Vec3Array>>
generateVerticesAndNormalsForTube(const std::vector<osg::Vec3> &basePoints,
                                  const osg::Vec3 &direction, int circleSegments,
                                  float radius) {
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
  osg::Vec3 up(0.0f, 0.0f, 1.0f);  // Default up vector.
  osg::Vec3 right = up ^ direction;
  for (size_t i = 0; i < basePoints.size(); ++i) {
    osg::Vec3 currentPoint = basePoints[i];

    // Calculate tangent, normal, and binormal.
    osg::Vec3 tangent;
    if (i == 0) {
      tangent = basePoints[1] - basePoints[0];
    } else if (i == basePoints.size() - 1) {
      tangent = basePoints[i] - basePoints[i - 1];
    } else {
      osg::Vec3 tangent1 = basePoints[i + 1] - basePoints[i];
      tangent1.normalize();
      osg::Vec3 tangent2 = basePoints[i] - basePoints[i - 1];
      tangent2.normalize();
      tangent = (tangent1 + tangent2);
    }
    tangent.normalize();

    osg::Vec3 normal = right ^ tangent;
    normal.normalize();
    osg::Vec3 binormal = tangent ^ normal;
    binormal.normalize();

    // Generate vertices around the tube.
    for (int j = 0; j <= circleSegments; ++j) {
      float angle = 2.0f * osg::PI * static_cast<float>(j) / circleSegments;
      osg::Vec3 vertex =
          currentPoint + (binormal * cos(angle) + normal * sin(angle)) * radius;
      vertices->push_back(vertex);

      osg::Vec3 outwardNormal = (binormal * cos(angle) + normal * sin(angle));
      outwardNormal.normalize();
      normals->push_back(outwardNormal);
    }
  }
  return std::make_pair(vertices, normals);
}

osg::ref_ptr<osg::DrawElementsUInt> createIndicesForTube(int lengthSegments,
                                                         int circleSegments) {
  osg::ref_ptr<osg::DrawElementsUInt> indices =
      new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLE_STRIP);
  for (int i = 0; i < lengthSegments; ++i) {
    for (int j = 0; j <= circleSegments; ++j) {
      indices->push_back((i + 0) * (lengthSegments + 1) + j);
      indices->push_back((i + 1) * (circleSegments + 1) + j);
    }
  }
  return indices;
}

void smoothGeometry(osg::ref_ptr<osg::Geometry> geometry) {
  osgUtil::SmoothingVisitor sv;
  geometry->accept(sv);
}

constexpr int SHADER_INDEX_ATTRIB = 5;

osg::ref_ptr<osg::Geometry> createCylinderBetweenPoints(
    osg::Vec3 start, osg::Vec3 end, float radius, int circleSegments,
    int lengthSegments, osg::ref_ptr<osg::TessellationHints> hints, bool colorInterpolation) {
  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
  osg::Vec3 direction = start - end;
  direction.normalize();
  direction = -direction;

  auto basePoints = calculatePointsAlongLine(start, end, lengthSegments);

  auto [vertices, normals] = generateVerticesAndNormalsForTube(
      basePoints, direction, circleSegments, radius);

  // Generate indices.
  auto indices = createIndicesForTube(lengthSegments, circleSegments);

  geometry->setVertexArray(vertices);
  geometry->setNormalArray(normals, osg::Array::BIND_PER_VERTEX);
  geometry->addPrimitiveSet(indices);

  // shader attribute mapping from vertices to data value in texture
  osg::IntArray *intArray = new osg::IntArray;
  for (size_t i = 0; i < lengthSegments + 1; i++) {
    for (size_t j = 0; j < circleSegments; j++) {
    //   intArray->push_back(i);
    //   intArray->push_back(colorInterpolation ? i : 0);
      int val = (colorInterpolation ? i : 0);
    //   std::cout << "[DEBUG] indexAttrib[" << (i * circleSegments + j)
    //             << "] = " << val << std::endl;
      intArray->push_back(val);
    }
  }
  intArray->setBinding(osg::Array::BIND_PER_VERTEX);
  geometry->setVertexAttribArray(SHADER_INDEX_ATTRIB, intArray,
                                 osg::Array::BIND_PER_VERTEX);
  // Set the color array for color interpolation between start and end.
  smoothGeometry(geometry);

  return geometry;
}

osg::ref_ptr<osg::Vec4Array>
createColorArrayForTubeColorInterpolationBetweenStartAndEnd(
    int lengthSegments, int circleSegments, const osg::Vec4 &colorStart,
    const osg::Vec4 &colorEnd) {
  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
  for (int i = 0; i <= lengthSegments; ++i) {
    auto color = i % 2 ? colorEnd : colorStart;
    for (int j = 0; j <= circleSegments; ++j) colors->push_back(color);
  }
  return colors;
}

osg::ref_ptr<osg::Texture2D> createValue1DTexture(const std::vector<double> &data)  {
    osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D;
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    texture->setBorderWidth(0);
    texture->setResizeNonPowerOfTwoHint(false);
    texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);

    // Create the image
    osg::ref_ptr<osg::Image> image = new osg::Image();
    image->setInternalTextureFormat(GL_R32F);
    image->allocateImage(data.size(), 1, 1, GL_RED, GL_FLOAT);
    float* values = reinterpret_cast<float*>(image->data());
    for (size_t i = 0; i < data.size(); ++i)
      values[i] = data[i];

    image->dirty();
    texture->setImage(image);
    return texture;
}

osg::ref_ptr<osg::Texture1D> createPointDataTexture(const std::vector<double>& data) {
    osg::ref_ptr<osg::Texture1D> tex = new osg::Texture1D();
    // tex->setInternalFormat(GL_R32F);
    tex->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
    tex->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
    osg::ref_ptr<osg::Image> img = new osg::Image();
    img->setInternalTextureFormat(GL_R32F);
    img->allocateImage(data.size(), 1, 1, GL_RED, GL_FLOAT);
    memcpy(img->data(), data.data(), data.size() * sizeof(float));
    img->dirty();
    tex->setImage(img);
    return tex;
}

osg::ref_ptr<osg::Texture2D> createValueTexture(const std::vector<double> &fromData,
                                                const std::vector<double> &toData) {
  assert(fromData.size() == toData.size() &&
         "fromData and toData must have the same size");
  osg::ref_ptr<osg::Texture2D> texture = new osg::Texture2D();
//   texture->setInternalFormat(GL_R32F);  // 1 channel, 32-bit float
  texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::NEAREST);
  texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::NEAREST);
  texture->setBorderWidth(0);
  texture->setResizeNonPowerOfTwoHint(false);
  texture->setWrap(osg::Texture2D::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
  // Create the image
  osg::ref_ptr<osg::Image> image = new osg::Image();
  image->setInternalTextureFormat(GL_R32F);
  image->allocateImage(fromData.size(), 2, 1, GL_RED, GL_FLOAT);
  unsigned char *v = image->data();

  auto values = (float *)(v);
  size_t index = 0;
  for (auto val : fromData) {
    values[index] = fromData[index];
    ++index;
  }
  for (const auto &val : toData) {
    values[index] = toData[index - fromData.size()];
    ++index;
  }

  image->dirty();
  texture->setImage(image);
  return texture;
}

osg::ref_ptr<osg::Geode> createCylinderBetweenPointsColorInterpolation(
    const osg::Vec3 &start, const osg::Vec3 &end, float halfCylinderHalf,
    float radius, int circleSegments, int lengthSegments,
    const osg::Vec4 &startColor, const osg::Vec4 &endColor,
    osg::ref_ptr<osg::TessellationHints> hints) {
  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  osg::ref_ptr<osg::Geometry> geometry = createCylinderBetweenPoints(
      start, end, radius, circleSegments, lengthSegments, hints);
  osg::ref_ptr<osg::Material> material = new osg::Material;

  // NOTE: for debugging basePoints
  // osg::Vec3Array* vertices =
  // dynamic_cast<osg::Vec3Array*>(geometry->getVertexArray()); for (const auto&
  // vertex : *vertices) {
  //     osg::ref_ptr<osg::Sphere> sph = new osg::Sphere(vertex, 0.25f);
  //     osg::ref_ptr<osg::ShapeDrawable> d = new osg::ShapeDrawable(sph);
  //     d->setColor(osg::Vec4(0.0f, 1.0f, 1.0f, 1.0f));
  //     geode->addChild(d);
  // }

  geode->addDrawable(geometry);

  // Add Material
  material->setAmbient(osg::Material::FRONT_AND_BACK,
                       osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
  material->setColorMode(osg::Material::AMBIENT_AND_DIFFUSE);

  // need to set the color per vertex because of the color interpolation between end
  // and start without a shader
  auto colors = createColorArrayForTubeColorInterpolationBetweenStartAndEnd(
      lengthSegments, circleSegments, startColor, endColor);

  geometry->setColorArray(colors, osg::Array::BIND_PER_VERTEX);
  geode->getOrCreateStateSet()->setAttribute(material);

  return geode;
}

osg::Vec3 cubicBezier(float t, const osg::Vec3 &p0, const osg::Vec3 &p1,
                      const osg::Vec3 &p2, const osg::Vec3 &p3) {
  float u = 1 - t;
  float tt = t * t;
  float uu = u * u;
  float uuu = uu * u;
  float ttt = tt * t;

  osg::Vec3 p = p0 * uuu;
  p += p1 * (3 * uu * t);
  p += p2 * (3 * u * tt);
  p += p3 * ttt;

  return p;
}

osg::ref_ptr<osg::Geode> createBezierTube(const osg::Vec3 &p1, const osg::Vec3 &p2,
                                          float midPointOffset, float tubeRadius,
                                          int numSegments, const osg::Vec4 &color) {
  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;

  osg::Vec3 midPoint = (p1 + p2) * 0.5f;
  osg::Vec3 direction = (p2 - p1);
  direction.normalize();

  // Find a vector perpendicular to the direction.
  // osg::Vec3 up(0.0f, 1.0f, 0.0f); // Default up vector.
  osg::Vec3 up(0.0f, 0.0f, 1.0f);  // Default up vector.
  osg::Vec3 right = direction ^ up;
  if (right.length2() < 1e-6) {
    // If direction is parallel to up, use another up.
    up.set(0.0f, 1.0f, 0.0f);
    right = direction ^ up;
  }
  right.normalize();
  osg::Vec3 newUp = direction ^ right;
  newUp.normalize();

  // Calculate the offset point.
  osg::Vec3 offsetPoint = midPoint + newUp * midPointOffset;

  // Create control points for the cubic Bezier curve.
  osg::Vec3 controlPoint1 = p1 + (offsetPoint - p1) * 0.33f;
  osg::Vec3 controlPoint2 = p2 + (offsetPoint - p2) * 0.33f;

  // Generate vertices along the Bezier curve.
  std::vector<osg::Vec3> bezierPoints;
  for (int i = 0; i <= numSegments; ++i) {
    float t = static_cast<float>(i) / numSegments;
    bezierPoints.push_back(cubicBezier(t, p1, controlPoint1, controlPoint2, p2));
  }

  // Generate tube vertices and normals.
  for (size_t i = 0; i < bezierPoints.size(); ++i) {
    osg::Vec3 currentPoint = bezierPoints[i];

    // Calculate tangent, normal, and binormal.
    osg::Vec3 tangent;
    if (i == 0) {
      tangent = bezierPoints[1] - bezierPoints[0];
      tangent.normalize();
    } else if (i == bezierPoints.size() - 1) {
      tangent = bezierPoints[i] - bezierPoints[i - 1];
      tangent.normalize();
    } else {
      osg::Vec3 tangent1 = bezierPoints[i + 1] - bezierPoints[i];
      tangent1.normalize();
      osg::Vec3 tangent2 = bezierPoints[i] - bezierPoints[i - 1];
      tangent2.normalize();
      tangent = (tangent1 + tangent2);
      tangent.normalize();
    }

    osg::Vec3 normal = right ^ tangent;
    normal.normalize();
    osg::Vec3 binormal = tangent ^ normal;
    binormal.normalize();

    // Generate vertices around the tube.
    for (int j = 0; j <= numSegments; ++j) {
      float angle = 2.0f * osg::PI * static_cast<float>(j) / numSegments;
      osg::Vec3 vertex =
          currentPoint + (normal * cos(angle) + binormal * sin(angle)) * tubeRadius;
      vertices->push_back(vertex);

      osg::Vec3 outwardNormal = (normal * cos(angle) + binormal * sin(angle));
      outwardNormal.normalize();
      normals->push_back(outwardNormal);
    }
  }

  // Generate indices.
  osg::ref_ptr<osg::DrawElementsUInt> indices =
      new osg::DrawElementsUInt(osg::PrimitiveSet::TRIANGLE_STRIP);
  for (int i = 0; i < numSegments; ++i) {
    for (int j = 0; j <= numSegments; ++j) {
      indices->push_back((i + 0) * (numSegments + 1) + j);
      indices->push_back((i + 1) * (numSegments + 1) + j);
    }
  }

  geometry->setVertexArray(vertices.get());
  geometry->setNormalArray(normals.get(), osg::Array::BIND_PER_VERTEX);
  geometry->addPrimitiveSet(indices.get());

  osgUtil::SmoothingVisitor sv;
  geometry->accept(sv);

  osg::ref_ptr<osg::Geode> geode = new osg::Geode;
  geode->addDrawable(geometry.get());

  // Add Material
  osg::ref_ptr<osg::Material> material = new osg::Material;
  material->setAmbient(osg::Material::FRONT_AND_BACK,
                       osg::Vec4(0.2f, 0.2f, 0.2f, 1.0f));
  material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
  geode->getOrCreateStateSet()->setAttribute(material.get());

  return geode;
}

osg::ref_ptr<osg::Geode> createBoundingBoxVisualization(const osg::BoundingBox &bb) {
  osg::ref_ptr<osg::Geode> geode = new osg::Geode();
  osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry();

  // Vertices of the bounding box
  osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
  vertices->push_back(osg::Vec3(bb.xMin(), bb.yMin(), bb.zMin()));  // 0
  vertices->push_back(osg::Vec3(bb.xMax(), bb.yMin(), bb.zMin()));  // 1
  vertices->push_back(osg::Vec3(bb.xMax(), bb.yMax(), bb.zMin()));  // 2
  vertices->push_back(osg::Vec3(bb.xMin(), bb.yMax(), bb.zMin()));  // 3
  vertices->push_back(osg::Vec3(bb.xMin(), bb.yMin(), bb.zMax()));  // 4
  vertices->push_back(osg::Vec3(bb.xMax(), bb.yMin(), bb.zMax()));  // 5
  vertices->push_back(osg::Vec3(bb.xMax(), bb.yMax(), bb.zMax()));  // 6
  vertices->push_back(osg::Vec3(bb.xMin(), bb.yMax(), bb.zMax()));  // 7

  geometry->setVertexArray(vertices);

  // Indices for the edges of the bounding box
  osg::ref_ptr<osg::DrawElementsUInt> lines =
      new osg::DrawElementsUInt(osg::PrimitiveSet::LINES, 0);
  lines->push_back(0);
  lines->push_back(1);
  lines->push_back(1);
  lines->push_back(2);
  lines->push_back(2);
  lines->push_back(3);
  lines->push_back(3);
  lines->push_back(0);
  lines->push_back(4);
  lines->push_back(5);
  lines->push_back(5);
  lines->push_back(6);
  lines->push_back(6);
  lines->push_back(7);
  lines->push_back(7);
  lines->push_back(4);
  lines->push_back(0);
  lines->push_back(4);
  lines->push_back(1);
  lines->push_back(5);
  lines->push_back(2);
  lines->push_back(6);
  lines->push_back(3);
  lines->push_back(7);

  geometry->addPrimitiveSet(lines);

  // Set the color of the lines (e.g., red)
  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
  colors->push_back(osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f));  // Red
  geometry->setColorArray(colors);
  geometry->setColorBinding(osg::Geometry::BIND_OVERALL);

  // Make the lines thicker
  osg::ref_ptr<osg::LineWidth> lineWidth = new osg::LineWidth(2.0f);
  geometry->getOrCreateStateSet()->setAttributeAndModes(lineWidth,
                                                        osg::StateAttribute::ON);

  geode->addDrawable(geometry);

  return geode;
}

osg::ref_ptr<osg::Geode> createBoundingSphereVisualization(
    const osg::BoundingSphere &bs) {
  osg::ref_ptr<osg::Geode> geode = new osg::Geode();

  // Create a wireframe sphere using osg::ShapeDrawable
  osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(bs.center(), bs.radius());
  osg::ref_ptr<osg::ShapeDrawable> shapeDrawable = new osg::ShapeDrawable(sphere);

  // Set wireframe mode
  osg::ref_ptr<osg::PolygonMode> polygonMode = new osg::PolygonMode;
  polygonMode->setMode(osg::PolygonMode::FRONT_AND_BACK, osg::PolygonMode::LINE);
  shapeDrawable->getOrCreateStateSet()->setAttributeAndModes(
      polygonMode, osg::StateAttribute::ON);

  // Set color (e.g., green)
  osg::ref_ptr<osg::Vec4Array> colors = new osg::Vec4Array;
  colors->push_back(osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f));  // Green
  shapeDrawable->setColorArray(colors);
  shapeDrawable->setColorBinding(osg::Geometry::BIND_OVERALL);

  // Make the lines thicker
  osg::ref_ptr<osg::LineWidth> lineWidth = new osg::LineWidth(2.0f);
  shapeDrawable->getOrCreateStateSet()->setAttributeAndModes(
      lineWidth, osg::StateAttribute::ON);

  geode->addDrawable(shapeDrawable);

  return geode;
}

osg::ref_ptr<osg::Node> readFileViaOSGDB(const std::string &filename,
                                         osg::ref_ptr<osgDB::Options> options,
                                         bool optimize) {
  osg::ref_ptr<osg::Node> node = osgDB::readNodeFile(filename, options);
  if (!node) {
    std::cerr << "Error: Unable to read file: " << filename << std::endl;
    return nullptr;
  }

  // Optimize the node
  if (optimize) {
    osgUtil::Optimizer optimizer;
    optimizer.optimize(node);
  }

  return node;
}

void addEmissionMaterial(osg::ref_ptr<osg::Geode> geo,
                         const osg::Vec4 &highlightColor) {
  if (geo) {
    for (unsigned int i = 0; i < geo->getNumDrawables(); ++i) {
      osg::StateSet *stateset = geo->getDrawable(i)->getOrCreateStateSet();
      osg::ref_ptr<osg::Material> material = new osg::Material;
      material->setEmission(osg::Material::FRONT_AND_BACK, highlightColor);
      stateset->setAttributeAndModes(material.get(), osg::StateAttribute::ON);
    }
  }
}

osg::ref_ptr<osg::Node> createOutline(osg::ref_ptr<osg::Node> originalNode,
                                      float scaleFactor,
                                      const osg::Vec4 &outlineColor) {
  osg::ref_ptr<osg::MatrixTransform> outlineMT = new osg::MatrixTransform();
  outlineMT->setMatrix(osg::Matrix::scale(scaleFactor, scaleFactor, scaleFactor));

  osg::ref_ptr<osg::Node> outlineNode = originalNode;

  osg::Geode *geode = outlineNode->asGeode();
  if (geode) {
    for (unsigned int i = 0; i < geode->getNumDrawables(); ++i) {
      osg::StateSet *stateset = geode->getDrawable(i)->getOrCreateStateSet();
      osg::ref_ptr<osg::Material> material = new osg::Material;
      material->setDiffuse(osg::Material::FRONT_AND_BACK, outlineColor);
      material->setAmbient(osg::Material::FRONT_AND_BACK,
                           outlineColor * 0.2f);  // Etwas dunklerer Rand
      stateset->setAttributeAndModes(material.get(), osg::StateAttribute::ON);
      stateset->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    }
  }

  osg::ref_ptr<osg::Group> group = new osg::Group;
  group->addChild(outlineMT.get());
  group->addChild(originalNode.get());

  return group.get();
}

osg::ref_ptr<osg::Geometry> createNormalVisualization(
    osg::Geometry *originalGeometry) {
  osg::Vec3Array *vertices =
      dynamic_cast<osg::Vec3Array *>(originalGeometry->getVertexArray());
  osg::Vec3Array *normals =
      dynamic_cast<osg::Vec3Array *>(originalGeometry->getNormalArray());

  if (!vertices || !normals || vertices->size() != normals->size()) {
    std::cerr << "Error: Invalid vertex or normal data." << std::endl;
    return nullptr;
  }

  osg::ref_ptr<osg::Geometry> normalsGeometry = new osg::Geometry;
  osg::ref_ptr<osg::Vec3Array> normalVertices = new osg::Vec3Array;
  osg::ref_ptr<osg::Vec4Array> normalColors = new osg::Vec4Array;
  osg::ref_ptr<osg::DrawArrays> normalPrimitives =
      new osg::DrawArrays(osg::PrimitiveSet::LINES, 0, 0);

  normalsGeometry->setVertexArray(normalVertices.get());
  normalsGeometry->setColorArray(normalColors.get());
  normalsGeometry->setColorBinding(osg::Geometry::BIND_OVERALL);
  normalsGeometry->addPrimitiveSet(normalPrimitives.get());

  float normalLength = 0.1f;
  osg::Vec4 normalColor(1.0f, 0.0f, 0.0f, 1.0f);

  for (size_t i = 0; i < vertices->size(); ++i) {
    osg::Vec3 vertex = (*vertices)[i];
    osg::Vec3 normal = (*normals)[i];
    osg::Vec3 endPoint = vertex + normal * normalLength;

    normalVertices->push_back(vertex);
    normalVertices->push_back(endPoint);
    normalColors->push_back(normalColor);
  }

  normalPrimitives->setCount(normalVertices->size());

  osg::ref_ptr<osg::LineWidth> lineWidth = new osg::LineWidth();
  lineWidth->setWidth(2.0f);
  normalsGeometry->getOrCreateStateSet()->setAttributeAndModes(
      lineWidth.get(), osg::StateAttribute::ON);
  normalsGeometry->getOrCreateStateSet()->setMode(GL_LIGHTING,
                                                  osg::StateAttribute::OFF);

  return normalsGeometry.get();
}

void applyOutlineShader(osg::ref_ptr<osg::Geode> geode,
                        const osg::Vec4 &outlineColor, float outlineWidth) {
  if (!geode.valid()) {
    OSG_WARN << "addOutlineShaderWithTexture: Ungültige Geode übergeben."
             << std::endl;
    return;
  }

  osg::StateSet *stateSet = geode->getOrCreateStateSet();

  // Kombinierter Vertex Shader Source
  const char *vertexShaderSource =
      "#version 150 core\n"
      "\n"
      "in vec4 osg_Vertex;\n"
      "in vec3 osg_Normal;\n"
      "in vec2 osg_TexCoord0;\n"
      "uniform mat4 osg_ModelViewMatrix;\n"
      "uniform mat4 osg_ProjectionMatrix;\n"
      "uniform float outlineWidth;\n"
      "\n"
      "out vec2 fragTexCoord;\n"
      "out float outlineFactor;\n"
      "\n"
      "void main()\n"
      "{\n"
      "    vec3 normalWorld = normalize(mat3(osg_ModelViewMatrix) * osg_Normal);\n"
      "    vec4 offset = vec4(normalWorld * outlineWidth, 0.0);\n"
      "    gl_Position = osg_ProjectionMatrix * (osg_ModelViewMatrix * osg_Vertex + "
      "offset);\n"
      "    fragTexCoord = osg_TexCoord0;\n"
      "    // Einfacher Ansatz: Wenn der Vertex verschoben wurde, ist er Teil der "
      "Umrandung\n"
      "    outlineFactor = length(offset.xyz) > 0.0 ? 1.0 : 0.0;\n"
      "}\n";

  // Kombinierter Fragment Shader Source
  const char *fragmentShaderSource =
      "#version 150 core\n"
      "\n"
      "uniform vec4 outlineColor;\n"
      "uniform sampler2D texture0;\n"
      "uniform bool hasTexture;\n"
      "\n"
      "in vec2 fragTexCoord;\n"
      "in float outlineFactor;\n"
      "out vec4 fragColor;\n"
      "\n"
      "void main()\n"
      "{\n"
      "    vec4 baseColor = vec4(1.0);\n"
      "    if (hasTexture)\n"
      "    {\n"
      "        baseColor = texture(texture0, fragTexCoord);\n"
      "    }\n"
      "    // Mische die Texturfarbe mit der Umrandungsfarbe basierend auf dem "
      "outlineFactor\n"
      "    fragColor = mix(baseColor, outlineColor, outlineFactor);\n"
      "}\n";

  // Erstelle die Shader-Objekte
  osg::ref_ptr<osg::Shader> vertexShader = new osg::Shader(osg::Shader::VERTEX);
  vertexShader->setShaderSource(vertexShaderSource);

  osg::ref_ptr<osg::Shader> fragmentShader = new osg::Shader(osg::Shader::FRAGMENT);
  fragmentShader->setShaderSource(fragmentShaderSource);

  // Erstelle das Shader-Programm
  osg::ref_ptr<osg::Program> program = new osg::Program();
  program->addShader(vertexShader.get());
  program->addShader(fragmentShader.get());

  // Weise das Programm dem StateSet zu
  stateSet->setAttributeAndModes(program.get(), osg::StateAttribute::ON);

  // Erstelle und setze die Uniformen
  osg::ref_ptr<osg::Uniform> colorUniform =
      new osg::Uniform("outlineColor", outlineColor);
  stateSet->addUniform(colorUniform.get());

  osg::ref_ptr<osg::Uniform> widthUniform =
      new osg::Uniform("outlineWidth", outlineWidth);
  stateSet->addUniform(widthUniform.get());

  // Überprüfe, ob die Geode eine Textur hat und setze die entsprechende Uniform
  osg::Texture *texture = nullptr;
  for (unsigned int i = 0; i < geode->getNumDrawables(); ++i) {
    osg::StateSet *drawableState = geode->getDrawable(i)->getStateSet();
    if (drawableState) {
      texture = dynamic_cast<osg::Texture *>(
          drawableState->getTextureAttribute(0, osg::StateAttribute::TEXTURE));
      if (texture) break;
    }
  }

  osg::ref_ptr<osg::Uniform> hasTextureUniform =
      new osg::Uniform("hasTexture", texture != nullptr);
  stateSet->addUniform(hasTextureUniform.get());

  if (texture) {
    stateSet->setTextureAttributeAndModes(0, texture, osg::StateAttribute::ON);
    osg::ref_ptr<osg::Uniform> textureUniform =
        new osg::Uniform("texture0", (int)0);  // Textur Unit 0
    stateSet->addUniform(textureUniform.get());
  }

  // Optional: Deaktiviere das Schreiben in den Tiefenpuffer für die Umrandung,
  //           um Z-Fighting zu vermeiden.
  // osg::ref_ptr<osg::Depth> depth = new osg::Depth;
  // depth->setWriteMask(false);
  // stateSet->setAttribute(depth.get(), osg::StateAttribute::ON);
}

void createOutlineFX(osg::ref_ptr<osg::Geode> geode, const osg::Vec4 &outlineColor,
                     float lineWidth) {
  osg::ref_ptr<osgFX::Outline> outline = new osgFX::Outline;
  outline->setColor(outlineColor);
  outline->setWidth(lineWidth);
  outline->addChild(geode.get());
  for (int i = 0; geode->getNumParents(); ++i) {
    osg::ref_ptr<osg::Group> parent = geode->getParent(i)->asGroup();
    if (parent) {
      if (parent->containsNode(geode)) parent->replaceChild(geode, outline);
      break;
    }
  }
}

void printNodeInfo(osg::Node *node, int indent) {
  for (int i = 0; i < indent; ++i) std::cout << "  ";
  std::cout << "(" << node->className() << ") " << node->getName() << std::endl;
  osg::Group *group = node->asGroup();
  if (group) {
    for (unsigned int i = 0; i < group->getNumChildren(); ++i) {
      printNodeInfo(group->getChild(i), indent + 1);
    }
  }
  osg::Geode *geode = node->asGeode();
  if (geode) {
    for (unsigned int i = 0; i < geode->getNumDrawables(); ++i) {
      for (int j = 0; j <= indent; ++j) std::cout << "  ";
      std::cout << "  Drawable[" << i << "] : " << geode->getDrawable(i)->className()
                << std::endl;
      if (geode->getDrawable(i)->asGeometry()) {
        osg::StateSet *ss = geode->getDrawable(i)->getStateSet();
        if (ss) {
          for (int j = 0; j <= indent + 1; ++j) std::cout << "  ";
          std::cout << "    StateSet with " << ss->getTextureAttributeList().size()
                    << " texture units." << std::endl;
        }
      }
    }
    osg::StateSet *ss = geode->getStateSet();
    if (ss) {
      for (int j = 0; j <= indent; ++j) std::cout << "  ";
      std::cout << "  Geode StateSet with " << ss->getTextureAttributeList().size()
                << " texture units." << std::endl;
    }
  }
}

}  // namespace core::utils::osgUtils
