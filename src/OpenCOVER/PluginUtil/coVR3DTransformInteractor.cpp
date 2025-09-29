/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR3DTransformInteractor.h"
#include <cmath>
#include <cover/coVRConfig.h>
#include <cover/coVRFileManager.h>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRPluginSupport.h>
#include <osg/AlphaFunc>
#include <osg/BlendFunc>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Group>
#include <osg/Image>
#include <osg/LineWidth>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/PolygonMode>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osgDB/FileUtils>
#include <osgDB/ReadFile>

using namespace opencover;

const osg::Vec4 xColor = osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f);         // Red
const osg::Vec4 yColor = osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f);         // Green
const osg::Vec4 zColor = osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f);         // Blue

const std::array<osg::Vec4, 3> axisColors = {xColor, yColor, zColor};

const osg::Vec4 hoverColor = osg::Vec4(0.6f, 0.6f, 0.0f, 1.0f);     // Yellow
const osg::Vec4 interactionColor = osg::Vec4(1.0, 0.3, 0.0, 1.0f);  // Orange

const osg::Vec4 centerColor = osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f);    // Gray
const osg::Vec4 uniformScaleColor = centerColor; 

constexpr float highlightFactor = 0.7f; // Factor to reduce color brightness
constexpr float arrowLength = 10.0f; 
constexpr float ringRadius = arrowLength * 0.7f;  // rotate gizmo ring radius
constexpr int maxSegments = 128; 

const std::array<osg::Vec3, 3> coordAxes = {osg::X_AXIS, osg::Y_AXIS, osg::Z_AXIS};

osg::Vec3 transformDeltaToLocalAxes(const osg::Vec3& worldDelta, const osg::Matrix& transformMatrix)
{
    // Get the rotation matrix (without translation)
    osg::Matrix rotationMatrix = transformMatrix;
    rotationMatrix.setTrans(osg::Vec3(0, 0, 0));
    
    // Invert to transform from world to local
    osg::Matrix invRotationMatrix;
    invRotationMatrix.invert(rotationMatrix);
    
    // Transform the delta vector
    return worldDelta * invRotationMatrix;
}

osg::Vec3 transformDeltaToWorldAxes(const osg::Vec3& localDelta, const osg::Matrix& transformMatrix)
{
    // Get the rotation matrix (without translation)
    osg::Matrix rotationMatrix = transformMatrix;
    rotationMatrix.setTrans(osg::Vec3(0, 0, 0));
    
    // Transform the delta vector from local to world
    return localDelta * rotationMatrix;
}

osg::Vec3 calculatePlaneIntersection(const osg::Matrix& pointerMat, 
                                                         const osg::Vec3& planeNormal, 
                                                         const osg::Vec3& planePoint)
{
    // Convert pointer matrix to object coordinates
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix pointerMat_o = pointerMat * w_to_o;
    
    // Extract ray origin and direction from pointer matrix
    osg::Vec3 rayOrigin = pointerMat_o.getTrans();
    osg::Vec3 rayDirection = osg::Y_AXIS * pointerMat_o; // Y-axis is forward
    rayDirection = rayDirection - rayOrigin; // Convert to direction vector
    rayDirection.normalize();
    
    float denominator = rayDirection * planeNormal;
    
    // Check if ray is parallel to plane
    if (fabs(denominator) < 1e-6)
    {
        // Ray is parallel to plane, return closest point on plane to ray origin
        osg::Vec3 toOrigin = rayOrigin - planePoint;
        float distance = toOrigin * planeNormal;
        return rayOrigin - planeNormal * distance;
    }
    
    // Calculate intersection parameter t
    osg::Vec3 toPlane = planePoint - rayOrigin;
    float t = (toPlane * planeNormal) / denominator;
    
    // Calculate intersection point
    osg::Vec3 intersection = rayOrigin + rayDirection * t;
    
    return intersection;
}

std::pair<osg::Vec3, osg::Vec3> getPerpendicularPlane(const osg::Vec3 &axis)
{
    osg::Vec3 u, v;
    if (fabs(axis.x()) > 0.9f) {
        u = osg::Vec3(0, 0, 1);  v = osg::Vec3(0, 1, 0);
    } else if (fabs(axis.y()) > 0.9f) {
        u = osg::Vec3(1, 0, 0);  v = osg::Vec3(0, 0, 1);
    } else {
        u = osg::Vec3(0, 1, 0);  v = osg::Vec3(1, 0, 0);
    }
    return {u, v};
}

coVR3DTransformInteractor::coVR3DTransformInteractor(float size, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority)
    : coVRIntersectionInteractor(size, type, iconName, interactorName, priority, false)
    , m_interactionMode(TRANSLATE)
    , m_root(new osg::MatrixTransform)
    , m_Arrows({detail::Arrow{m_root, osg::Vec3(1, 0, 0), xColor}, detail::Arrow{m_root, osg::Vec3(0, 1, 0), yColor}, detail::Arrow{m_root, osg::Vec3(0, 0, 1), zColor}})
{
    createGeometry();
    _standardHL = true;
}

coVR3DTransformInteractor::~coVR3DTransformInteractor()
{
    hide();
}

void coVR3DTransformInteractor::createGeometry()
{
    // Create root node for all gizmos
    m_root->setName("TransformInteractorRoot");
    
    createTranslateGizmos();
    CreateRotationTori(); 
    
    // Set this as the geometry node for the base class
    geometryNode = m_root;
    scaleTransform->addChild(geometryNode);

    // Set initial arrow colors
    updateArrowColors();
    createRotationVisualization();
    createModeCube();
    updateGizmoAppearance();
}


void setMaterial(osg::Geode* geode, const osg::Vec4 &color)
{
    osg::ref_ptr<osg::Material> material = new osg::Material;
    material->setDiffuse(osg::Material::FRONT_AND_BACK, color);
    material->setAmbient(osg::Material::FRONT_AND_BACK, color);
    // material->setSpecular(osg::Material::FRONT_AND_BACK, osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f));
    material->setShininess(osg::Material::FRONT_AND_BACK, 10.0f);
    material->setEmission(osg::Material::FRONT_AND_BACK, osg::Vec4f(0.1f, 0.1f, 0.1f, 1.0f));
    material->setColorMode(osg::Material::OFF);
    auto ss = geode->getOrCreateStateSet();
    ss->setAttributeAndModes(material);
    ss->setMode(GL_BLEND, osg::StateAttribute::ON);
    ss->setMode(GL_LIGHTING, osg::StateAttribute::ON);
}


detail::Arrow::Arrow(osg::Group *parent, const osg::Vec3 &direction, const osg::Vec4 &color)
: parent(parent)
, shaft(new osg::Geode)
, tip(new osg::Geode)
, tipTransform(new osg::MatrixTransform)
, shaftTransform(new osg::MatrixTransform)
, direction(direction)
{
    
    shaftTransform->setMatrix(osg::Matrix::translate(direction * 0.5f));
    tipTransform->setMatrix(osg::Matrix::translate(direction));

    // Add geodes to their respective transforms
    tipTransform->addChild(tip);
    shaftTransform->addChild(shaft);

    osg::Quat rot;
    rot.makeRotate(osg::Vec3(0, 0, 1), direction);
    
    // Arrow shaft (cylinder) - create at origin with base dimensions
    osg::ref_ptr<osg::Cylinder> cylinder = new osg::Cylinder(
        osg::Vec3(0, 0, 0),   // At origin - transform will position it
        arrowLength * 0.02f,         // Thickness
        arrowLength);         // Base length
    cylinder->setRotation(rot);
    cylinder->setName("ArrowShaft");

    osg::ref_ptr<osg::ShapeDrawable> shaftDrawable = new osg::ShapeDrawable(cylinder);
    shaftDrawable->setName("ArrowShaft");
    shaft->addDrawable(shaftDrawable);
    
    // Arrow head (cone) - create at origin
    osg::ref_ptr<osg::Cone> head = new osg::Cone(
        osg::Vec3(0, 0, 0),   // At origin - transform will position it
        arrowLength * 0.10f,         // Width
        arrowLength * 0.3f);         // Length
    head->setRotation(rot);
    head->setName("ArrowTip");

    osg::ref_ptr<osg::ShapeDrawable> headDrawable = new osg::ShapeDrawable(head);
    headDrawable->setName("ArrowTip");
    tip->addDrawable(headDrawable);
    
    // Set initial transforms
    // Shaft positioned at its center
    osg::Matrix shaftMatrix;
    shaftMatrix.makeTranslate(direction * arrowLength * 0.5f);
    shaftTransform->setMatrix(shaftMatrix);
    
    // Tip positioned at end of shaft
    osg::Matrix tipMatrix;
    tipMatrix.makeTranslate(direction * arrowLength);
    tipTransform->setMatrix(tipMatrix);

    setMaterial(shaft.get(), color);
    setMaterial(tip.get(), color);

    setVisible(true);
}

void detail::Arrow::setScale(float scale)
{
    currentScale = scale;
    
    // Scale the shaft length using shaftTransform
    osg::Matrix shaftMatrix;
    shaftMatrix.makeIdentity();
    
    // Create a scale matrix that only scales along the shaft direction
    osg::Vec3 scaleVec = direction * scale + osg::Vec3{ 1.0f, 1.0f, 1.0f } - direction;
    shaftMatrix.makeScale(scaleVec);
    
    // Also translate the shaft center to account for the scaling
    // When scaling, the center moves, so we need to adjust position
    osg::Vec3 newCenter = direction * arrowLength * 0.5f * scale;
    osg::Matrix translateMatrix;
    translateMatrix.makeTranslate(newCenter);
    
    // Combine scale and translation for shaft
    shaftMatrix = shaftMatrix * translateMatrix;
    shaftTransform->setMatrix(shaftMatrix);
    
    // Position tip at the end of the scaled shaft using tipTransform
    osg::Matrix tipMatrix;
    tipMatrix.makeTranslate(direction * arrowLength * scale);
    tipTransform->setMatrix(tipMatrix);
}

void detail::Arrow::setVisible(bool on)
{
    if (on)
    {
        if (parent && !parent->containsNode(shaftTransform))
            parent->addChild(shaftTransform);
        if (parent && !parent->containsNode(tipTransform))
            parent->addChild(tipTransform);
    }
    else
    {
        parent->removeChild(shaftTransform);
        parent->removeChild(tipTransform);
    }
}

int axisToIndex(const osg::Vec3 &axis)
{
    for (size_t i = 0; i < coordAxes.size(); i++)
    {
        if (axis == coordAxes[i]) return i;
    }
    return -1; // Invalid axis
}

void detail::Arrow::resetColor(bool uniformScale)
{
    if (uniformScale)
        setMaterial(tip, uniformScaleColor);
    else
        setMaterial(tip,  axisColors[axisToIndex(direction)]);

    setMaterial(shaft, axisColors[axisToIndex(direction)]);
}

std::vector<osg::Geode*> detail::Arrow::hit(const osg::Node *node) const
{
    std::vector<osg::Geode*> geodes;
    if (node == shaft.get())
        geodes.push_back(shaft.get());
    if (node == tip.get())
        geodes.push_back(tip.get());
    return geodes;
}

bool detail::Arrow::isTip(osg::Node* node) const
{
    return node == tip.get();
}

void coVR3DTransformInteractor::createTranslateGizmos()
{
    for (size_t i = 0; i < m_gizmoPlanes.size(); i++)
    {
        m_gizmoPlanes[i] = createPlane(coordAxes[2 - i], axisColors[2 - i]);
        m_root->addChild(m_gizmoPlanes[i]);
    }

    // Center sphere for free movement
    m_centerSphere = createSphere(arrowLength * 0.08f, centerColor);
    m_root->addChild(m_centerSphere);
}

void coVR3DTransformInteractor::CreateRotationTori()
{
    for (size_t i = 0; i < m_gizmoRings.size(); i++)
    {
        m_gizmoRings[i] = createRotationRing(coordAxes[i], axisColors[i]);
        m_root->addChild(m_gizmoRings[i]);
    }
}

osg::Texture2D* loadTexture(const std::string &filePath)
{
    auto texture = coVRFileManager::instance()->loadTexture(filePath.c_str());
    assert(texture && "Failed to load texture");
    texture->setWrap(osg::Texture::WRAP_S, osg::Texture::CLAMP_TO_EDGE);
    texture->setWrap(osg::Texture::WRAP_T, osg::Texture::CLAMP_TO_EDGE);
    texture->setFilter(osg::Texture::MIN_FILTER, osg::Texture::LINEAR);
    texture->setFilter(osg::Texture::MAG_FILTER, osg::Texture::LINEAR);
    return texture;
}

osg::Geode* coVR3DTransformInteractor::createPlane(const osg::Vec3 &normal, const osg::Vec4 &color)
{
    osg::Geode* geode = new osg::Geode;
    // Procedural 4-way move icon built from triangles (no texture) + invisible pick quad.
    float planeSize = arrowLength * 0.3f; // scale reference kept
    auto [u, v] = getPerpendicularPlane(normal);
    // Rotate the local 2D basis (u,v) by 45 degrees around the plane normal to twist the cross
    osg::Vec3 squareMinCorner = (u + v) * arrowLength * 0.3f;
    osg::Vec3 offset = normal * (arrowLength * 0.002f);
    osg::Vec3 iconCenter = squareMinCorner + (u + v) * (planeSize * 0.5f) + offset;
    {
        const float ang = osg::PI_4; // 45 deg
        osg::Vec3 uR =  u * cosf(ang) + v * sinf(ang);
        osg::Vec3 vR = -u * sinf(ang) + v * cosf(ang);
        u = uR; v = vR;
    }

    // Geometric parameters for "crossing double arrows" look:
    const float baseHalf        = planeSize;      // reference scale
    const float armHalfLength   = baseHalf * 0.60f; // length from center to start of arrow head
    const float shaftHalfWidth  = baseHalf * 0.12f; // slimmer shaft
    const float arrowHeadLength = baseHalf * 0.55f; // slightly longer heads
    const float arrowBaseFactor = 2.5f;             // how much wider arrow head base is vs shaft
    const float arrowBaseHalfWidth = shaftHalfWidth * arrowBaseFactor;

    // Invisible pick quad first (slightly behind)
    {
        osg::ref_ptr<osg::Geometry> pickGeom = new osg::Geometry;
        osg::ref_ptr<osg::Vec3Array> pVerts = new osg::Vec3Array;
        osg::ref_ptr<osg::Vec3Array> pNormals = new osg::Vec3Array;
        osg::ref_ptr<osg::Vec4Array> pColors = new osg::Vec4Array;
        float pickHalf = armHalfLength + arrowHeadLength;
        osg::Vec3 pickCenter = iconCenter - normal * (arrowLength * 0.001f);
        osg::Vec3 p0 = pickCenter - u * pickHalf - v * pickHalf;
        osg::Vec3 p1 = pickCenter + u * pickHalf - v * pickHalf;
        osg::Vec3 p2 = pickCenter + u * pickHalf + v * pickHalf;
        osg::Vec3 p3 = pickCenter - u * pickHalf + v * pickHalf;
        pVerts->push_back(p0); pVerts->push_back(p1); pVerts->push_back(p2); pVerts->push_back(p3);
        pNormals->push_back(normal);
        pColors->push_back(osg::Vec4(0,0,0,0));
        pickGeom->setVertexArray(pVerts.get());
        pickGeom->setNormalArray(pNormals.get(), osg::Array::BIND_OVERALL);
        pickGeom->setColorArray(pColors.get(), osg::Array::BIND_OVERALL);
        osg::ref_ptr<osg::DrawElementsUShort> pickIdx = new osg::DrawElementsUShort(GL_TRIANGLES);
        pickIdx->push_back(0); pickIdx->push_back(1); pickIdx->push_back(2);
        pickIdx->push_back(0); pickIdx->push_back(2); pickIdx->push_back(3);
        pickGeom->addPrimitiveSet(pickIdx.get());
        osg::StateSet *pss = pickGeom->getOrCreateStateSet();
        pss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
        pss->setMode(GL_BLEND, osg::StateAttribute::ON);
        // use alpha func to avoid sorting issues
        osg::ref_ptr<osg::AlphaFunc> alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.05f);
        pss->setAttributeAndModes(alphaFunc.get(), osg::StateAttribute::ON);
        pss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
        pss->removeAttribute(osg::StateAttribute::MATERIAL);
        geode->addDrawable(pickGeom.get());
    }

    osg::ref_ptr<osg::Geometry> iconGeom = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
    auto P = [&](float x, float y){ return iconCenter + u * x + v * y; };

    // Shaft rectangles (horizontal A-D, vertical E-H)
    osg::Vec2 A(-armHalfLength, -shaftHalfWidth);
    osg::Vec2 B( armHalfLength, -shaftHalfWidth);
    osg::Vec2 C( armHalfLength,  shaftHalfWidth);
    osg::Vec2 D(-armHalfLength,  shaftHalfWidth);
    osg::Vec2 E(-shaftHalfWidth, -armHalfLength);
    osg::Vec2 F( shaftHalfWidth, -armHalfLength);
    osg::Vec2 G( shaftHalfWidth,  armHalfLength);
    osg::Vec2 H(-shaftHalfWidth,  armHalfLength);

    // Arrowhead base lines are wider than shaft (use arrowBaseHalfWidth)
    osg::Vec2 RB1( armHalfLength, -arrowBaseHalfWidth); // right base lower
    osg::Vec2 RB2( armHalfLength,  arrowBaseHalfWidth); // right base upper
    osg::Vec2 LB1(-armHalfLength, -arrowBaseHalfWidth); // left base lower
    osg::Vec2 LB2(-armHalfLength,  arrowBaseHalfWidth); // left base upper
    osg::Vec2 UB1(-arrowBaseHalfWidth,  armHalfLength); // up base left
    osg::Vec2 UB2( arrowBaseHalfWidth,  armHalfLength); // up base right
    osg::Vec2 DB1(-arrowBaseHalfWidth, -armHalfLength); // down base left
    osg::Vec2 DB2( arrowBaseHalfWidth, -armHalfLength); // down base right

    // Tips
    osg::Vec2 tipR( armHalfLength + arrowHeadLength, 0.0f);
    osg::Vec2 tipL(-armHalfLength - arrowHeadLength, 0.0f);
    osg::Vec2 tipU(0.0f, armHalfLength + arrowHeadLength);
    osg::Vec2 tipD(0.0f, -armHalfLength - arrowHeadLength);

    verts->reserve(3 * 12);
    auto addTri = [&](const osg::Vec2 &p1, const osg::Vec2 &p2, const osg::Vec2 &p3){
        verts->push_back(P(p1.x(), p1.y()));
        verts->push_back(P(p2.x(), p2.y()));
        verts->push_back(P(p3.x(), p3.y()));
    };
    // Horizontal arm
    addTri(A, B, C); addTri(A, C, D);
    // Vertical arm
    addTri(E, F, G); addTri(E, G, H);
    // Arrow heads (widened bases)
    addTri(RB1, RB2, tipR); // right
    addTri(LB2, LB1, tipL); // left (order chosen for consistent normal orientation)
    addTri(UB2, UB1, tipU); // up
    addTri(DB1, DB2, tipD); // down

    normals->push_back(normal);
    iconGeom->setVertexArray(verts.get());
    iconGeom->setNormalArray(normals.get(), osg::Array::BIND_OVERALL);
    iconGeom->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLES, 0, verts->size()));

    osg::StateSet *iss = iconGeom->getOrCreateStateSet();
    iss->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    iss->setMode(GL_LIGHTING, osg::StateAttribute::ON);
    geode->addDrawable(iconGeom.get());

    setMaterial(geode, color); // enables highlight changes
    return geode;
}

osg::Geode* coVR3DTransformInteractor::createRotationRing(const osg::Vec3 &torusNormal, const osg::Vec4 &color)
{
    osg::Geode* geode = new osg::Geode;
    
    auto makeTorusGeom = [](const osg::Vec3 &torusNormal,
                            const osg::Vec3 &u, const osg::Vec3 &v,
                            int ringSegments, int tubeSegments,
                            float majorRadius, float minorRadius)
    {
        osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
        osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
        osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;

        for (int i = 0; i < ringSegments; ++i) {
            float theta = 2.0f * osg::PI * i / ringSegments;
            osg::Vec3 majorCirclePoint = u * cos(theta) * majorRadius + v * sin(theta) * majorRadius;
            osg::Vec3 tubeU = u * cos(theta) + v * sin(theta);
            osg::Vec3 tubeV = torusNormal;
            for (int j = 0; j < tubeSegments; ++j) {
                float phi = 2.0f * osg::PI * j / tubeSegments;
                osg::Vec3 tubeOffset = (tubeU * cos(phi) + tubeV * sin(phi)) * minorRadius;
                osg::Vec3 vertex = majorCirclePoint + tubeOffset;
                osg::Vec3 normal = tubeOffset;
                normal.normalize();
                vertices->push_back(vertex);
                normals->push_back(normal);
            }
        }

        osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(GL_TRIANGLES);
        for (int i = 0; i < ringSegments; ++i) {
            int nextI = (i + 1) % ringSegments;
            for (int j = 0; j < tubeSegments; ++j) {
                int nextJ = (j + 1) % tubeSegments;
                int v0 = i * tubeSegments + j;
                int v1 = i * tubeSegments + nextJ;
                int v2 = nextI * tubeSegments + nextJ;
                int v3 = nextI * tubeSegments + j;
                indices->push_back(v0); indices->push_back(v1); indices->push_back(v2);
                indices->push_back(v0); indices->push_back(v2); indices->push_back(v3);
            }
        }

        geometry->setVertexArray(vertices.get());
        geometry->setNormalArray(normals.get());
        geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
        geometry->addPrimitiveSet(indices.get());
        return geometry.release();
    };

    const int ringSegments = 64;
    const int tubeSegments = 16;
    const float majorRadius = ringRadius;
    const float minorRadius = arrowLength * 0.02f;

    // Build orthonormal basis in torus plane
    auto [u, v] = getPerpendicularPlane(torusNormal);


    // Visible torus (as before)
    osg::Geometry* visGeom = makeTorusGeom(torusNormal, u, v, ringSegments, tubeSegments, majorRadius, minorRadius);
    geode->addDrawable(visGeom);
    setMaterial(geode, color); // keep existing material for visible part

    // Invisible (by default) slightly larger torus used only for hit detection
    const float hitScale = 4.0f; // increase hit radius without changing visual size
    osg::ref_ptr<osg::Geometry> hitGeom = makeTorusGeom(torusNormal, u, v, ringSegments/4, tubeSegments/4, majorRadius, minorRadius * hitScale);

    // Color the hit geometry. When not compiling with SHOW_EXTENDED_ROTATION_RINGS
    // the hit geometry is fully transparent so it remains invisible but pickable.
    osg::ref_ptr<osg::Vec4Array> hitColor = new osg::Vec4Array;

    hitColor->push_back(osg::Vec4(0.0f, 0.0f, 0.0f, 0.0f)); // invisible
    hitGeom->setColorArray(hitColor.get(), osg::Array::BIND_OVERALL);

    osg::StateSet* ss = hitGeom->getOrCreateStateSet();
    ss->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    ss->setMode(GL_BLEND, osg::StateAttribute::ON);
    osg::ref_ptr<osg::AlphaFunc> alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.05f);
    ss->setAttributeAndModes(alphaFunc.get(), osg::StateAttribute::ON);
    // ensure it doesn't get a material/shading that would make it visible
    ss->removeAttribute(osg::StateAttribute::MATERIAL);

    geode->addDrawable(hitGeom.get());

    // Keep smooth shading off for the combined geode
    visGeom->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);

    return geode;
}

osg::Geode* coVR3DTransformInteractor::createSphere(float radius, const osg::Vec4 &color)
{
    osg::Geode* geode = new osg::Geode;
    
    osg::ref_ptr<osg::Sphere> sphere = new osg::Sphere(osg::Vec3(0, 0, 0), radius);
    osg::ref_ptr<osg::ShapeDrawable> sphereDrawable = new osg::ShapeDrawable(sphere);
    
    geode->addDrawable(sphereDrawable);
    setMaterial(geode, color);
    
    return geode;
}

void coVR3DTransformInteractor::highlightComponent(ComponentColor component, HighlightType type)
{
    if (!component || !component.color) return;
    
    osg::Vec4 color;
    switch (type)
    {
        case HIGHLIGHT_HOVER:
            color = hoverColor; // Use yellow hover color
            break;
        case HIGHLIGHT_ACTIVE:
            color = interactionColor; // Use orange interaction color
            break;
        case HIGHLIGHT_NONE:
        default:
            // Use mode-appropriate color
            if (m_gizmoMode == SCALE && m_scaleMode == SCALE_UNIFORM && component.nodes.size()==1 && isArrowTip(component.nodes.front()))
                color = uniformScaleColor;
            else
                color = *component.color; // Original color
            break;
    }
    for(auto *g: component.nodes)
        setMaterial(g, color);
}

coVR3DTransformInteractor::ComponentColor coVR3DTransformInteractor::getComponentFromNode(osg::Node* node)
{
    for (size_t i = 0; i < m_Arrows.size(); i++)
    {
        auto nodes = m_Arrows[i].hit(node);
        if(!nodes.empty()) return {nodes, &axisColors[i]};
        if(node == m_gizmoPlanes[i].get()) return {m_gizmoPlanes[i], &axisColors[2 - i]};
        if(node == m_gizmoRings[i].get()) return {m_gizmoRings[i], &axisColors[i]};
    }
    if (node == m_centerSphere.get()) return {m_centerSphere.get(), &centerColor};
    if (node == m_modeCubeGeode.get()|| 
       node->getParent(0)->getParent(0) == m_modeCubeIconsGroup.get()) //the icons)
        return {m_modeCubeGeode.get(), &m_cubeColor};
    // Remove scale box checks
    
    return {nullptr, nullptr};
}

coVR3DTransformInteractor::TransformMode coVR3DTransformInteractor::determineActiveMode(osg::Node* hitNode)
{
    if(std::find(m_gizmoRings.begin(), m_gizmoRings.end(), hitNode) != m_gizmoRings.end())
    {
        return ROTATE;
    }

    if (m_gizmoMode == SCALE && std::find_if(m_Arrows.begin(), m_Arrows.end(), [hitNode](const detail::Arrow& arrow){ return !arrow.hit(hitNode).empty(); }) != m_Arrows.end())
    {
        return SCALE;
    }
    return TRANSLATE; // Default to translate mode (shaft hit)
}

osg::Vec3 coVR3DTransformInteractor::determineActiveAxis(osg::Node* hitNode)
{
    // Check which component was hit
    
    if (hitNode == m_centerSphere.get()) return {1,1,1}; // XYZ axis
    for (size_t i = 0; i < m_Arrows.size(); i++)
    {
        if (m_Arrows[i].hit(hitNode).size() > 0 || m_gizmoRings[i].get() == hitNode) return coordAxes[i];
        auto [u, v] = getPerpendicularPlane(coordAxes[2 - i]);
        if(m_gizmoPlanes[i].get() == hitNode) return u + v; // Plane between two coordAxes
    }
    return {1,1,1}; // Default to free movement
}

static inline double determinant3x3(const osg::Matrix &m)
{
    return m(0,0)*(m(1,1)*m(2,2)-m(1,2)*m(2,1))
         - m(0,1)*(m(1,0)*m(2,2)-m(1,2)*m(2,0))
         + m(0,2)*(m(1,0)*m(2,1)-m(1,1)*m(2,0));
}

osg::Matrix extractRotation(const osg::Matrix &m)
{
    
    auto noTrans = m;
    noTrans.setTrans(osg::Vec3(0,0,0));
    // Extract 3x3 block

    // Quick check: if already orthonormal (rotation) then return early
    {
        auto noTranstransposed = noTrans;
        noTranstransposed.transpose(noTrans);
        auto ATA = noTranstransposed * noTrans;

        // check deviation from identity
        double maxDev = 0.0;
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
                maxDev = std::max(maxDev, fabs(ATA(r,c) - (r==c ? 1.0 : 0.0)));
        if (maxDev < 1e-6) // already rotation
            return noTrans;
    }

    // Newton iteration for polar decomposition: X_{k+1} = 0.5 * (X_k + inv(transpose(X_k)))
    auto x = noTrans;

    const int maxIter = 12;
    for (int iter = 0; iter < maxIter; ++iter)
    {
        auto invX = osg::Matrix::inverse(x);
        // transpose(invX)
        auto Tinv = invX;
        Tinv.transpose(invX);

        double maxDiff = 0.0;
        for (int r=0;r<3;++r)
            for (int c=0;c<3;++c)
            {
                double next = 0.5 * (x(r,c) + Tinv(r,c));
                maxDiff = std::max(maxDiff, fabs(next - x(r,c)));
                x(r,c) = next;
            }
        if (maxDiff < 1e-9) break;
    }

    // Fix handedness if necessary
    double d = determinant3x3(x);
    if (d < 0.0)
        for (int c=0;c<3;++c) x(0,c) = -x(0,c);
    return x;
}

void coVR3DTransformInteractor::updateTransform(const osg::Matrix &matrix)
{
    // build a matrix that keeps rotation (orthonormalized) and translation, but removes scale
    auto rot = extractRotation(matrix);
    auto rotAndTrans = rot;
    rotAndTrans.setTrans(matrix.getTrans());
    moveTransform->setMatrix(rotAndTrans);

    // Extract scale factors from the original matrix along the local axes
    m_scaleVector = (matrix * osg::Matrix::inverse(rot)).getScale();
    m_oldScaleVector = m_scaleVector;
}

void coVR3DTransformInteractor::updateScale(const osg::Vec3 &scale)
{
    m_scaleVector = scale;
}

float coVR3DTransformInteractor::calculateCurrentRotationAngle()
{
    // Get the current rotation of the object
    osg::Quat currentRotation = m_oldInteractorXformMat_o.getRotate();
    
    // Extract the rotation angle around this specific axis
    osg::Vec3 rotAxis;
    double rotAngle;
    currentRotation.getRotate(rotAngle, rotAxis);
    
    // Project the rotation onto the desired axis
    double projectedAngle = rotAngle * (rotAxis * m_activeAxis_world);

    return projectedAngle;
}

static inline float wrapAngle0to2Pi(float a)
{
    const float twoPi = 2.0f * osg::PI;
    a = fmodf(a, twoPi);
    if (a < 0.0f) a += twoPi;
    return a;
}

void coVR3DTransformInteractor::startInteraction()
{
    // Record interaction start time for toggle detection
    m_interactionStartTime = std::chrono::steady_clock::now();
    
    // Call parent implementation first
    // coVRIntersectionInteractor::startInteraction();
    
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix o_to_w = cover->getBaseMat();

    osg::Matrix hm = getPointerMat(); // hand matrix world coord
    osg::Matrix hm_o = hm * w_to_o; // hand matrix object coord
    m_startHandMat = hm; // Store the world coordinate start matrix
    m_invOldHandMat_o.invert(hm_o); // store the inv hand matrix
    
    m_oldInteractorXformMat_o = getMatrix();

    osg::Vec3 interPos = getMatrix().getTrans();
    // get diff between intersection point and sphere center
    m_diff = interPos - getHitPos();
    m_distance = (getHitPos() - hm_o.getTrans()).length();

    // Determine which axis/component was hit
    m_activeAxis = determineActiveAxis(_hitNode.get());
    // Transform to world coordinates
    osg::Matrix rotationOnly = m_oldInteractorXformMat_o;
    rotationOnly.setTrans(osg::Vec3(0, 0, 0));
    m_activeAxis_world = m_activeAxis * rotationOnly;
    m_activeAxis_world.normalize();

    m_activeComponent = getComponentFromNode(_hitNode.get());
    
    if(_hitNode == m_modeCubeGeode.get() || 
       _hitNode->getParent(0)->getParent(0) == m_modeCubeIconsGroup.get()) //the icons
    {
        // Cycle display (gizmo) mode and refresh cube icons/colors
        m_gizmoMode = static_cast<TransformMode>((m_gizmoMode + 1) % MODE_CUBE);
        updateModeCubeAppearance();
        updateGizmoAppearance();
        m_interactionMode = MODE_CUBE;
        return; // Don't start interaction when clicking mode cube
    }
    
    m_interactionMode = determineActiveMode(_hitNode.get());
    
    // Clear hover highlighting and apply active highlighting
    if (m_hoveredComponent && m_hoveredComponent != m_activeComponent)
    {
        highlightComponent(m_hoveredComponent, HIGHLIGHT_NONE);
        m_hoveredComponent = {};
    }
    
    if (m_activeComponent)
    {
        highlightComponent(m_activeComponent, HIGHLIGHT_ACTIVE);
    }
    
    // Store the initial absolute rotation angle for snapping offset
    if (m_interactionMode == ROTATE)
    {
        m_initialRotationAngle = calculateCurrentRotationAngle();
        m_firstRotationFrame = true;
        m_lastFrameAngle = 0.0f;
        m_accumulatedAngle = 0.0f;
        
        // Calculate the angle where the user grabbed the rotation handle
        osg::Vec3 rotCenter_o = m_oldInteractorXformMat_o.getTrans();
        osg::Vec3 grabIntersection = calculatePlaneIntersection(m_startHandMat, m_activeAxis_world, rotCenter_o);
        
        // Calculate vector from center to grab point
        osg::Vec3 grabVec = grabIntersection - rotCenter_o;
        
        // Project onto plane perpendicular to rotation axis
        grabVec = grabVec - m_activeAxis_world * (grabVec * m_activeAxis_world);
        
        if (grabVec.length() > 1e-6)
        {
            grabVec.normalize();
            auto [u, v] = getPerpendicularPlane(m_activeAxis);
            
            // Transform u and v to world coordinates
            osg::Matrix rotationOnly = m_oldInteractorXformMat_o;
            rotationOnly.setTrans(osg::Vec3(0, 0, 0));
            osg::Vec3 u_world = u * rotationOnly;
            osg::Vec3 v_world = v * rotationOnly;
            u_world.normalize();
            v_world.normalize();
            
            float u_component = grabVec * u_world;
            float v_component = grabVec * v_world;
            m_grabStartAngle = atan2(v_component, u_component);
            m_grabStartAngle = wrapAngle0to2Pi(m_grabStartAngle); // normalize to [0, 2π)
        }
        else
        {
            m_grabStartAngle = 0.0f;
        }
    }
    if (m_interactionMode == SCALE)
    {
        m_activeArrow = &m_Arrows[axisToIndex(m_activeAxis)];
    }
}

void coVR3DTransformInteractor::doInteraction()
{

    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    if (m_interactionMode == TransformMode::ROTATE)
    {
        handleRotation(currHandMat_o);
    }
    else if(m_interactionMode == TransformMode::SCALE)
    {
        handleScale(currHandMat_o);
    }
    else if(m_interactionMode == TransformMode::TRANSLATE)
    {
        handleTranslation(currHandMat_o);
    }
}

void coVR3DTransformInteractor::stopInteraction()
{
    // Check for toggle if we were in scale mode
    if (m_interactionMode == SCALE)
    {
        auto interactionDuration = std::chrono::steady_clock::now() - m_interactionStartTime;
        
        // If interaction was short enough, toggle scale mode
        if (interactionDuration < TOGGLE_CLICK_THRESHOLD)
        {
            toggleScaleMode();
            std::cerr << "Toggled to " << (m_scaleMode == SCALE_UNIFORM ? "uniform" : "per-axis") << " scaling mode" << std::endl;
        }
        else
        {
            m_oldScaleVector = m_scaleVector;             
        }
        
        for(auto &arrow : m_Arrows)
            arrow.setScale(1.0f);
        m_activeArrow = nullptr;
    }
    
    // Reset active component highlighting
    if (m_activeComponent)
    {
        highlightComponent(m_activeComponent, HIGHLIGHT_NONE);
        m_activeComponent = {};
    }
    
    // Update arrow colors based on current mode
    updateArrowColors();
    m_rotationVisualizationTransform->setNodeMask(0);
    if (cover->debugLevel(3))
        fprintf(stderr, "TransformInteractor::stopInteraction\n");
}

int coVR3DTransformInteractor::hit(vrui::vruiHit *hit)
{
    int result = coVRIntersectionInteractor::hit(hit);
    if (result && !isRunning()) // Only do hover effects when not actively interacting
    {
        auto hitComponent = getComponentFromNode(_hitNode.get());
        
        if (hitComponent != m_hoveredComponent)
        {
            // Clear previous hover
            if (m_hoveredComponent)
            {
                highlightComponent(m_hoveredComponent, HIGHLIGHT_NONE);
            }
            
            // Apply new hover
            m_hoveredComponent = hitComponent;
            if (m_hoveredComponent)
            {
                highlightComponent(m_hoveredComponent, HIGHLIGHT_HOVER);
            }
        }
    }
    
    return result;
}

void coVR3DTransformInteractor::createModeCube()
{
    m_modeCubeTransform = new osg::MatrixTransform;
    float displacement = arrowLength * 0.7f;
    osg::Matrix T; T.makeTranslate(displacement, displacement, displacement);
    m_modeCubeTransform->setMatrix(T);
    m_modeCubeGeode = new osg::Geode;
    m_modeCubeGeode->setName("ModeCube");
    float side = 2.0f; // full side length
    osg::ref_ptr<osg::Box> boxShape = new osg::Box(osg::Vec3(0,0,0), side);
    boxShape->setName("ModeCubeBox");
    osg::ref_ptr<osg::ShapeDrawable> boxDrawable = new osg::ShapeDrawable(boxShape);
    boxDrawable->setName("DrawableModeCubeBox");
    boxDrawable->setColor(osg::Vec4(0.9f,0.9f,0.9f,1.0f));
    m_modeCubeGeode->addDrawable(boxDrawable.get());
    m_modeCubeGeode->setUserValue("originalColor", osg::Vec4(1,1,1,1));
    m_modeCubeTransform->addChild(m_modeCubeGeode.get());
    m_root->addChild(m_modeCubeTransform.get());
    m_modeCubeIconsGroup = new osg::Group;
    m_modeCubeTransform->addChild(m_modeCubeIconsGroup.get());

    // Create icons for each face (6) oriented outward
    struct FaceInfo { osg::Vec3 normal; osg::Quat rot; osg::Vec3 pos; };
    std::vector<FaceInfo> faces = {
        { osg::Vec3(0,0,1), osg::Quat(), osg::Vec3(0,0, side*0.5f+0.01f)}, // front
        { osg::Vec3(0,0,-1), osg::Quat(osg::PI, osg::Vec3(0,1,0)), osg::Vec3(0,0,-side*0.5f-0.01f)}, // back
        { osg::Vec3(0,1,0), osg::Quat(-osg::PI_2, osg::Vec3(1,0,0)), osg::Vec3(0, side*0.5f+0.01f,0)}, // top
        { osg::Vec3(0,-1,0), osg::Quat(osg::PI_2, osg::Vec3(1,0,0)), osg::Vec3(0,-side*0.5f-0.01f,0)}, // bottom
        { osg::Vec3(1,0,0), osg::Quat(-osg::PI_2, osg::Vec3(0,1,0)), osg::Vec3(side*0.5f+0.01f,0,0)}, // right
        { osg::Vec3(-1,0,0), osg::Quat(osg::PI_2, osg::Vec3(0,1,0)), osg::Vec3(-side*0.5f-0.01f,0,0)}  // left
    };

    // Load PNG icons from "share/covise/icons/" folder;
    std::string iconNames[4] = {"moveAndRotate.png", "move.png", "rotate.png", "scale.png"};
    osg::ref_ptr<osg::Image> img;
    for (size_t i = 0; i < 4; i++)
        m_modeCubeFaceTextures[i] = loadTexture(iconNames[i]);
    osg::ref_ptr<osg::Geode> g = new osg::Geode;
        
    // textured quad centered at origin (z=0), size slightly smaller than face
    float h = side*0.4f;
    osg::ref_ptr<osg::Geometry> quad = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> verts = new osg::Vec3Array;
    verts->push_back(osg::Vec3(-h,-h,0)); verts->push_back(osg::Vec3(h,-h,0)); verts->push_back(osg::Vec3(h,h,0)); verts->push_back(osg::Vec3(-h,h,0));
    quad->setVertexArray(verts.get());
    osg::ref_ptr<osg::Vec2Array> texcoords = new osg::Vec2Array;
    texcoords->push_back(osg::Vec2(0,0)); texcoords->push_back(osg::Vec2(1,0)); texcoords->push_back(osg::Vec2(1,1)); texcoords->push_back(osg::Vec2(0,1));
    quad->setTexCoordArray(0, texcoords.get());
    quad->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
    g->addDrawable(quad.get());

    m_modeCubeFaceStateSet = g->getOrCreateStateSet();
    m_modeCubeFaceStateSet->setTextureAttributeAndModes(0, m_modeCubeFaceTextures[0].get(), osg::StateAttribute::ON);
    m_modeCubeFaceStateSet->setMode(GL_LIGHTING, osg::StateAttribute::OFF);
    osg::ref_ptr<osg::AlphaFunc> alphaFunc = new osg::AlphaFunc(osg::AlphaFunc::GREATER, 0.05f);
    m_modeCubeFaceStateSet->setAttributeAndModes(alphaFunc.get(), osg::StateAttribute::ON);
    m_modeCubeFaceStateSet->setMode(GL_BLEND, osg::StateAttribute::ON);

    for(const auto &f: faces)
    {
        osg::ref_ptr<osg::MatrixTransform> mt = new osg::MatrixTransform;
        osg::Matrix M; M.makeRotate(f.rot); M.setTrans(f.pos); mt->setMatrix(M);

        // offset quad slightly outward to avoid z-fighting
        mt->addChild(g.get());
        m_modeCubeIconsGroup->addChild(mt.get());
    }
    // Build initial icons
    updateModeCubeAppearance();
}

void coVR3DTransformInteractor::setMode(TransformMode mode)
{
    m_interactionMode = mode;
    updateArrowColors();
}

void coVR3DTransformInteractor::updateModeCubeAppearance()
{
    if (!m_modeCubeGeode) return;
    // Colors per mode
    switch(m_gizmoMode)
    {
        case TRANSFORM: m_cubeColor = osg::Vec4(0.9f,0.9f,0.9f,1.0f); break; // light gray
        case TRANSLATE: m_cubeColor = osg::Vec4(0.55f,0.35f,0.85f,1.0f); break; // purple
        case ROTATE:    m_cubeColor = osg::Vec4(1.0f,0.55f,0.15f,1.0f); break; // orange
        case SCALE:     m_cubeColor = osg::Vec4(0.1f,0.75f,0.65f,1.0f); break; // teal
        default: m_cubeColor = osg::Vec4(1,1,1,1); break;

    }
    setMaterial(m_modeCubeGeode.get(), m_cubeColor);
    m_modeCubeFaceStateSet->setTextureAttributeAndModes(0, m_modeCubeFaceTextures[m_gizmoMode].get(), osg::StateAttribute::ON);
}
void coVR3DTransformInteractor::updateGizmoAppearance()
{
    switch (m_gizmoMode)
    {
    case TRANSFORM:
    {
        for (size_t i = 0; i < m_Arrows.size(); i++)
        {
            m_Arrows[i].setVisible(true);
            m_gizmoRings[i]->setNodeMask(~0);
            m_gizmoPlanes[i]->setNodeMask(0);
        }
        
    }
    break;
    case TRANSLATE:
    {
        for (size_t i = 0; i < m_Arrows.size(); i++)
        {
            m_Arrows[i].setVisible(true);
            m_gizmoRings[i]->setNodeMask(0);
            m_gizmoPlanes[i]->setNodeMask(~0);
        }
    }
    break;
    case ROTATE:
    {
        for (size_t i = 0; i < m_Arrows.size(); i++)
        {
            m_Arrows[i].setVisible(false);
            m_gizmoRings[i]->setNodeMask(~0);
            m_gizmoPlanes[i]->setNodeMask(0);
        }
    }
    break;
    case SCALE:
    {
        for (size_t i = 0; i < m_Arrows.size(); i++)
        {
            m_Arrows[i].setVisible(true);
            m_gizmoRings[i]->setNodeMask(0);
            m_gizmoPlanes[i]->setNodeMask(0);
        }
    }
    break;
    default:
        break;
    }
}


void coVR3DTransformInteractor::miss()
{
    coVRIntersectionInteractor::miss();
    
    // Clear hover highlighting when not hovering over any component
    if (m_hoveredComponent && !isRunning())
    {
        highlightComponent(m_hoveredComponent, HIGHLIGHT_NONE);
        m_hoveredComponent = {};
    }
}

osg::Vec3 projectToAxis(const osg::Vec3 &point, osg::Vec3 axis)
{
    return osg::Vec3(point.x() * axis.x(), 
                     point.y() * axis.y(), 
                     point.z() * axis.z());

}

osg::Vec3 pointerDirection_o(const osg::Matrix &currHandMat_o)
{
    osg::Vec3 rayOrigin = currHandMat_o.getTrans();
    osg::Vec3 rayDirection = osg::Y_AXIS * currHandMat_o;
    rayDirection = rayDirection - rayOrigin;
    rayDirection.normalize();
    return rayDirection;
}

osg::Vec3 coVR3DTransformInteractor::handPosOnCurrentAxis_o(const osg::Matrix &currHandMat_o)
{
    auto pointerDir_o = pointerDirection_o(currHandMat_o);
    
    // Get hand pos in object coords
    auto currHandPos_o = currHandMat_o.getTrans();

    auto interPos = currHandPos_o + pointerDir_o * m_distance + m_diff;
    auto delta = interPos - m_oldInteractorXformMat_o.getTrans();
    
    // Transform the delta into the interactor's local coordinate system
    osg::Vec3 localDelta = transformDeltaToLocalAxes(delta, m_oldInteractorXformMat_o);
    
    // Project delta onto the active axis in local space
    return projectToAxis(localDelta, m_activeAxis);
}

void coVR3DTransformInteractor::handleTranslation(const osg::Matrix &currHandMat_o)
{

    osg::Vec3 newPos_o = handPosOnCurrentAxis_o(currHandMat_o);
    osg::Vec3 projectedWorldDelta = transformDeltaToWorldAxes(newPos_o, m_oldInteractorXformMat_o);
    auto newPos_w = m_oldInteractorXformMat_o.getTrans() + projectedWorldDelta;
    osg::Matrix interactorXformMat_o = m_oldInteractorXformMat_o;
    interactorXformMat_o.setTrans(newPos_w);

    moveTransform->setMatrix(interactorXformMat_o);
    m_posChanged = true;
    // _oldInteractorXformMat_o = interactorXformMat_o;
}

osg::Vec3 findClosestPointOnLine1ToLine2(
    const osg::Vec3& point1, const osg::Vec3& direction1,  // Line 1
    const osg::Vec3& point2, const osg::Vec3& direction2)  // Line 2
{
    osg::Vec3 w = point1 - point2;  // Vector between line origins
    
    float a = direction1 * direction1;  // dot(d1, d1)
    float b = direction1 * direction2;  // dot(d1, d2)
    float c = direction2 * direction2;  // dot(d2, d2)
    float d = direction1 * w;          // dot(d1, w)
    float e = direction2 * w;          // dot(d2, w)
    
    float denominator = a * c - b * b;
    
    // Check if lines are parallel
    if (fabs(denominator) < 1e-6) {
        // Lines are parallel, return the starting point of line 1
        return point1;
    }
    
    // Calculate parameter s for line 1
    float s = (b * e - c * d) / denominator;
    
    // Calculate the closest point on line 1
    osg::Vec3 closestPoint = point1 + direction1 * s;
    
    return closestPoint;
}

void coVR3DTransformInteractor::handleScale(const osg::Matrix &currHandMat_o)
{
    // Extract ray from current hand matrix
    osg::Vec3 rayOrigin = currHandMat_o.getTrans();
    osg::Vec3 rayDirection = pointerDirection_o(currHandMat_o);

    auto interactionPoint = findClosestPointOnLine1ToLine2(
        m_oldInteractorXformMat_o.getTrans(), m_activeAxis_world,
        rayOrigin, rayDirection);

    osg::Vec3 centerPos = m_oldInteractorXformMat_o.getTrans();
    auto currentDistance_o = (interactionPoint - centerPos).length();

    // Reference distance is the default arrow length
    float referenceDistance = arrowLength * _scale;
    float scaleFactor = currentDistance_o / referenceDistance;
    scaleFactor = std::max(0.1f, scaleFactor); // Prevent scaling below 10% of original size

    // Update visual feedback
    if (m_scaleMode == SCALE_UNIFORM)
    {
        // In uniform mode, scale all arrows together
        setUniformScale(scaleFactor);
        m_scaleVector = m_oldScaleVector * scaleFactor;
    }
    else
    {
        m_activeArrow->setScale(scaleFactor);
        auto scaleMatrix = osg::Matrix::scale(m_activeAxis * scaleFactor + osg::Vec3(1.0f, 1.0f, 1.0f) - m_activeAxis);
        m_scaleVector = scaleMatrix * m_oldScaleVector;
    }
    m_scaleChanged = true;
}

void coVR3DTransformInteractor::handleRotation(const osg::Matrix &currHandMat_o)
{
    // Get the center of rotation
    osg::Vec3 rotCenter_o = m_oldInteractorXformMat_o.getTrans();
    
    // Calculate intersection with rotation plane for START position
    osg::Vec3 startIntersection = calculatePlaneIntersection(m_startHandMat, m_activeAxis_world, rotCenter_o);
    
    // Calculate intersection with rotation plane for CURRENT position
    osg::Vec3 currentIntersection = calculatePlaneIntersection(getPointerMat(), m_activeAxis_world, rotCenter_o);
    
    // Check if current pointer is inside the interactor circle
    float interactorRadius = ringRadius * _scale;
    
    // Calculate distance in the PLANE
    osg::Vec3 vectorToIntersection = currentIntersection - rotCenter_o;
    osg::Vec3 vectorInPlane = vectorToIntersection - m_activeAxis_world * (vectorToIntersection * m_activeAxis_world);
    float distanceFromCenter = vectorInPlane.length();
    bool isInsideCircle = distanceFromCenter < interactorRadius;
    
    // Calculate vectors from center to intersection points
    osg::Vec3 startVec = startIntersection - rotCenter_o;
    osg::Vec3 currentVec = currentIntersection - rotCenter_o;
    
    // Project vectors onto plane perpendicular to rotation axis
    startVec = startVec - m_activeAxis_world * (startVec * m_activeAxis_world);
    currentVec = currentVec - m_activeAxis_world * (currentVec * m_activeAxis_world);

    // Check if vectors are valid
    if (startVec.length() < 1e-6 || currentVec.length() < 1e-6)
        return;
    
    // Normalize for angle calculation
    startVec.normalize();
    currentVec.normalize();
    
    // Calculate raw rotation angle using dot and cross product (always 0-180°)
    float cosAngle = osg::clampTo(startVec * currentVec, -1.0f, 1.0f);
    osg::Vec3 crossProduct = startVec ^ currentVec;
    float rawAngle = acos(cosAngle);
    
    // Determine rotation direction
    if ((crossProduct * m_activeAxis_world) < 0.0f)
    {
        rawAngle = -rawAngle;
    }
    
    // Calculate full continuous angle
    float continuousAngle = rawAngle;
    
    if (!m_firstRotationFrame)
    {
        // Calculate the difference from last frame
        float angleDiff = rawAngle - m_lastFrameAngle;
        
        // Check for 180° boundary crossing
        if (angleDiff > osg::PI)
        {
            // We crossed from positive to negative side (e.g., 179° to -179°)
            angleDiff -= 2.0f * osg::PI;
        }
        else if (angleDiff < -osg::PI)
        {
            // We crossed from negative to positive side (e.g., -179° to 179°)
            angleDiff += 2.0f * osg::PI;
        }
        
        // Accumulate the angle change
        m_accumulatedAngle += angleDiff;
        continuousAngle = m_accumulatedAngle;
    }
    else
    {
        // First frame - initialize
        m_accumulatedAngle = rawAngle;
        continuousAngle = rawAngle;
        m_firstRotationFrame = false;
    }
    
    // Store for next frame
    m_lastFrameAngle = rawAngle;
    
    // Apply snapping if inside circle
    float finalAngle = continuousAngle;
    if (isInsideCircle)
    {
        const float snapIncrement = 45.0f * osg::PI / 180.0f; 
        
        // Calculate the total rotation angle from initial position
        float totalAngle = m_initialRotationAngle + continuousAngle;
        
        // Snap the total angle to increments
        float snappedTotalAngle = round(totalAngle / snapIncrement) * snapIncrement;
        
        // Calculate the delta angle needed to reach the snapped position
        finalAngle = snappedTotalAngle - m_initialRotationAngle;
        
        // std::cerr << "Continuous: " << (continuousAngle * 180.0f / osg::PI) 
        //           << "°, Final: " << (finalAngle * 180.0f / osg::PI) << "°" << std::endl;
    }
    


    // Create rotation quaternion around the world rotation axis
    osg::Quat deltaRotation(finalAngle, m_activeAxis_world);
    
    // Get the current rotation and position
    osg::Vec3 translation = m_oldInteractorXformMat_o.getTrans();
    osg::Quat currentRotation = m_oldInteractorXformMat_o.getRotate();
    
    // Apply the delta rotation
    osg::Quat newRotation = currentRotation * deltaRotation;
    
    // Create new transformation matrix
    osg::Matrix interactorXformMat_o;
    interactorXformMat_o.makeRotate(newRotation);
    interactorXformMat_o.setTrans(translation);
    
    moveTransform->setMatrix(interactorXformMat_o);

    updateRotationVisualization(finalAngle);
    m_rotChanged = true;
}

void coVR3DTransformInteractor::toggleScaleMode()
{
    m_scaleMode = (m_scaleMode == SCALE_PER_AXIS) ? SCALE_UNIFORM : SCALE_PER_AXIS;
    updateArrowColors();
}

void coVR3DTransformInteractor::updateArrowColors()
{
    for(auto &arrow : m_Arrows)
        arrow.resetColor(m_gizmoMode == SCALE && m_scaleMode == SCALE_UNIFORM);
}

void coVR3DTransformInteractor::setUniformScale(float scale)
{
    // Scale all arrows together in uniform mode
    for(auto &arrow : m_Arrows)
        arrow.setScale(scale);
}

bool coVR3DTransformInteractor::isArrowTip(osg::Node* node)
{
    return std::find_if(m_Arrows.begin(), m_Arrows.end(), [node](const detail::Arrow& arrow){ return arrow.isTip(node); }) != m_Arrows.end();
}

osg::ref_ptr<osg::Geode> createRotationVisualization(osg::MatrixTransform *transform, bool cw, float size)
{
    osg::ref_ptr<osg::Geode> rotationVisualization = new osg::Geode;

    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec4Array> colors   = new osg::Vec4Array;
    
    // Slightly transparent gray color
    osg::Vec4 grayColor(0.7f, 0.7f, 0.7f, 0.4f); // Light gray with 40% opacity
    
    // Center vertex
    vertices->push_back(osg::Vec3(0, 0, 0));
    colors->push_back(grayColor);
    const float radius = ringRadius;
    
    // Create full circle vertices in XY plane
    if (cw) {
        for (int i = 0; i <= maxSegments; ++i) {
            float a = 2.0f * osg::PI * float(i) / float(maxSegments);
            vertices->push_back(osg::Vec3(std::cos(a) * radius, std::sin(a) * radius, 0.0f));
        }
    } else {
        for (int i = maxSegments; i >= 0; --i) {
            float a = 2.0f * osg::PI * float(i) / float(maxSegments);
            vertices->push_back(osg::Vec3(std::cos(a) * radius, std::sin(a) * radius, 0.0f));
        }
    }

    // Set up geometry
    geometry->setVertexArray(vertices.get());
    geometry->setColorArray(colors.get(), osg::Array::BIND_OVERALL);
    geometry->setDataVariance(osg::Object::DYNAMIC);
    colors->setDataVariance(osg::Object::DYNAMIC);
    geometry->setUseDisplayList(false);
    geometry->setUseVertexBufferObjects(true);

    rotationVisualization->addDrawable(geometry.get());
    transform->addChild(rotationVisualization.get());

    // State: transparent, no cull, and NO lighting/material 
    osg::StateSet* stateset = rotationVisualization->getOrCreateStateSet();
    stateset->setMode(GL_BLEND, osg::StateAttribute::ON);
    stateset->setRenderingHint(osg::StateSet::TRANSPARENT_BIN);
    stateset->setMode(GL_DEPTH_TEST, osg::StateAttribute::ON);
    stateset->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    stateset->setMode(GL_LIGHTING, osg::StateAttribute::OFF);  
    stateset->removeAttribute(osg::StateAttribute::MATERIAL);   // avoid inherited material

    osg::BlendFunc* blendFunc = new osg::BlendFunc;
    blendFunc->setFunction(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    stateset->setAttributeAndModes(blendFunc);

    transform->setNodeMask(0);
    return rotationVisualization;
}

void coVR3DTransformInteractor::createRotationVisualization()
{
    m_rotationVisualizationTransform = new osg::MatrixTransform;
    m_rotationVisualizationCCW = ::createRotationVisualization(m_rotationVisualizationTransform.get(), false, _scale);
    m_rotationVisualizationCW = ::createRotationVisualization(m_rotationVisualizationTransform.get(), true, _scale);

    m_root->addChild(m_rotationVisualizationTransform.get());
}

osg::Matrix coVR3DTransformInteractor::calculateRotationToActiveAxis()
{
    auto [u, v] = getPerpendicularPlane(m_activeAxis);

    // Step 1: rotate Z -> n
    osg::Quat q1; 
    q1.makeRotate(osg::Z_AXIS, m_activeAxis);
    osg::Matrix R1; 
    R1.makeRotate(q1);

    // Step 2: rotate around n to align +X after R1 with u
    osg::Vec3 x1 = osg::X_AXIS * R1;                 // +X after step 1
    x1 = (x1 - m_activeAxis * (x1 * m_activeAxis)); x1.normalize();        // project to plane
    float alpha = atan2((x1 ^ u) * m_activeAxis, x1 * u);       // signed angle around n

    osg::Matrix R2 = osg::Matrix::rotate(alpha, m_activeAxis);

    // final basis mapping matrix
    return R1 * R2;    
}

const std::array<osg::Matrix, 3> visRotationCounterClockWise{
    // Axis = X: XY -> YZ, +X→+Y, +Y→+Z, +Z→+X
    osg::Matrix::rotate(2.0f * osg::PI / 3.0f, osg::Vec3(1,1,1)),
    osg::Matrix::rotate(M_PI_2, osg::X_AXIS),
    osg::Matrix::identity()
};


void coVR3DTransformInteractor::updateRotationVisualization(float angle)
{
    bool ccw = (angle >= 0.0f);
    if (m_activeAxis == osg::Y_AXIS) //don't know why this flip is necessary, but it works
        ccw = !ccw;
    osg::Geometry* geometry = dynamic_cast<osg::Geometry*>(ccw ? m_rotationVisualizationCCW->getDrawable(0) : m_rotationVisualizationCW ->getDrawable(0));

    float normalizedAngle = std::min(1.0f, float(fabs(angle) / (2.0f * osg::PI)));
    int segmentsToShow = std::max(1, std::min(int(normalizedAngle * maxSegments) + 1, maxSegments));

    geometry->removePrimitiveSet(0, geometry->getNumPrimitiveSets());
    

    geometry->addPrimitiveSet(new osg::DrawArrays(GL_TRIANGLE_FAN, 0, segmentsToShow + 1));


    osg::Matrix rotateToGrabPos = osg::Matrix::rotate(m_grabStartAngle, osg::Z_AXIS);
    const osg::Matrix rotationToActiveAxis = visRotationCounterClockWise[axisToIndex(m_activeAxis)];
    osg::Matrix finalM = rotateToGrabPos * rotationToActiveAxis;

    m_rotationVisualizationTransform->setMatrix(finalM);
    m_rotationVisualizationTransform->setNodeMask(~0u);

    geometry->dirtyDisplayList();
    geometry->dirtyBound();
}
