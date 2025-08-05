/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#include "coVR3DTransformInteractor.h"
#include <cover/coVRPluginSupport.h>
#include <osg/ShapeDrawable>
#include <osg/Geometry>
#include <osg/Geode>
#include <osg/MatrixTransform>
#include <osg/Material>
#include <osg/PolygonMode>
#include <osg/LineWidth>
#include <osg/BlendFunc>
#include <cmath>
#include <cover/coVRNavigationManager.h>
#include <cover/coVRConfig.h>

using namespace opencover;

const osg::Vec4 xColor = osg::Vec4(1.0f, 0.0f, 0.0f, 1.0f);         // Red
const osg::Vec4 yColor = osg::Vec4(0.0f, 1.0f, 0.0f, 1.0f);         // Green
const osg::Vec4 zColor = osg::Vec4(0.0f, 0.0f, 1.0f, 1.0f);         // Blue

const osg::Vec4 hoverColor = osg::Vec4(0.6f, 0.6f, 0.0f, 1.0f);     // Yellow
const osg::Vec4 interactionColor = osg::Vec4(1.0, 0.3, 0.0, 1.0f);  // Orange

const osg::Vec4 centerColor = osg::Vec4(0.8f, 0.8f, 0.8f, 1.0f);    // Gray
const osg::Vec4 uniformScaleColor = centerColor; 

constexpr float highlightFactor = 0.7f; // Factor to reduce color brightness
constexpr float arrowLength = 0.8f; 
constexpr int maxSegments = 128; 

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
        u = osg::Vec3(0, 1, 0);  v = osg::Vec3(0, 0, 1);
    } else if (fabs(axis.y()) > 0.9f) {
        u = osg::Vec3(1, 0, 0);  v = osg::Vec3(0, 0, 1);
    } else {
        u = osg::Vec3(1, 0, 0);  v = osg::Vec3(0, 1, 0);
    }
    return {u, v};
}

coVR3DTransformInteractor::coVR3DTransformInteractor(float size, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority)
    : coVRIntersectionInteractor(size, type, iconName, interactorName, priority, false)
    , m_currentMode(TRANSLATE)
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
    m_root = new osg::MatrixTransform;
    m_root->setName("TransformInteractorRoot");
    
    createArrows();
    CreateRotationTori(); 
    
    // Set this as the geometry node for the base class
    geometryNode = m_root;
    scaleTransform->addChild(geometryNode);

    // Set initial arrow colors
    updateArrowColors();
    createRotationVisualization();
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
    geode->getOrCreateStateSet()->setAttributeAndModes(material);
    geode->getOrCreateStateSet()->setMode(GL_BLEND, osg::StateAttribute::ON);
    
    // Store original color as user data
    geode->setUserValue("originalColor", color);
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
    osg::Vec3 newCenter = direction * baseSize * 0.4f * scale;
    osg::Matrix translateMatrix;
    translateMatrix.makeTranslate(newCenter);
    
    // Combine scale and translation for shaft
    shaftMatrix = shaftMatrix * translateMatrix;
    shaftTransform->setMatrix(shaftMatrix);
    
    // Position tip at the end of the scaled shaft using tipTransform
    osg::Matrix tipMatrix;
    tipMatrix.makeTranslate(direction * baseSize * arrowLength * scale);
    tipTransform->setMatrix(tipMatrix);
}

detail::Arrow createArrow(const osg::Vec3 &direction, const osg::Vec4 &color, float size)
{
    detail::Arrow arrow;
    arrow.shaft = new osg::Geode;
    arrow.tip = new osg::Geode;
    arrow.tipTransform = new osg::MatrixTransform;
    arrow.shaftTransform = new osg::MatrixTransform;
    
    // Store parameters for later scaling
    arrow.direction = direction;
    arrow.baseSize = size;

    // Add geodes to their respective transforms
    arrow.tipTransform->addChild(arrow.tip);
    arrow.shaftTransform->addChild(arrow.shaft);
    
    osg::Quat rot;
    rot.makeRotate(osg::Vec3(0, 0, 1), direction);
    
    // Arrow shaft (cylinder) - create at origin with base dimensions
    osg::ref_ptr<osg::Cylinder> shaft = new osg::Cylinder(
        osg::Vec3(0, 0, 0),   // At origin - transform will position it
        size * 0.02f,         // Thickness
        size * arrowLength);         // Base length
    shaft->setRotation(rot);
    shaft->setName("ArrowShaft");

    osg::ref_ptr<osg::ShapeDrawable> shaftDrawable = new osg::ShapeDrawable(shaft);
    shaftDrawable->setName("ArrowShaft");
    arrow.shaft->addDrawable(shaftDrawable);
    
    // Arrow head (cone) - create at origin
    osg::ref_ptr<osg::Cone> head = new osg::Cone(
        osg::Vec3(0, 0, 0),   // At origin - transform will position it
        size * 0.12f,         // Width
        size * 0.2f);         // Length
    head->setRotation(rot);
    head->setName("ArrowTip");

    osg::ref_ptr<osg::ShapeDrawable> headDrawable = new osg::ShapeDrawable(head);
    headDrawable->setName("ArrowTip");
    arrow.tip->addDrawable(headDrawable);
    
    // Set initial transforms
    // Shaft positioned at its center
    osg::Matrix shaftMatrix;
    shaftMatrix.makeTranslate(direction * size * 0.4f);
    arrow.shaftTransform->setMatrix(shaftMatrix);
    
    // Tip positioned at end of shaft
    osg::Matrix tipMatrix;
    tipMatrix.makeTranslate(direction * size * arrowLength);
    arrow.tipTransform->setMatrix(tipMatrix);

    setMaterial(arrow.shaft, color);
    setMaterial(arrow.tip, color);
    return arrow;
}

void coVR3DTransformInteractor::createArrows()
{
    // Create arrows for each axis
    m_xArrow = createArrow(osg::Vec3(1, 0, 0), xColor, _interSize);
    m_yArrow = createArrow(osg::Vec3(0, 1, 0), yColor, _interSize);
    m_zArrow = createArrow(osg::Vec3(0, 0, 1), zColor, _interSize);
    
    // Create plane handles
    m_xyPlane = createPlane(osg::Vec3(0, 0, 1), osg::Vec4(0.5f, 0.5f, 0.0f, 0.3f));
    m_xzPlane = createPlane(osg::Vec3(0, 1, 0), osg::Vec4(0.5f, 0.0f, 0.5f, 0.3f));
    m_yzPlane = createPlane(osg::Vec3(1, 0, 0), osg::Vec4(0.0f, 0.5f, 0.5f, 0.3f));
    
    // Center sphere for free movement
    m_centerSphere = createSphere(_interSize * 0.05f, centerColor);

    //todo: test replacement with moveTransform
    m_root->addChild(m_xArrow.shaftTransform);
    m_root->addChild(m_yArrow.shaftTransform);
    m_root->addChild(m_zArrow.shaftTransform);
    m_root->addChild(m_xArrow.tipTransform);
    m_root->addChild(m_yArrow.tipTransform);
    m_root->addChild(m_zArrow.tipTransform);
    m_root->addChild(m_xyPlane);
    m_root->addChild(m_xzPlane);
    m_root->addChild(m_yzPlane);
    m_root->addChild(m_centerSphere);
    
}

void coVR3DTransformInteractor::CreateRotationTori()
{
    // Create rotation rings for each axis
    m_xRotRing = createRotationRing(osg::Vec3(1, 0, 0), xColor);
    m_yRotRing = createRotationRing(osg::Vec3(0, 1, 0), yColor);
    m_zRotRing = createRotationRing(osg::Vec3(0, 0, 1), zColor);

    m_root->addChild(m_xRotRing);
    m_root->addChild(m_yRotRing);
    m_root->addChild(m_zRotRing);
    
}

osg::Geode* coVR3DTransformInteractor::createPlane(const osg::Vec3 &normal, const osg::Vec4 &color)
{
    osg::Geode* geode = new osg::Geode;
    
    // Create a small square plane handle
    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    
    float planeSize = _interSize * 0.15f;
    // Calculate plane vectors perpendicular to normal
    auto [u, v] = getPerpendicularPlane(normal);
    
    osg::Vec3 center = (u + v) * _interSize * 0.2f;
    
    vertices->push_back(center);
    vertices->push_back(center + u * planeSize);
    vertices->push_back(center + u * planeSize + v * planeSize);
    vertices->push_back(center + v * planeSize);
    
    geometry->setVertexArray(vertices);
    geometry->addPrimitiveSet(new osg::DrawArrays(GL_QUADS, 0, 4));
    
    geode->addDrawable(geometry);
    setMaterial(geode, color);
    
    return geode;
}

osg::Geode* coVR3DTransformInteractor::createRotationRing(const osg::Vec3 &axis, const osg::Vec4 &color)
{
    osg::Geode* geode = new osg::Geode;
    
    osg::ref_ptr<osg::Geometry> geometry = new osg::Geometry;
    osg::ref_ptr<osg::Vec3Array> vertices = new osg::Vec3Array;
    osg::ref_ptr<osg::Vec3Array> normals = new osg::Vec3Array;
    
    const int ringSegments = 64;  // Number of segments around the ring
    const int tubeSegments = 16;  // Number of segments around the tube cross-section
    const float majorRadius = _interSize * 0.6f;  // Ring radius
    const float minorRadius = _interSize * 0.03f; // Tube thickness radius
    
    // Create a proper coordinate system for the torus
    osg::Vec3 torusNormal = axis;  // The axis is the normal to the torus plane
    torusNormal.normalize();
    
    // Create two perpendicular vectors in the torus plane
    osg::Vec3 u, v;
    if (fabs(torusNormal.x()) < 0.9f) {
        u = torusNormal ^ osg::Vec3(1, 0, 0);
    } else {
        u = torusNormal ^ osg::Vec3(0, 1, 0);
    }
    u.normalize();
    v = torusNormal ^ u;
    v.normalize();
    
    // Generate torus vertices and normals
    for (int i = 0; i < ringSegments; ++i) {
        float theta = 2.0f * osg::PI * i / ringSegments;  // Angle around the major radius
        
        // Point on the major radius circle (center of the tube at this position)
        osg::Vec3 majorCirclePoint = u * cos(theta) * majorRadius + v * sin(theta) * majorRadius;
        
        // Direction vectors for the tube cross-section
        osg::Vec3 tubeU = u * cos(theta) + v * sin(theta);  // Radial direction from torus center
        osg::Vec3 tubeV = torusNormal;                       // Axial direction
        
        // Generate the tube cross-section
        for (int j = 0; j < tubeSegments; ++j) {
            float phi = 2.0f * osg::PI * j / tubeSegments;  // Angle around the tube cross-section
            
            // Point on the tube circumference
            osg::Vec3 tubeOffset = (tubeU * cos(phi) + tubeV * sin(phi)) * minorRadius;
            osg::Vec3 vertex = majorCirclePoint + tubeOffset;
            
            // Normal points outward from the tube center
            osg::Vec3 normal = tubeOffset;
            normal.normalize();
            
            vertices->push_back(vertex);
            normals->push_back(normal);
        }
    }
    
    // Generate triangle indices
    osg::ref_ptr<osg::DrawElementsUInt> indices = new osg::DrawElementsUInt(GL_TRIANGLES);
    
    for (int i = 0; i < ringSegments; ++i) {
        int nextI = (i + 1) % ringSegments;
        
        for (int j = 0; j < tubeSegments; ++j) {
            int nextJ = (j + 1) % tubeSegments;
            
            // Calculate vertex indices for the current quad
            int v0 = i * tubeSegments + j;
            int v1 = i * tubeSegments + nextJ;
            int v2 = nextI * tubeSegments + nextJ;
            int v3 = nextI * tubeSegments + j;
            
            // Create two triangles for each quad
            // Triangle 1: v0 -> v1 -> v2
            indices->push_back(v0);
            indices->push_back(v1);
            indices->push_back(v2);
            
            // Triangle 2: v0 -> v2 -> v3
            indices->push_back(v0);
            indices->push_back(v2);
            indices->push_back(v3);
        }
    }
    
    geometry->setVertexArray(vertices);
    geometry->setNormalArray(normals);
    geometry->setNormalBinding(osg::Geometry::BIND_PER_VERTEX);
    geometry->addPrimitiveSet(indices);
    
    // Enable smooth shading
    geometry->getOrCreateStateSet()->setMode(GL_CULL_FACE, osg::StateAttribute::OFF);
    
    geode->addDrawable(geometry);
    setMaterial(geode, color);
    
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
    if (!component.node || !component.color) return;
    
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
            if (m_scaleMode == SCALE_UNIFORM && isArrowTip(component.node))
            {
                color = uniformScaleColor;
            }
            else
            {
                color = *component.color; // Original color
            }
            break;
    }
    
    setMaterial(component.node, color);
}

coVR3DTransformInteractor::ComponentColor coVR3DTransformInteractor::getComponentFromNode(osg::Node* node)
{
    // Check which component was hit
    if (node == m_xArrow.shaft.get()) return {m_xArrow.shaft.get(), &xColor};
    if (node == m_yArrow.shaft.get()) return {m_yArrow.shaft.get(), &yColor};
    if (node == m_zArrow.shaft.get()) return {m_zArrow.shaft.get(), &zColor};
    if (node == m_xArrow.tip.get()) return {m_xArrow.tip.get(), &xColor};
    if (node == m_yArrow.tip.get()) return {m_yArrow.tip.get(), &yColor};
    if (node == m_zArrow.tip.get()) return {m_zArrow.tip.get(), &zColor};
    if (node == m_xyPlane.get()) return {m_xyPlane.get(), &xColor};
    if (node == m_xzPlane.get()) return {m_xzPlane.get(), &xColor};
    if (node == m_yzPlane.get()) return {m_yzPlane.get(), &yColor};
    if (node == m_centerSphere.get()) return {m_centerSphere.get(), &centerColor};
    if (node == m_xRotRing.get()) return {m_xRotRing.get(), &xColor};
    if (node == m_yRotRing.get()) return {m_yRotRing.get(), &yColor};
    if (node == m_zRotRing.get()) return {m_zRotRing.get(), &zColor};
    // Remove scale box checks
    
    return {nullptr, nullptr};
}

coVR3DTransformInteractor::TransformMode coVR3DTransformInteractor::determineActiveMode(osg::Node* hitNode)
{
    if (hitNode == m_xRotRing.get() || hitNode == m_yRotRing.get() || hitNode == m_zRotRing.get())
    {
        return ROTATE;
    }
    if (hitNode == m_xArrow.tip.get() || hitNode == m_yArrow.tip.get() || hitNode == m_zArrow.tip.get())
    {
        return SCALE;
    }

    return TRANSLATE; // Default to translate mode (shaft hit)
}

osg::Vec3 coVR3DTransformInteractor::determineActiveAxis(osg::Node* hitNode)
{
    // Check which component was hit
    if (hitNode == m_xArrow.shaft.get()) return osg::X_AXIS;
    if (hitNode == m_yArrow.shaft.get()) return osg::Y_AXIS;
    if (hitNode == m_zArrow.shaft.get()) return osg::Z_AXIS;
    if (hitNode == m_xArrow.tip.get()) return osg::X_AXIS;
    if (hitNode == m_yArrow.tip.get()) return osg::Y_AXIS;
    if (hitNode == m_zArrow.tip.get()) return osg::Z_AXIS;
    if (hitNode == m_xyPlane.get()) return osg::X_AXIS + osg::Y_AXIS; // XY plane
    if (hitNode == m_xzPlane.get()) return osg::X_AXIS + osg::Z_AXIS; // XZ plane
    if (hitNode == m_yzPlane.get()) return osg::Y_AXIS + osg::Z_AXIS; // YZ plane
    if (hitNode == m_centerSphere.get()) return osg::X_AXIS + osg::Y_AXIS + osg::Z_AXIS; // XYZ axis
    if (hitNode == m_xRotRing.get()) return osg::X_AXIS;
    if (hitNode == m_yRotRing.get()) return osg::Y_AXIS;
    if (hitNode == m_zRotRing.get()) return osg::Z_AXIS;
    // Remove scale box checks

    return osg::X_AXIS + osg::Y_AXIS + osg::Z_AXIS; // Default to free movement
}

void coVR3DTransformInteractor::updateTransform(const osg::Matrix &matrix)
{
    moveTransform->setMatrix(matrix);
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
    

    
    m_currentMode = determineActiveMode(_hitNode.get());
    
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
    if (m_currentMode == ROTATE)
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
        std::cerr << "Grab start angle: " << m_grabStartAngle * 180.0f / osg::PI << std::endl;
    }
    if (m_currentMode == SCALE)
    {
        m_activeArrow = m_activeAxis == osg::X_AXIS ? &m_xArrow :
                        m_activeAxis == osg::Y_AXIS ? &m_yArrow : &m_zArrow;
        
    }
}

void coVR3DTransformInteractor::doInteraction()
{

    osg::Matrix currHandMat = getPointerMat();
    osg::Matrix w_to_o = cover->getInvBaseMat();
    osg::Matrix currHandMat_o = currHandMat * w_to_o;

    if (m_currentMode == TransformMode::ROTATE)
    {
        handleRotation(currHandMat_o);
    }
    else if(m_currentMode == TransformMode::SCALE)
    {
        handleScale(currHandMat_o);
    }
    else
    {
        handleTranslation(currHandMat_o);
    }
}

void coVR3DTransformInteractor::stopInteraction()
{
    // Check for toggle if we were in scale mode
    if (m_currentMode == SCALE)
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
            // Apply the scale transformation
            if (m_scaleMode == SCALE_UNIFORM)
            {
                // Apply uniform scale to all axes
                float scaleFactor = m_activeArrow->currentScale;
                m_scaleVector.x() *= scaleFactor;
                m_scaleVector.y() *= scaleFactor;
                m_scaleVector.z() *= scaleFactor;
            }
            else
            {
                // Apply per-axis scale
                auto scaleMatrix = osg::Matrix::scale(m_activeAxis * m_activeArrow->currentScale + osg::Vec3(1.0f, 1.0f, 1.0f) - m_activeAxis);
                m_scaleVector = scaleMatrix * m_scaleVector;
            }
        }
        
        // Reset all arrows to original scale
        m_xArrow.setScale(1.0f);
        m_yArrow.setScale(1.0f);
        m_zArrow.setScale(1.0f);
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

    updateTransform(interactorXformMat_o);
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
    float referenceDistance = _interSize * arrowLength * _scale;
    float scaleFactor = currentDistance_o / referenceDistance;
    scaleFactor = std::max(0.1f, scaleFactor); // Prevent scaling below 10% of original size

    // Update visual feedback
    if (m_scaleMode == SCALE_UNIFORM)
    {
        // In uniform mode, scale all arrows together
        setUniformScale(scaleFactor);
    }
    else
    {
        m_activeArrow->setScale(scaleFactor);
    }
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
    float interactorRadius = _interSize * 0.6f * scaleTransform->getMatrix().getScale().x();
    
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
    
    // Update the transformation
    updateTransform(interactorXformMat_o);


    updateRotationVisualization(finalAngle);

}

void coVR3DTransformInteractor::toggleScaleMode()
{
    m_scaleMode = (m_scaleMode == SCALE_PER_AXIS) ? SCALE_UNIFORM : SCALE_PER_AXIS;
    updateArrowColors();
}

void coVR3DTransformInteractor::updateArrowColors()
{
    if (m_scaleMode == SCALE_UNIFORM)
    {
        // Set all arrow tips to gray
        setMaterial(m_xArrow.tip, uniformScaleColor);
        setMaterial(m_yArrow.tip, uniformScaleColor);
        setMaterial(m_zArrow.tip, uniformScaleColor);
    }
    else
    {
        // Restore original colors
        setMaterial(m_xArrow.tip, xColor);
        setMaterial(m_yArrow.tip, yColor);
        setMaterial(m_zArrow.tip, zColor);
    }
    
    // Keep shaft colors unchanged
    setMaterial(m_xArrow.shaft, xColor);
    setMaterial(m_yArrow.shaft, yColor);
    setMaterial(m_zArrow.shaft, zColor);
}

void coVR3DTransformInteractor::setUniformScale(float scale)
{
    // Scale all arrows together in uniform mode
    m_xArrow.setScale(scale);
    m_yArrow.setScale(scale);
    m_zArrow.setScale(scale);
}

bool coVR3DTransformInteractor::isArrowTip(osg::Node* node)
{
    return (node == m_xArrow.tip.get() || node == m_yArrow.tip.get() || node == m_zArrow.tip.get());
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
    const float radius = size * 0.55f;
    
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
    m_rotationVisualizationCCW = ::createRotationVisualization(m_rotationVisualizationTransform.get(), false, _interSize);
    m_rotationVisualizationCW = ::createRotationVisualization(m_rotationVisualizationTransform.get(), true, _interSize);

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

int axisToIndex(const osg::Vec3 &axis)
{
    if (axis == osg::X_AXIS) return 0;
    if (axis == osg::Y_AXIS) return 1;
    if (axis == osg::Z_AXIS) return 2;
    return -1; // Invalid axis
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
