/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef TRANSFORM_INTERACTOR_H
#define TRANSFORM_INTERACTOR_H

#include <array>
#include <chrono> 
#include <vector>
#include <cover/coVRIntersectionInteractor.h>
#include <osg/AutoTransform>
#include <osg/Geode>
#include <osg/Geometry>
#include <osg/Material>
#include <osg/MatrixTransform>
#include <osg/ShapeDrawable>
#include <osg/Texture2D>
#include <osgText/Text>

namespace opencover
{
namespace detail
{
class PLUGIN_UTILEXPORT Arrow {
public:
    Arrow(osg::Group *parent, const osg::Vec3 &direction, const osg::Vec4 &color);
    void setScale(float scale);
    void setVisible(bool on);
    void resetColor(bool uniformScale);
    bool isTip(osg::Node* node) const;
    std::vector<osg::Geode*> hit(const osg::Node *node) const;
private:
    osg::ref_ptr<osg::Geode> shaft;
    osg::ref_ptr<osg::Geode> tip;
    osg::ref_ptr<osg::MatrixTransform> tipTransform, shaftTransform;
    osg::Vec3 direction;
    float currentScale = 1.0f;
    osg::Group* parent = nullptr;
};
}
class PLUGIN_UTILEXPORT coVR3DTransformInteractor : public coVRIntersectionInteractor
{
public:
    coVR3DTransformInteractor(float size, coInteraction::InteractionType type, const char *iconName, const char *interactorName, coInteraction::InteractionPriority priority);
    virtual ~coVR3DTransformInteractor();
    
    // Transform operations
    void updateTransform(const osg::Matrix &matrix);
    void updateScale(const osg::Vec3 &scale);
    const osg::Vec3 &getScale() const { return m_scaleVector; }
    
private:

    enum TransformMode
    {
        TRANSFORM, //actually translate and rotate
        TRANSLATE,
        ROTATE, 
        SCALE,
        MODE_CUBE
    };
    enum ScaleMode {
        SCALE_PER_AXIS,
        SCALE_UNIFORM
    }; 
    enum HighlightType
    {
        HIGHLIGHT_NONE,
        HIGHLIGHT_HOVER,
        HIGHLIGHT_ACTIVE
    };
    //todo: test replacement of m_root with moveTransform
    osg::ref_ptr<osg::MatrixTransform> m_root;

    std::array<detail::Arrow, 3> m_Arrows;

    detail::Arrow* m_activeArrow = nullptr;
    std::array<osg::ref_ptr<osg::Geode>, 3> m_gizmoPlanes; //xy,xz,yz
    std::array<osg::ref_ptr<osg::Geode>, 3> m_gizmoRings;
    osg::Vec3 m_scaleVector = osg::Vec3(1.0f, 1.0f, 1.0f);
    osg::Vec3 m_oldScaleVector = osg::Vec3(1.0f, 1.0f, 1.0f);

    osg::ref_ptr<osg::Geode> m_centerSphere;
    TransformMode m_interactionMode, m_gizmoMode = TRANSFORM;
    
    // Interaction state
    osg::Vec3 m_activeAxis;
    osg::Vec3 m_activeAxis_world;
    float m_initialRotationAngle = 0.0f;
    osg::Matrix m_startHandMat;
    osg::Matrix m_invOldHandMat_o;
    osg::Matrix m_oldInteractorXformMat_o;
    osg::Vec3 m_diff;
    float m_distance;
    // Rotation tracking for rotation visualization
    float m_lastFrameAngle = 0.0f;
    float m_accumulatedAngle = 0.0f;
    bool m_firstRotationFrame = true;
    float m_grabStartAngle = 0.0f;

    osg::ref_ptr<osg::DrawElementsUShort> m_rotationFan;
    struct ComponentColor
    {
        std::vector<osg::Geode*> nodes;   // one or many geodes making up the component
        const osg::Vec4 *color = nullptr; // original base color pointer (shared for all nodes)

        ComponentColor() = default;
        ComponentColor(osg::Geode *single, const osg::Vec4 *c) : nodes{single}, color(c) {}
        ComponentColor(const std::vector<osg::Geode*> &many, const osg::Vec4 *c) : nodes(many), color(c) {}

        bool operator==(const ComponentColor& other) const {
            return color == other.color && nodes == other.nodes;
        }
        bool operator!=(const ComponentColor& other) const { return !(*this == other); }
        explicit operator bool() const { return !nodes.empty(); }
    };
    // Current highlighting state
    ComponentColor m_hoveredComponent;
    ComponentColor m_activeComponent;
    
    
    // Scale mode functionality
    ScaleMode m_scaleMode = SCALE_UNIFORM;
    std::chrono::steady_clock::time_point m_interactionStartTime;
    static constexpr std::chrono::milliseconds TOGGLE_CLICK_THRESHOLD{200};
    
    // Color management methods
    void highlightComponent(ComponentColor component, HighlightType type);
    ComponentColor getComponentFromNode(osg::Node* node);
    

    // Interaction handling
    void createGeometry() override;
    void startInteraction() override;
    void doInteraction() override;
    void stopInteraction() override;
    
    // Override these methods for proper highlighting
    void miss() override;
    int hit(vrui::vruiHit *hit) override;
    void addIcon() override {} 
    void removeIcon() override {} 
    void resetState() override {} 

    // Geometry creation helpers
    void createTranslateGizmos();
    void CreateRotationTori();
    osg::Geode* createPlane(const osg::Vec3 &normal, const osg::Vec4 &color);
    osg::Geode* createRotationRing(const osg::Vec3 &axis, const osg::Vec4 &color);
    osg::Geode* createSphere(float radius, const osg::Vec4 &color);
    
    // Rotation visualization
    osg::ref_ptr<osg::Geode> m_rotationVisualizationCCW, m_rotationVisualizationCW;
    osg::ref_ptr<osg::MatrixTransform> m_rotationVisualizationTransform;
    
    void createRotationVisualization();
    // positions projected on the currently used rotation torus
    void updateRotationVisualization(float angle);
    osg::Matrix calculateRotationToActiveAxis();

    // Interaction helpers
    TransformMode determineActiveMode(osg::Node* hitNode);
    osg::Vec3 determineActiveAxis(osg::Node* hitNode);
    osg::Vec3 handPosOnCurrentAxis_o(const osg::Matrix &currHandMat_o);

    void handleScale(const osg::Matrix &currHandMat_o);
    void handleTranslation(const osg::Matrix &currHandMat_o);
    void handleRotation(const osg::Matrix &currHandMat_o);
    void handleScaling(const osg::Matrix &currHandMat_o);
    
    // Utility functions
    osg::Vec3 restrictToVisibleScene(const osg::Vec3 &pos);
    void snapTo45Degrees(osg::Matrix *matrix);
    void snapToDegrees(float degrees, osg::Matrix *matrix);
    
    void visualizeRotationTriangle(const osg::Vec3& center, 
                                 const osg::Vec3& startPos, 
                                 const osg::Vec3& currentPos, 
                                 const osg::Vec3& axis,
                                 bool isSnapping = false);
    osg::Quat snapQuaternionToAbsoluteIncrements(const osg::Quat& rotation, 
                                                 const osg::Vec3& axis, 
                                                 float incrementDegrees);
    float calculateCurrentRotationAngle();

    void handleArrowTipScale(const osg::Matrix &currHandMat_o);
    void resetArrowTipPosition();

    // Scale mode functionality methods
    void toggleScaleMode();
    void updateArrowColors();
    void setUniformScale(float scale);
    bool isArrowTip(osg::Node* node);  // Add this missing declaration

    // Mode cube (uniform cube with icon that changes per mode)
    void createModeCube();
    void setMode(TransformMode mode);
    void updateModeCubeAppearance();
    void updateGizmoAppearance();
    osg::ref_ptr<osg::MatrixTransform> m_modeCubeTransform; // placement
    osg::ref_ptr<osg::Geode> m_modeCubeGeode; // the cube itself
    osg::ref_ptr<osg::Geode> m_modeCubeIconGeode; // icon overlay (recreated per mode)
    osg::ref_ptr<osg::Group> m_modeCubeIconsGroup; // all face icons
    std::array<osg::ref_ptr<osg::Texture2D>, 4> m_modeCubeFaceTextures;
    osg::StateSet * m_modeCubeFaceStateSet = nullptr;
    osg::Vec4 m_cubeColor;
};

}

#endif
