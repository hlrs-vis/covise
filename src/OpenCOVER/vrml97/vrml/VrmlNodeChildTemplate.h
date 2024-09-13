#ifndef COVER_VRMLFIELDTEMPLATE_H
#define COVER_VRMLFIELDTEMPLATE_H

#include "VrmlField.h"
#include "VrmlNode.h"
#include "VrmlScene.h"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <variant>

#include "VrmlSFBool.h"
#include "VrmlSFColor.h"
#include "VrmlSFColorRGBA.h"
#include "VrmlSFDouble.h"
#include "VrmlSFFloat.h"
#include "VrmlSFImage.h"
#include "VrmlSFInt.h"
#include "VrmlSFNode.h"
#include "VrmlSFRotation.h"
#include "VrmlSFString.h"
#include "VrmlSFTime.h"
#include "VrmlSFVec2d.h"
#include "VrmlSFVec3d.h"
#include "VrmlSFVec2f.h"
#include "VrmlSFVec3f.h"
#include "VrmlSFMatrix.h"
#include "VrmlMFBool.h"
#include "VrmlMFColor.h"
#include "VrmlMFColorRGBA.h"
#include "VrmlMFDouble.h"
#include "VrmlMFFloat.h"
#include "VrmlMFInt.h"
#include "VrmlMFNode.h"
#include "VrmlMFRotation.h"
#include "VrmlMFString.h"
#include "VrmlMFTime.h"
#include "VrmlMFVec2d.h"
#include "VrmlMFVec3d.h"
#include "VrmlMFVec2f.h"
#include "VrmlMFVec3f.h"
#include "vrmlexport.h"

namespace vrml{

class VrmlNodeChildTemplateImpl;

class VRMLEXPORT VrmlNodeChildTemplate : public VrmlNode
{

public:
    VrmlNodeChildTemplate(VrmlScene *scene);
    VrmlNodeChildTemplate(const VrmlNodeChildTemplate& other);
    ~VrmlNodeChildTemplate();

    template<typename T>
    T* registerField(const std::string& name, const std::function<void()> &updateCb =  std::function<void()>{});
    bool initialized(const std::string& name) const;
    bool allInitialized() const;

private:
    std::unique_ptr<VrmlNodeChildTemplateImpl> m_impl;
    void setField(const char *fieldName, const VrmlField &fieldValue) override;

};

#define VRMLNODECHILD2_TEMPLATE_DECL(type) \
extern template type VRMLEXPORT *VrmlNodeChildTemplate::registerField<type>(const std::string&, const std::function<void()>&);

VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFBool)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFColor)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFColorRGBA)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFDouble)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFFloat)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFInt)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFRotation)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFTime)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFVec2d)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFVec3d)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFVec2f)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFVec3f)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFImage)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFString)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFBool)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFColor)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFColorRGBA)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFDouble)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFFloat)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFInt)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFRotation)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFString)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFTime)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFVec2d)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFVec3d)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFVec2f)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFVec3f)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFNode)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlMFNode)
VRMLNODECHILD2_TEMPLATE_DECL(VrmlSFMatrix)


} // vrml


#endif // COVER_VRMLFIELDTEMPLATE_H