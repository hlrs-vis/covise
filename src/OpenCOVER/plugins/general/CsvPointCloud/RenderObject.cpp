#include "RenderObject.h"

void CsvRenderObject::setObjName(const std::string &name)
{
    m_objName = name;
}

const char *CsvRenderObject::getAttribute(const char *attrib) const
{
    if(attrib == std::string("OBJECTNAME"))
    {
        return m_objName.c_str();
    }
    return "";
}
