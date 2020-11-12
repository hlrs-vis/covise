#ifndef VRB_REGISTRY_VARIABLE_H
#define VRB_REGISTRY_VARIABLE_H

#include <net/tokenbuffer_serializer.h>
#include <net/dataHandle.h>

#include <util/coExport.h>

#include <string>
#include <map>

namespace covise{
    class TokenBuffer;
} // namespace covise

namespace vrb
{
class regClass;
class SessionID;
class clientRegVar;

class VRBEXPORT regVarObserver
{
public:
    virtual void update(clientRegVar *theChangedVar) = 0;
};

class VRBEXPORT regVar
{
public:

    regVar(regClass* c, const std::string& name, const covise::DataHandle & value, bool isStatic = true);
    virtual ~regVar() = default;

    /// returns the value
    const covise::DataHandle &value() const;

    /// returns the class of this variable
    regClass *getClass();

    /// set value
    void setValue(const covise::DataHandle &v);

    /// returns true if this Var is static
    int isStatic();

    /// returns the Name
    const std::string &name() const;

    bool isDeleted();

    void setDeleted(bool isdeleted = true);

    template <typename Stream>
    void writeVar(Stream &file, bool sharedMap = false) const
    {
        file << m_name;
        if (sharedMap)
        {
            file << m_wholeMap.length();
            file.write(m_wholeMap.data(), m_wholeMap.length());
            file << (int)m_changedEtries.size();
            for (auto change : m_changedEtries)
            {
                file << change.first;
                file << change.second.length();
                file.write(change.second.data(), change.second.length());
            }
        }
        else
        {
            file << m_value.length();
            file.write(m_value.data(), m_value.length());
        }
    }
    void serialize(covise::TokenBuffer &tb) const;
    void deserialize(covise::TokenBuffer &tb);

protected:
    std::string m_name;
    regClass *m_class = nullptr;
    bool m_isStatic = false;
    bool m_isDeleted = false;
    covise::DataHandle m_value;
    //for SahredMaps
    typedef std::map<int, covise::DataHandle> EntryMap;
    covise::DataHandle m_wholeMap;
    EntryMap m_changedEtries;

    ///writes value to tb
    void sendValueChange(covise::TokenBuffer &tb);
    ///writes value to tb, in case of SahredMap also writes all changes
    void sendValue(covise::TokenBuffer &tb);
};

} // namespace vrb

namespace covise{

template <>
void serialize(TokenBuffer &tb, const vrb::regVar &value);

template <>
void deserialize(TokenBuffer &tb, vrb::regVar &value);
}

#endif