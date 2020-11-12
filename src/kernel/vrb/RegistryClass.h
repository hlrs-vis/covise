#ifndef VRB_REGISTRY_CLASS_H
#define VRB_REGISTRY_CLASS_H

#include <net/tokenbuffer_serializer.h>
#include <util/coExport.h>

#include <map>
#include <memory>
#include <string>

namespace covise{
    class TokenBuffer;
}

namespace vrb{
constexpr char sharedMapName[] = "SharedMap";

class regVar;

class VRBEXPORT regClass
{
public:
    typedef std::map<const std::string, std::shared_ptr<regVar>> Variables;
    typedef Variables::const_iterator Iter;
    regClass(const std::string &name = "", int ID = -1);
    virtual ~regClass() = default;

    Iter begin();
    Iter end();

    /// get Class ID
	int getID();

	void setID(int id);

    const std::string &name() const;
    bool isMap() const;
    ///creates a  a regvar entry  in the map
    void append(regVar* var);

    /// getVariableEntry, returns NULL if not found
	regVar* getVar(const std::string& n);

    /// remove a Variable
	void deleteVar(const std::string& n);

    /// remove some Variables
	void deleteAllNonStaticVars();

	bool isDeleted();

	void setDeleted(bool isdeleted = true);


	void serialize(covise::TokenBuffer& file) const;
	void deserialize(covise::TokenBuffer& file);

    virtual std::shared_ptr<regVar> createVar(const std::string &m_name, const covise::DataHandle &value) = 0;

protected:
    std::string m_name;
    int m_classID = -1;
    bool m_isDel = false;
    Variables m_variables;
};

}//vrb

namespace covise{
    template <>
void serialize(TokenBuffer &tb, const vrb::regClass &value);

template <>
void deserialize(TokenBuffer &tb, vrb::regClass &value);
}


#endif // !VRB_REGISTRY_CLASS_H