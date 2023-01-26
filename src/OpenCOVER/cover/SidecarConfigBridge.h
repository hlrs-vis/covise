#include <OpenConfig/access.h>
#include <string>
#include <set>

#include <OpenConfig/covconfig/detail/toml/toml.hpp> // in order to avoid another copy of toml.hpp

class SidecarConfigBridge: public opencover::config::Bridge
{
public:
    SidecarConfigBridge(const std::string &file, bool saveOnDestroy = false);
    ~SidecarConfigBridge();

    bool wasChanged(const opencover::config::ConfigBase *entry) override;

    struct Key
    {
        Key();
        Key(const opencover::config::ConfigBase *entry);

        std::string path;
        std::string section;
        std::string name;

        bool operator<(const Key &o) const;
        std::string configSection() const;
        std::string configName() const;
    };

private:
    template<class V>
    bool toOrFromToml(const opencover::config::ConfigBase *entry, const Key &key, bool seen);

    std::string m_file;
    toml::table m_toml;
    bool m_modified = false;
    bool m_save = false;

    std::set<Key> m_seen;
};
