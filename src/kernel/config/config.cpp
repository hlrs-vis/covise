#include "config.h"

#include <memory>
#include <cstdlib>
#include <boost/asio/ip/host_name.hpp>

namespace {

class ConfigAccess
{
public:
    ConfigAccess();
private:
    std::unique_ptr<covise::config::Access> m_access;
};

ConfigAccess::ConfigAccess()
{
    const auto host_name = boost::asio::ip::host_name();
    m_access = std::make_unique<covise::config::Access>(host_name, std::string());
    if (auto covisedir = getenv("COVISEDIR"))
    {
        m_access->setPrefix(covisedir);
    }
}

}

static ConfigAccess covconfig_access;
