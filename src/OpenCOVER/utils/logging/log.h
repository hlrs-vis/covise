#include "export.h"
#include <spdlog/spdlog.h>
#include "spdlog/sinks/stdout_color_sinks.h"
#include <string>

namespace opencover::utils::logging
{
/**
    Creates spdlog::logger shared_ptr with console color sink. Logger can be accessed via spdlog::get("<name>").
    For further configuration options look at https://github.com/gabime/spdlog
 */
inline auto LOGGINGUTIL create_logger(const std::string&name, const std::string& fmt="") {
    return spdlog::stdout_color_mt(name);
}
}
