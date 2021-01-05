#ifndef COMSG_COVISE_LAUCH_OPTIONS_H
#define COMSG_COVISE_LAUCH_OPTIONS_H
#include <array>
namespace covise
{
enum class LaunchStyle
{
    Local,
    Partner,
    Host,
    Disconnect,
    LAST_DUMMY
};
constexpr int numLaunchStyles = static_cast < int>(LaunchStyle::LAST_DUMMY);
namespace detail
{
    constexpr std::array<const char *, numLaunchStyles> LaunchStyleNames{"Local", "Partner", "Host", "Disconnect"};
}
struct LaunchStyleNames{
    const char *operator[](LaunchStyle l)const{
        return detail::LaunchStyleNames[static_cast<int>(l)];
    }
};
constexpr LaunchStyleNames launchStyleNames;

} // namespace covise

#endif