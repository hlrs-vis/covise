#ifndef VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H
#define VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H

#include <array>
#include <QMetaType>
#include <vrb/ProgramType.h>
namespace vrb
{
namespace launcher
{

enum class LaunchType
{
    LAUNCH,
    TERMINATE

};


template <typename Stream>
Stream &operator<<(Stream &s, LaunchType t)
{
    s << static_cast<int>(t);
    return s;
}

template <typename Stream>
Stream &operator>>(Stream &s, LaunchType &t)
{
    int tt;
    s >> tt;
    t = static_cast<LaunchType>(tt);
    return s;
}

} // namespace launcher
} // namespace vrb

Q_DECLARE_METATYPE(vrb::Program);
Q_DECLARE_METATYPE(vrb::launcher::LaunchType);

#endif // !VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H