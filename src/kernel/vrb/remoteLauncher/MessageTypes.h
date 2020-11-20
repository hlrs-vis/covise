#ifndef VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H
#define VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H

#include "export.h"
#include <array>
#include <QMetaType>
namespace vrb
{
namespace launcher
{

enum class LaunchType
{
    LAUNCH,
    TERMINATE

};

enum class Program
{
    COVISE,
    COVER,
    DUMMY
};
struct REMOTELAUNCHER_EXPORT ProgramNames
{
    static const std::array<const char *, static_cast<int>(Program::DUMMY)> names;
    const char *operator[](Program p) const
    {
        return names[static_cast<int>(p)];
    }
    const char *operator[](size_t p) const
    {
        return names[static_cast<int>(p)];
    }
    std::array<const char *, static_cast<int>(Program::DUMMY)>::const_iterator begin() const
    {
        return names.begin();
    }
    std::array<const char *, static_cast<int>(Program::DUMMY)>::const_iterator end() const
    {
        return names.end();
    }

    constexpr size_t size() const
    {
        return names.size();
    }
};
constexpr ProgramNames programNames;

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

template <typename Stream>
Stream &operator<<(Stream &s, Program t)
{
    s << static_cast<int>(t);
    return s;
}

template <typename Stream>
Stream &operator>>(Stream &s, Program &t)
{
    int tt;
    s >> tt;
    t = static_cast<Program>(tt);
    return s;
}

} // namespace launcher
} // namespace vrb

Q_DECLARE_METATYPE(vrb::launcher::Program);
Q_DECLARE_METATYPE(vrb::launcher::LaunchType);

#endif // !VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H