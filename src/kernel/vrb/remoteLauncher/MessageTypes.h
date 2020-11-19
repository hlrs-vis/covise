#ifndef VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H
#define VRB_REMOTE_LAUNCHER_MESSAGE_TYPES_H

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
    LAST_DUMMY
};
namespace detail{
    typedef std::array<const char *, static_cast<int>(Program::LAST_DUMMY)> ProgramContainer;
    constexpr ProgramContainer programNames{
        "covise",
        "opencover"};
}
struct ProgramNames
{

    const char *operator[](Program p) const
    {
        return detail::programNames[static_cast<int>(p)];
    }
    const char *operator[](size_t p) const
    {
        return detail::programNames[static_cast<int>(p)];
    }
    detail::ProgramContainer::const_iterator begin() const
    {
        return detail::programNames.begin();
    }
    detail::ProgramContainer::const_iterator end() const
    {
        return detail::programNames.end();
    }

    constexpr size_t size() const
    {
        return detail::programNames.size();
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