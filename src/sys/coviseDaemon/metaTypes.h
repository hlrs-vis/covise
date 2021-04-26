#ifndef VRB_REMOTE_LAUNCHER_META_TYPES_H
#define VRB_REMOTE_LAUNCHER_META_TYPES_H
#include <vrb/ProgramType.h>
#include <QObject>

Q_DECLARE_METATYPE(std::vector<std::string>);
Q_DECLARE_METATYPE(vrb::Program);

#endif // !VRB_REMOTE_LAUNCHER_META_TYPES_H