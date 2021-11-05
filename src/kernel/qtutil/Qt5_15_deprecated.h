#ifndef QT_UTIL_QT_5_15_DEPRECATED_H
#define QT_UTIL_QT_5_15_DEPRECATED_H

#include <QtGlobal>

#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
#define QT Qt
#define SplitBehaviorFlags Qt
#else
#define QT
#define SplitBehaviorFlags QString
#endif

#endif //QT_UTIL_QT_5_15_DEPRECATED_H
