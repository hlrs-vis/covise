/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef FILESYSACC_H_
#define FILESYSACC_H_

#include <QString>
#include <QStringList>
#include <util/coExport.h>
namespace opencover
{
class FileSysAccess
{
public:
    /**
       * Returns a list of type QStringList of either files at a given location not
       * traversing subdirs or either a list fo directories of a given location not
       * traversing subdirs
       * @param filter	- filter to be used to apply on list to filter entries
       * @param path		- location in the filesystem where the listing of file-
       *					  system entries shall be performed
       * @param listFiles - set to true, to only get a list of file entries, set to
       *					  false to only get a list of directory entries
       * @return A list of type QStringList containing the desired list of entries
       */
    QStringList getFileSysList(QString filter, QString &path, bool listFiles);

    std::string getFileName(const std::string dir, const char *dirSep);

    std::string getTempDir();

    QStringList getLocalHomeDir();

    QString getLocalHomeDirStr();

    QStringList getLocalHomeFiles(const std::string filter);
};
}
#endif
