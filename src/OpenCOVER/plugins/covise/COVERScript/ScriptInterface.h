/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _ScriptInterface_H
#define _ScriptInterface_H
#include <QString>
#include <QObject>
class ScriptPlugin;

class ScriptInterface : public QObject
{
    Q_OBJECT
    // define the enabled property
    Q_PROPERTY(bool enabled WRITE setEnabled READ isEnabled)

public:
    ScriptInterface(ScriptPlugin *p);

public slots: // these functions (slots) will be available in Qt Script
    void loadFile(QString file);

    void snapshotDir(QString dirName); // send a message to the PBufferSnapshot plugin to create a snapshot and store it in the directory specified
    void snapshot(QString fileName); // send a message to the PBufferSnapshot plugin to create a snapshot and store it in filename
    void setVisible(const QString &name, bool on);
    void setVariant(const QString &name, bool on);
    void viewAll(bool resetView = false);
    void setCuCuttingSurface(int number, float x, float y, float z, float h, float p, float r);

    void setEnabled(bool e)
    {
        this->enabled = e;
    }
    bool isEnabled() const
    {
        return this->enabled;
    }

signals: // the signals
    void enabledChanged(bool newState);

private:
    bool enabled;
    ScriptPlugin *plugin;
};
#endif
