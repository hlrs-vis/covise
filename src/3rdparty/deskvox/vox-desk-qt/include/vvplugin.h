// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#ifndef VV_PLUGIN_H
#define VV_PLUGIN_H

#include <QtPlugin>
#include <QString>

class vvVolDesc;
class QDialog;
class QWidget;

/*! plugin interface for the DeskVOX application
 */
class vvPlugin
{
public:
  vvPlugin(const QString& name)
    : _name(name)
    , _active(true)
    , _vd(NULL) {}
  virtual ~vvPlugin() {}

  /*! \brief  override this method to perform custom logic when the volume is loaded
   */
  virtual void setVolDesc(vvVolDesc* vd) { _vd = vd; }
  /*! \brief  override this method if the plugin renders before volume rendering
   */
  virtual void prerender() {}
  /*! \brief  override this method if the plugin renders after volume rendering
   */
  virtual void postrender() {}
  /*! \brief  override this method to adjust state before a new time step is processed
   */
  virtual void timestep() {}
  /*! \brief  override this method if the plugin provides a dialog for the plugins menu
   */
  virtual QDialog* dialog(QWidget* parent = 0)
  {
    (void)parent;
    return NULL;
  }
  QString name() { return _name; }
  bool isActive() { return _active; }
protected:
  QString _name;
  bool _active;
  vvVolDesc* _vd;
};
Q_DECLARE_INTERFACE(vvPlugin, "DeskVOX.Plugin")

#endif

