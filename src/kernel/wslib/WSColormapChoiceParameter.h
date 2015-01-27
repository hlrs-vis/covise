/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef WSCOLORMAPCHOICEPARAMETER_H
#define WSCOLORMAPCHOICEPARAMETER_H

#include "WSParameter.h"
#include "WSTools.h"

#include "WSColormap.h"

namespace covise
{

class WSLIBEXPORT WSColormapChoiceParameter : public WSParameter
{

    Q_OBJECT

    Q_PROPERTY(int selected READ getSelected WRITE setSelected)

public:
    WSColormapChoiceParameter(const QString &name, const QString &description,
                              const QList<WSColormap> & = QList<WSColormap>(),
                              int selected = 1);

    virtual ~WSColormapChoiceParameter();

public slots:
    /**
       * Set the values and selected value of the parameter
       * @param value A new list of colormaps
       * @param selected The currently selected colormap
       * @return true if the parameter was changed
       */
    bool setValue(const QList<WSColormap> &value, int selected = 1);

    /**
       * Set the selected value of the parameter
       * @param inValue The new value of value
       * @return true if the parameter was changed
       */
    bool setValue(int selected);

    /**
       * Get the list of colormaps
       * @return value
       */
    QList<WSColormap> getValue() const;

    /**
       * Get the selected colormap
       */
    const WSColormap getSelectedValue();

    /**
       * Set the selected colormap
       * @param index The selected colormap
       * @return true if the parameter was changed
       */
    bool setSelected(int index);

    int getSelected() const;

    virtual int getComponentCount() const;

public:
    virtual QString toString() const;

    virtual WSParameter *clone() const;

    virtual const covise::covise__Parameter *getSerialisable();

protected:
    virtual bool setValueFromSerialisable(const covise::covise__Parameter *serialisable);

    virtual QString toCoviseString() const;

private:
    bool setValue(int selected, bool changed);
    covise::covise__ColormapChoiceParameter parameter;
    WSColormapChoiceParameter();
    static WSColormapChoiceParameter *prototype;
};
}

#endif // WSCHOICEPARAMETER_H
