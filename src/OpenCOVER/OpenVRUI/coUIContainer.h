/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef CO_UICONTAINER_H
#define CO_UICONTAINER_H

#include <OpenVRUI/coUIElement.h>
#include <list>

/** Basic Container
 * This class provides basic functionality and a
 * common interface to all Container elements.<BR>
 * The functionality implemented in this class represents a container
 * which arranges its children on top of each other.
 */
namespace vrui
{

class OPENVRUIEXPORT coUIContainer : public virtual coUIElement
{
public:
    /// Alignment specification for children.
    enum
    {
        CENTER = 0,
        MIN,
        MAX,
        BOTH
    } alignments;

    coUIContainer();
    virtual ~coUIContainer();

    virtual void setEnabled(bool enabled);
    virtual void setHighlighted(bool highlighted);
    virtual void resizeToParent(float x, float y, float z, bool shrink = true);
    virtual void shrinkToMin();
    virtual void addElement(coUIElement *element); ///< Appends a child to this container.
    virtual void removeElement(coUIElement *element); ///< Removes a child from this container.
    virtual void removeLastElement(); ///< Removes a child from this container.
    virtual void showElement(coUIElement *element); ///< Adds the specified element to the scenegraph
    void setXAlignment(int a); ///< set the alignment in X direction of the children
    void setYAlignment(int a); ///< set the alignment in Y direction of the children
    void setZAlignment(int a); ///< set the alignment in Z direction of the children

    /// get the Element's classname
    virtual const char *getClassName() const;
    /// check if the Element or any ancestor is this classname
    virtual bool isOfClassName(const char *) const;

    float getMaxH() const; ///<maximum size in H direction
    float getMaxW() const; ///<maximum size in W direction
    float getMaxD() const; ///<maximum size in D direction

    float getSumH() const; ///<sum of sizes in H direction
    float getSumW() const; ///<sum of sizes in W direction
    float getSumD() const; ///<sum of sizes in D direction

    int getSize()
    {
        return (int)elements.size();
    }

protected:
    virtual void resizeGeometry();

    /// alignment on children in X direction
    int xAlignment;
    /// alignment on children in Y direction
    int yAlignment;
    /// alignment on children in Z direction
    int zAlignment;
    /// List of children elements
    std::list<coUIElement *> elements;

    /// try to get That high/wide if possible
    float prefWidth, prefHeight;
};
}
#endif
