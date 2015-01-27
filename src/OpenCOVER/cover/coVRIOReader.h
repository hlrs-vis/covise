/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef COVRIOREADER_H
#define COVRIOREADER_H

#include <osg/Node>

#include <list>
#include <string>

#include <cover/coVRIOBase.h>

namespace opencover
{
class COVEREXPORT coVRIOReader : public virtual coVRIOBase
{
public:
    coVRIOReader();
    virtual ~coVRIOReader();

    enum IOStatus
    {
        Idle = 0x00,
        Loading,
        Finished,
        Failed
    };

    /**
     * This method is called for loading a full file. The loader must load the complete file and
     * should attach it to the group node passed to the loader.
     * @param location The location to load the data from.
     * @param group An optional group node the loaded scene should be attached to.
     * @returns A node that all loaded nodes are attached to or 0 on error.
     */
    virtual osg::Node *load(const std::string &location, osg::Group *group = 0) = 0;

    /**
     * Is used by the file manager to check if a handler can chunk its loading process into
     * parts.
     * Loading chunks instead of the full file enhances the responsiveness of the renderer.
     * For the actual loading, the method loadPart has to be implemented. By the handler
     * @returns true If the plugin supports partial loads.
     */
    virtual bool canLoadParts() const = 0;

    /**
     * Is used by the file manager to check if a handler can unload a file.
     * @returns true If the plugin supports unloading.
     */
    virtual bool canUnload() const = 0;

    /**
     * Is used by the file manager to check if a handler is currently loading a file.
     * @returns true If the handler is loading data.
     */
    virtual bool inLoading() const = 0;

    /**
     * Loads the next (or first) part of the current data set.
     * This method must be implemented by handlers claiming to canLoadParts. It is
     * called by the file manager every frame. The file manager ensures, that new
     * partial load operations are only initiated after all previous operations have been
     * completed. The current progress of the load operation should be reflected in the
     * return value of getProgress.
     * @param location The location to load the data from.
     * @param group An optional group node the loaded scene should be attached to.
     * @returns The current status of file loading.
     */
    virtual IOStatus loadPart(const std::string &location, osg::Group *group = 0);

    /**
     * Gets the loaded subgraph after finishing the partial load.
     * @returns The node loaded or 0 if nothing was loaded.
     */
    virtual osg::Node *getLoaded();

    /**
     * This method is called for unloading data.
     * This method must be implemented by handlers claiming to canUnload.
     * @returns If the scene was successfully unloaded.
     */
    virtual bool unload(osg::Node *node);

    virtual const std::list<std::string> &getSupportedReadMimeTypes() const;
    virtual const std::list<std::string> &getSupportedReadFileExtensions() const;

    virtual bool isReader() const
    {
        return true;
    }

protected:
    std::list<std::string> supportedReadFileTypes;
    std::list<std::string> supportedReadFileExtensions;
};
}
#endif // COVRIOREADER_H
