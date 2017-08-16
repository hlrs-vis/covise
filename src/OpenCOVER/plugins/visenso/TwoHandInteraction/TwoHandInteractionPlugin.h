/* This file is part of COVISE.

   You can use it under the terms of the GNU Lesser General Public License
   version 2.1 or later, see lgpl-2.1.txt.

 * License: LGPL 2+ */

#ifndef _TWOHANDINTERACTION_PLUGIN_H
#define _TWOHANDINTERACTION_PLUGIN_H

#include <cover/coVRPluginSupport.h>
#include <cover/coVRPlugin.h>
#include <memory>

namespace TwoHandInteraction
{

class InteractionHandler;

class TwoHandInteractionPlugin : public opencover::coVRPlugin, vrui::coMenuListener
{
public:
    TwoHandInteractionPlugin();
    virtual ~TwoHandInteractionPlugin();

    struct InteractionStart
    {
        osg::Matrix ScalingMatrix;
        osg::Matrix RotationMatrix;
        osg::Matrix TranslationMatrix;
    };

    struct InteractionResult
    {
        InteractionResult()
        {
        }
        explicit InteractionResult(const InteractionStart &start)
            : ScalingMatrix(start.ScalingMatrix)
            , RotationMatrix(start.RotationMatrix)
            , TranslationMatrix(start.TranslationMatrix)
        {
        }

        osg::Matrix ScalingMatrix;
        osg::Matrix RotationMatrix;
        osg::Matrix TranslationMatrix;
    };

protected:
    // from coVRPlugin
    //! this function is called when COVER is up and running and the plugin is initialized
    bool init();
    //! this function is called from the main thread before rendering a frame
    void preFrame();

private:
    void applyInteractionResult(const InteractionResult &interactionResult);
    void createIndicators(float indicatorSize);

    osg::ref_ptr<osg::MatrixTransform> m_HandIndicator;
    osg::ref_ptr<osg::MatrixTransform> m_SecondHandIndicator;
    bool m_HasIndicators;

    std::shared_ptr<InteractionHandler> m_InteractionHandler;
};
}
#endif
