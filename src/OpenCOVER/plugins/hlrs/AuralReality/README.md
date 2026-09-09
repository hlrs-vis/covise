# AuralReality plugin

This plugin was developed as part of the [AuralReality
project](https://www.hlrs.de/projects/detail/auralreality). It allows editing
of audio components in VR through the related "Tooly McToolface" software (TMT) by
atmoky GmbH for authoring audio experiences such as theme park rides or museum
exhibits.

The plugin also supports the playback of audio content through a digital audio
twin, rendering realistic soundscapes for the virtual environments, and for
preview of the designed audio experiences in TMT.

## Software design

The source of truth for audio object state is always a running TMT instance.
There a project file is loaded and the objects are parsed into memory. Through
a network interface, the plugin can synchronize this state and display the
objects. Editing object positions or properties then synchronizes back to TMT.

TMT supports seeking along timelines of so-called "choreographies" which
contain other items to play back, possibly with conditions and timing offsets
or triggers. The composed state of those playback items generates 3D transforms
based on the seek timing. Those transforms are computed by TMT and emitted
through the network interface for this plugin to consume and display. Editing a
component however will need to project back to the underlying object
definition, not its interpreted state.

The instanciation of an object in a choreography is called an "element" (e. g.
sound element, choreography element). Elements therefore change their
transforms and playback state over time. Since choreographies can be nested,
elements are placed in a tree structure, and each element has a parent. Moving
an element in VR therefore changes its underlying position relative to the
parent's transform, and TMT will be able to reinterpret that to a global
transform based on seek time. This plugin does not implement that
transformation, as it does not need to.

## Communications protocol

The communications protocol is described in `./aural-reality-protocol`, which
is its own git repository hosted [on
GitHub](https://github.com/hlrs-vis/aural-reality-protocol).
