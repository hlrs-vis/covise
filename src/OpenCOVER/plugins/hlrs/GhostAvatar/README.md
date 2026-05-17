# The Ghost Avatar Plugin

This plugin is the first prototype of the VR Terroir project (more information [here](https://www.cunicode.com/works/vr-terroir)) that works in the CAVE. This prototype was developed with the artist Bernat Cuní during his one-year residency in the STARTS EC(H)O project. In short, the idea is that the appearance of the VR avatar changes depending on how the user interacts with the virtual environment. 

The plugin loads one of two available ghost avatar models: a planar avatar whose shape was generated algorithmatically and a simpler 3D ghost avatar. The avatar's arm (and head in the case of the planar avatar) are controlled by the VR glasses and controller in the CAVE. The avatar is placed at the same position as the user.

As the user explores the virtual world, screenshots of the environment from the user's point of view are added to the avatar's texture either in the form of splotches (which appear at random positions) or
stripes. 

For the test scene during the GATE festival, mirrors can also be loaded into the scene s.t. the user can inspect the avatar (which represents themselves). **In the future the avatar will be embedded into COVER's collaborative mode.**

## Config file settings
The user can change the following settings in the config file (first option is default):
- `avatarType="planar"` or `"ghost"` - chooses the model of the avatar
- `textureType="splotches"` or `"stripes"` - chooses the shape of the screenshots
                                                 that are added to the avatar's texture
- `distanceThreshold="5"` - the distance the user must travel (in world units) 
                                until the texture is updated
- `useInteractors="false"` - (for debugging) if true, the avatar is visible in the scene
                                  and can be controlled with three pick interactors which mimick
                                  the output from the MoCap system (floor, glasses, 3D controller)
- `mirrorsForScene=0` - Depending on the scene number, mirrors will be placed into the scene at pre-defined positions (this is used for the tech demo during the GATE festival). 0 = no mirrors, 1 = mirrors for "cavescene_lin.wrl", 2 = mirrors for "cavescene_lin_xl_modifiers_sd8.glb" (more scenes can be added in the class's `addMirrorsToScene` method)

## Adding your own avatar

The two available avatars have been created in Blender and exported as FBX files.

### What to keep in mind when creating the rigged model in Blender
To make sure the rig is moved correctly, the following adjustments have to be made in Blender:

- The bones that are to be moved need to have a common parent (which can be empty). So if the avatar has one arm and one head bone the hierarchy should look something like this:
    ```
    Bones
    - Head
    - Arm
    ```
- The scale of all parts of the avatar (including the rig) needs to be `1x1x1`. To ensure this do the following:
    ```
    For each node belonging to the avatar in the hierarchy (especially the rig): Click on the node and check its scale in the "Objects Properties" window below.
    --> If it is not 1x1x1: Press "Ctrl+A" and then click on "Rotation & Scale".
    ```
- When exporting the model to FBX, make sure the following options are set:
    ```
    - "Animation" is OFF

    - In "Transform:
        --> "Apply Unit" is OFF
        --> "Apply Scalings" is "FBX Units Scale"

    - In "Armature":
        --> "Add Leaf Bones" is OFF
    ```

You can then create a derived class of the `controls/GhostAvatarControls` class to load this FBX file into COVER (make sure to adjust the default bone names, if applicable, see, e.g., the constructor of `PlanarAvatarControls`) and extend the `GhostAvatarControlsFactory` class with your new avatar.

### Adjust Matrix

Since the bone's coordinate system in Blender and COVER's coordinate system will most likely be different, you will most likely also have to define an adjust matrix for the bones to move correctly (see e.g. the `PlanarAvatarControls` class on how this can be set in the plugin). For testing, the adjust matrix can also be changed in the `GhostAvatar` tab in the TabletUI.
Here is an example on how you can find out what the adjust matrix should be (note that this has to be done for each bone):

1. Open your original avatar model in Blender and enable "Wireframe" viewport shading mode. Then select your avatar, and in the menu beneath the hierarchy go to "Data. Object Data Properties" (the icon is a stick figure), then on "Viewport Display" and there, enable "Axes". This will show you each bone's frame. Let's say your arm bone looks like this:
    ``` 
    y __ __
          /|
         / |
        x  z
    ```
2. Open COVER. In the `GhostAvatar` tab enable  "Debugging" --> "Show Frames", this shows you the frame COVER uses for the head and arm bones (**Please note that the TabletUI only supports two bones right now (called head and arm) and would have to be extended for more bones**). Let's say the arm bone looks like this:

    ```    
            y
            |
     z __ __|
           /
          / 
         x  
    ```

3. To turn the COVER frame into the Blender frame, flip the y axis:
    ```
    (1  0  0)       (1  0  0)
    (0  1  0) -->   (0 -1  0)
    (0  0  1)       (0  0  1)
    ```

    And then swap the y and z axes:
    ```
    (1  0  0)       (1  0  0)
    (0 -1  0) -->   (0  0  1)
    (0  0  1)       (0 -1  0)
    ```
    That's your adjust matrix (for testing set it in the TabletUI, to make the change permanent set the matrix in the constructor of your `GhostAvatarControls` child class).

4. Finally, make sure the arm base vector is set to the axis the arms extends out of the shoulder in blender (for testing, you can set it in the TabletUI, to keep it permanent set it in the constructor of your `GhostAvatarControls` child class).
