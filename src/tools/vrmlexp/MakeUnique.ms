
-------------------------------------------------------------------------------
-- MakeUnique.ms
-- By Neil Blevins (soulburn@blur.com)
-- v 1.03
-- Created On: 10/04/00
-- Modified On: 08/22/01
-- tested using Max 4.2
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- Description:
-- Takes a selected object and makes it unique (ie, if it's an instance of 
-- another object, it will no longer be an instance). Originally found as part
-- of Blur's Object utilities, now here it is in script form.
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
-- Revision History:
--
-- v 1.00 Takes a selected object and makes it unique (ie, if it's an instance of 
-- another object, it will no longer be an instance). Originally found as part
-- of Blur's Object utilities, now here it is in script form.
--
-- v 1.01 The script now stops the screen from redrawing between each object, 
-- which speeds it up when you have a lot of objects. Objects also retain the
-- same name as they originally had. Added a progress dialogue.
--
-- v 1.02 Cleaned the code a lot, and removed all code pertaining to selection
-- sets, it uses arrays of scene objects now instead.
--
-- v 1.03 Added a try catch so if the script crashes it still turns screen 
-- redraw back on, and added an error message if you try and use the script
-- on a group.
-------------------------------------------------------------------------------

-------------------------------------------------------------------------------
(
-- Globals

global make_unique

-- Functions

fn make_unique = 
	(
	undo "Make Unique" on
		(
		objs = #()
		new_objs = #()
		group_error = false
		
		disableSceneRedraw()
		
		try
			(
			for i in selection do 
				(
				if isgroupmember i == false and isgrouphead i == false then
					(
					append objs i
					)
				else group_error = true
				)

			progressStart "Make Unique"
			escapeEnable = false
			itemnum = objs.count
			
			curiter = 0
			for i in objs do
				(
				curiter += 1
				m = ((curiter as float)/(itemnum as float))*100
				
				mo = copy i
				mo.name = i.name
				mo.wirecolor = i.wirecolor
				
				append new_objs mo
				progressUpdate m
				)
			progressEnd()
		
			for i in objs do delete i
			if new_objs.count != 0 then select new_objs
			)
		
		catch
			(
			MessageBox "An error has occured, you may want to undo." title:"MakeUnique"
			)
		
		enableSceneRedraw()
		completeRedraw()
		
		if group_error == true then
			(
			MessageBox "At least one object in your selection is a group, and this script cannot operate on groups. Use Blur's Object Utilities instead (http://www.blur.com/blurbeta/). All other objects should now be unique." title:"MakeUnique"
			)
		)
	)

-- The Script

if selection.count == 0 then (MessageBox "Please select at least one object" title:"MakeUnique")
else
	(
	if (queryBox "Do you wish to make these objects Unique?" title:"MakeUnique") == true then make_unique()
	)
)

