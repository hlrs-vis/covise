-- Merge Scene Materials
-- designed by hab, Aug 2nd, 2000
-- Thanks to Egon Heller for this challenge
-- May 3rd, 2001 update for MAX 4
-- Thanks to Norbert Raetz for his suggestions
--
-- matMerge Description:
-- you select the name of a material in the scene,
-- the script merges all materials with the same properties
-- - ambient color
-- - diffuse color
-- - secular color
-- - shininess
-- - shininess strength
-- - opacity
-- - diffuse map
--
-- How to use:
-- * set the material name in the Material Editor
-- * push the 'Update Materials' button
-- * push the 'Merge to selected' button
-- * or push the 'Merge all' button
-- When you "merge all", the first material in the list is used to merge to
-- Materials already merged, are marked merged, if youwant to start from
-- scratch, youhaveto push the 'Update Material List' once again
--
-- Todo:
-- - hardcoded Standard Material classID in file 
-- - mergeTo material class ID is checked to often
-- - hardcoded Bitmap texture classID in file 
-- - check with MAX 3 (shader etc ...)
--
-- Pitfalls (limitations in MAXScript 2.5):
-- - Material list is not updated properly after merge
-- - there's no UNDU after material change
-- - can't compare 2 materials by value, gc in MAX 2.x
--
-- Viel Spass!
--

-- 8-AUG-2000: need this here, don't ask me why ...
global matSelMatList = #()
global matIsMerged = #()

-- 17-AUG-2000: to select nodes only once ...
global matMergeSelected	= 0

--
-- compareArrays
-- compare the values of 2 arrays- here used for classIDs
-- this is necessary, as MAXScript handles array by reference:(
--
fn compareArrays arr1 arr2 =
(
	if arr1.count != arr2.count then return 0
	for i = 1 to arr1.count do
	(
		if arr1[i] != arr2[i] then return 0
	)
	return 1
)

--
-- compareMaps
-- compare 2 maps, if they areequal and bitmaps, compare the names
--
fn compareMaps map1 map2 =
(
	-- both undefined, is ok
	if map1 == undefined and map2 == undefined then return 1
	-- only one undefined is not ok
	if map1 == undefined or map2 == undefined then return 0
	
--	format "--compMap % %\n" map1 map2
	-- same classid
	if compareArrays map1.classID map2.classID == 0 then return 0
	
	-- for a bitmap, compare the file name
	if compareArrays map1.classID #(240,0) == 0 then
	(
		if map1.filename != map2.filename then return 0
	)
	
	-- success
	return 1
)

--
-- compareMaterials
-- compare to materials if they have equal atrributes
--
fn compareMaterials mat1 mat2 =
(
--	format "--compMat % %\n" mat1.name mat2.name
	-- both materials are std materials
	if compareArrays mat1.classID #(2,0) == 0 then return 0
	if compareArrays mat1.classID mat2.classID == 0 then return 0
	-- MAX 3: same shader
--	if mat1.shadertype != mat2.shadertype then return 0
	-- same ambient color
	if mat1.ambient != mat2.ambient then return 0
	-- same diffuse color
	if mat1.diffuse != mat2.diffuse then return 0
	-- same specular color
	if mat1.specular != mat2.specular then return 0
	-- same shininess
	-- MAX 4: now it's called glossiness
--	if mat1.shininess != mat2.shininess then return 0
	if mat1.glossiness != mat2.glossiness then return 0
	-- same shininess strength
	-- MAX 4: now it's called specular level
--	if mat1.shinestrength != mat2.shinestrength then return 0
	if mat1.specularLevel != mat2.specularLevel then return 0
	-- same opacity
	if mat1.opacity != mat2.opacity then return 0
	
	-- same diffuse map
	return compareMaps mat1.diffuseMap mat2.diffuseMap
	
--	return 1
)

--
-- replaceMaterials
-- replace a material with another material
-- find all nodes which use one of the replace materials 
-- and set the new material
--
fn replaceMaterials replaceList mat =
(
	-- select all nodes in the scene
	-- 17-AUG-2000
	if (matMergeSelected == 0) then
	(
		max select all
		matMergeSelected = 1
	)
	
	for node in selection do
	(
		for iMat = 1 to replaceList.count do
		(
		    -- 3-MAY-2001
			-- no material, skip it
			if node.material != undefined then
			(
				-- 17-AUG-2000
				-- need to use the name here, 'value' changes
				if node.material.name == replaceList[iMat].name then 
				(
					node.material = mat
					break
				)
			)
		)
	)
)

-- mergeToMaterial
-- check all materials in the scene if they are equal to a given
-- material and if so, replace the othr material with this material
fn mergeToMaterial mat0 matNr =
(
	local i
	
--	format "--mergeto: %\n" matSelMatList[matNr].name
    -- we create a list of materials
	replaceMatList = #()
--	format "%\n" matSelMatList
	
	-- check each material in the give range
	-- and build the list of equal materials
	for iMerge = mat0 to matSelMatList.count do
	(
		-- don't merge with the one we have
		if iMerge == matNr then 
			continue
		
		-- we don't merge already merged materials
		if matIsMerged[iMerge] != 0 then 
			continue
			
--		format "--compare: %\n" matSelMatList[iMerge]
		if compareMaterials matSelMatList[matNr] matSelMatList[iMerge] > 0 then
		(
--			format "--found: %\n" matSelMatList[iMerge]
			-- add it to the replace list
			append replaceMatList matSelMatList[iMerge]
			-- mark it as merged
			matIsMerged[iMerge] = 1
		)
	)
	
	-- do the replace of the equal materials
	if replaceMatList.Count > 0 then
	(
		format "--mergeTo %: %\n" matSelMatList[matNr] replaceMatList
		replaceMaterials replaceMatList matSelMatList[matNr] 
	)
)


-- setToWhite
fn setToWhite matNr =
(
	local i
	
	matSelMatList[i].diffuseMap.bitmap != undefined
	matSelMatList[i].diffuse = [1,1,1,1]
)
-- matMerge
-- Utility to merge (standard) material
utility matMerge "Merge Materials"
(
	local i
	
	listbox matList "Scene Materials:" width:140
	button matUpdate "Update Material List" width:140
	button matRename "rename Materials" width:140
	button matShow "show all maps" width:140
    button matSelOne "Merge to selected" width:140
    button matSelAll "Merge all" width:140
    button matWhiteTexture "SetWhiteIfTextured" width:140

	-- initialize the dropdown from scene materials
	on matUpdate pressed do (
		matSelMatList = #()
		matSelMatList = sceneMaterials
		matList.items = for i in 1 to matSelMatList.count
			collect matSelMatList[i].name
		-- initialize the matIsMerged array
		matIsMerged = #()
		for i = 1 to matSelMatList.count do
			append matIsMerged 0
		-- 17-AUG-2000
		matMergeSelected = 0
	)
	on matRename pressed do (
		matSelMatList = #()
		matSelMatList = sceneMaterials
		for i in 1 to matSelMatList.count do
		(
			matSelMatList[i].name = "mat"+i as string+matSelMatList[i].name
		)
	)
	on matShow pressed do (
		matSelMatList = #()
		matSelMatList = sceneMaterials
		for i in 1 to matSelMatList.count do
		(
			matSelMatList[i].showInViewport = true
		)
	)

	on matSelOne pressed do 
	(
		-- merge to selected material
		if matList.selection > 0 then 
		(
			-- 17-AUG-2000
			matMergeSelected = 0
			mergeToMaterial 1 matList.selection
		)
		
	)
	
	on matSelAll pressed do 
	(
		-- 17-AUG-2000
		matMergeSelected = 0
		
		format "\n ***merging materials***\n"
		-- merge all possible materials	
		for i = 1 to matSelMatList.Count do
		(
			-- if already merged, skip it 
			if matIsMerged[i] != 0 then continue
			
			format "Material: %\n" matSelMatList[i]
			
			-- merge all after this one
			j = i + 1
			mergeToMaterial j i
		)
		
	)
	
	on matWhiteTexture pressed do 
	(
		-- 17-AUG-2000
		matMergeSelected = 0
		
		format "\n ***merging materials***\n"
		-- merge all possible materials	
		for i = 1 to matSelMatList.Count do
		(
			setToWhite i
		)
		
	)
)

-- EOF
