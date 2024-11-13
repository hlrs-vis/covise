
fn AlignPivotTo Obj Trgt =
(
-- Get matrix from object
if classOf Trgt != matrix3 then Trgt = Trgt.transform

-- Store child transforms
local ChldTms = in coordSys Trgt ( for Chld in Obj.children collect Chld.transform )

-- Current offset transform matrix
local TmScale = scaleMatrix Obj.objectOffsetScale
local TmRot = Obj.objectOffsetRot as matrix3
local TmPos = transMatrix Obj.objectOffsetPos
local TmOffset = TmScale * TmRot * TmPos

-- New offset transform matrix
TmOffset *= obj.transform * inverse Trgt

-- Apply matrix
Obj.transform = Trgt

-- Restore offsets
Obj.objectOffsetPos = TmOffset.translation
Obj.objectOffsetRot = TmOffset.rotation
Obj.objectOffsetScale = TmOffset.scale

-- Restore child transforms
for i = 1 to Obj.children.count do Obj.children[i].transform = ChldTms[i] * inverse Trgt * Obj.transform
)

fn doGroup =
(
	
    sel = selection as array
	theName = "BranchGroup"
	print theName
	groupName = uniquename thename
	print groupName
    group $selection name:groupName
    groupObj = execute ("$'"+groupName + "'")
	print groupObj.name
	setGroupOpen groupObj true
	print groupObj.pivot
	print sel[1].name
	print sel[1].pivot
	AlignPivotTo groupObj sel[1].transform
	tMat = groupObj.transform
	identityMat = matrix3 1
	groupObj.transform = identityMat
	WorldAlignObject groupObj
	groupObj.transform = tMat
	setGroupOpen groupObj false
)

fn doResetTransform =
(
	allObjects = $*
for OBJ in allObjects do(
	 if OBJ.category == #Editable_Objects do 
	(
print OBJ as string
	tMat = OBJ.transform
	identityMat = matrix3 1
	OBJ.transform = identityMat
		ResetXForm OBJ
		
	WorldAlignObject OBJ
		try(convertToMesh OBJ) catch()
	OBJ.transform = tMat
	)
	if OBJ.category == #Group do 
	(
print OBJ as string
	tMat = OBJ.transform
	identityMat = matrix3 1
	OBJ.transform = identityMat
	WorldAlignObject OBJ
	OBJ.transform = tMat
	)
)
    
)

fn doResetTransformSelected =
(
	allObjects = $selection
for OBJ in allObjects do(
	 if OBJ.category == #Editable_Objects do 
	(
print OBJ as string
	tMat = OBJ.transform
	identityMat = matrix3 1
	OBJ.transform = identityMat
		ResetXForm OBJ
		
	WorldAlignObject OBJ
		try(convertToMesh OBJ) catch()
	OBJ.transform = tMat
	)
	if OBJ.category == #Group do 
	(
print OBJ as string
	tMat = OBJ.transform
	identityMat = matrix3 1
	OBJ.transform = identityMat
	WorldAlignObject OBJ
	OBJ.transform = tMat
	)
)
    
)

macroScript groupPivot category:"Optimize" 
(
	
doGroup()
)


macroScript mergeSameMaterial category:"Optimize" 
(
sel = selection as array
	count = 1

	if sel.count > 0 then
	(
		undo on
		(
			while count <= sel.count do
			(
				select sel[count]
				--print $
				--Find if any instance is there for this object
				InstanceMgr.GetInstances $ &instances
				
				--For each instance, make it unique
				if instances.count > 0 do
					InstanceMgr.MakeObjectsUnique &instances #individual
				
				deselect $
				count += 1
			)
		)
		select sel
		
		mergeddObjects = true
		while mergeddObjects == true do
		(
			mergeddObjects=false
			for x in $ do
			(
				if IsValidNode x  do
				(
					if x.category == #Editable_Objects do
					(
						
						print x.name
						merged = false
						for y in $ do
						(
							if x != y and x.material == y.material do
						   (
							   
								merged = true
								s = "merge "+y.name+" to "+x.name
								print s
								meshop.attach x y attachMat:#IDToMat condenseMat:true
							   
								mergeddObjects=true
						   )
						)
						if merged == true do
						(
							print x.name
						)
					)
				)
			)
	    )
		print "removing empty dummys"
		deletedObjects = true
		while deletedObjects == true do
		(
			deletedObjects=false
			for x in $ do
			(
				if IsValidNode x  do
				(
					if classof x == Dummy do
					(
						if x.children.count == 0 do
						(
							deletedObjects=true
							print x.name
							delete x
						)
					)
				)
			)
		)
		
		print "Done"
		
	)
	else
	(
		messageBox "You must select at least one or more objects" title:"MatID Random Material Assigner"
	)
)