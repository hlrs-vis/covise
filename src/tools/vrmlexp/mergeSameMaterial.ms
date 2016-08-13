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
        for x in $ do
		(
			if IsValidNode x  do
			(
				print x.name
			    if x.category == #Editable_Objects do
				(
					
                    for y in $ do
		            (
						if x != y and x.material == y.material do
					   (
						    s = "merge "+y.name+" to "+x.name
					        print s
					        meshop.attach x y attachMat:#IDToMat condenseMat:true
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