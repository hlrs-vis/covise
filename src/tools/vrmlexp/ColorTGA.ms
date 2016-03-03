macroScript colorTGA category:"Optimize" 
(
	local defaultMat = undefined
	local foundMat = undefined
	print "matlib"
	for mat in scenematerials do
(
	print mat.name
    if mat.name == "Default" then
(
	defaultMat = mat
	print "found defaultMat"
)    
)
	print "objects"
        for x in $ do
		(
			if IsValidNode x  do
			(
				print x.name
				typ = getUserProp x "System Type"
				if typ != undefined then
				(
					 foundMat = undefined
					for mat in meditMaterials do
(
	
    if mat.name == typ then
(
	foundMat = mat
)    
)
			      if foundMat != undefined then				  
				  (
					x.material  = foundMat
			      )
			    )
			)
		)
		print "Done"
)