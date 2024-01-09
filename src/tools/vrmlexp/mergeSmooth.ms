macroScript weldSmooth category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					if x.numverts > 0 do
					(
					print (ClassOf x)
						   case (ClassOf x) of
				(
					Editable_Poly:	(	x.weldThreshold = 0.0001
						                polyOp.weldVertsByThreshold x #{1..x.numverts}
						            )
					PolyMeshObject:	(	x.weldThreshold = 0.0001
						                polyOp.weldVertsByThreshold x #{1..x.numverts}
						            )
					
					Editable_mesh:   (	meshOp.weldVertsByThreshold x #{1..x.numverts} 0.0001   )
				)
					   if x.numfaces > 0 do
					   (
						   case (ClassOf x) of
				(
					Editable_Poly:	(	x.autoSmoothThreshold = 45
						                polyOp.autosmooth x 	)
					PolyMeshObject:	(	x.autoSmoothThreshold = 45
						                polyOp.autosmooth x	)
					
					Editable_mesh:   (	meshOp.autosmooth x #{1..x.numfaces} 45.0   )
				)
					        
					   )
					print x.name
				   )
				)
		)
)
macroScript weldSmoothUnify category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					if x.numverts > 0 do
					(
					   if x.numfaces > 0 do
					   (
					        meshop.unifyNormals x #{1..x.numfaces}
					   )
					   meshop.weldVertsByThreshold x #{1..x.numverts} 0.0001
					   if x.numfaces > 0 do
					   (
					        meshop.autosmooth x #{1..x.numfaces} 25.0
					   )
					print x.name
				   )
				)
		)
)

macroScript flipBackfacing category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					
                    convertto x editable_mesh
					meshop.explodeAllFaces x 0.0
					if x.numverts > 0 do
					(
					   for i = 1 to meshop.getNumFaces x do
					   (
						   local verts = meshop.getFace x i
						   local vertPos = #()
						   vertPos[1] = meshop.getVert x verts[1]
						   vertPos[2] = meshop.getVert x verts[2]
						   vertPos[3] = meshop.getVert x verts[3]
						   local normals meshop.getFaceRNormals x i
 					       local normal = getNormal x verts[1] -- get normal of vertex 1
						   normal = normalize normal
						   if verts.count > 2 do
						   (
							   local v1 = vertPos[1] - vertPos[2]
							   local v2 = vertPos[1] - vertPos[3]
							   local cr = cross v1 v2
							   cr = normalize cr
							   print cr
							   print normal
							   print normals
							   if dot cr normals[1] < 0  do
							   (
								   print "flip"
								   setFace x i verts[1] verts[3] verts[2]
							   )
						   )
					    )
					    print x.name
				    )
				)
		)
)

macroScript flipNormals category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					if x.numverts > 0 do
					(
					 case (ClassOf x) of
				(
					Editable_Poly:	(	polyOp.flipNormals x #{1..x.numfaces} )
					
					Editable_mesh:   (	meshOp.flipNormals x #{1..x.numfaces} )
				)
					print x.name
				    )
				)
		)
)