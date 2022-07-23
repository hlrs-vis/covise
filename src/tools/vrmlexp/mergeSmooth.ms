macroScript weldSmooth category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					if x.numverts > 0 do
					(
						   case (ClassOf obj) of
				(
					Editable_Poly:	(	polyOp.weldVertsByThreshold x #{1..x.numverts} 0.0001	)
					
					Editable_mesh:   (	meshOp.weldVertsByThreshold x #{1..x.numverts} 0.0001   )
				)
					   if x.numfaces > 0 do
					   (
						   case (ClassOf obj) of
				(
					Editable_Poly:	(	polyOp.autosmooth x #{1..x.numfaces} 25.0	)
					
					Editable_mesh:   (	meshOp.autosmooth x #{1..x.numfaces} 25.0   )
				)
					        meshop
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