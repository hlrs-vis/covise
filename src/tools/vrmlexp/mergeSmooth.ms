macroScript weldSmooth category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					if x.numverts > 0 do
					(
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