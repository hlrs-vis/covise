macroScript flipNormals category:"Optimize" 
(
        for x in $ do
		(
			    if x.category == #Editable_Objects do
				(
					
					   if x.numfaces > 0 do
					   (
					        meshop.flipNormals x #{1..x.numfaces}
					        print x.name
					   )
				)
		)
)