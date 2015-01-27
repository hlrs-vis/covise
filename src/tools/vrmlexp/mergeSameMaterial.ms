macroScript mergeSameMaterial category:"Optimize" 
(
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