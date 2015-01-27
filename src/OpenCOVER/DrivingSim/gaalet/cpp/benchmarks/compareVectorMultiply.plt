set title "Multivector product benchmark, Gaalet cpp version, compiler flag '-O3'"
set auto x
set auto y
set ylabel 'Computation time [s]'
set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9
#set xtic rotate by -45
#set bmargin 10
#set terminal png transparent font "arial" 8
set terminal png transparent
set output "compare_product.png"
plot 'compareVectorMultiply.times' using 2:xtic(1) ti col, '' u 3 ti col

