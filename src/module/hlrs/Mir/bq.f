!=====================================================================
!
!	This module calculates the matrix products
!	and matrix vector products necessary for
!	the calculations performed
!
!	Original Author: Michael M. Resch (MR)
!	First released: 1. November 1994
!
!	First revision:
!			date:
!			author: (MR)
!			reason: when distributing results onto
!				cells calculate minima and maxima
!				of values.
!=====================================================================
	module visco_bq
	contains

!=====================================================================
	subroutine visco_calc_Bq()
!=====================================================================
!	This subroutine calculates B(q)*q.
!	Calculation is done using a splitting algorithm.
!	For further description see the manual.
!=====================================================================

	use visco_declaration
	use auxiliary
        use visco_kernel
        

	implicit none

!=====================================================================
!	matrices for the splitting:
!	B_i	the whole system matrix
!
!	B_i_p	the matrix of positive eigenvalues
!	B_i_m	the matrix of negative eigenvalues
!
!	the last two matrices are made up of a set of matrices
!	as described below.
!
!=====================================================================
	real(double), dimension(5,5)	:: B_i,B_i_p,B_i_m
!=====================================================================
!	matrices of eigenvector products
!	B_1/2/3/4/5 is matrix of r_i*l_i^T*lambda_i
!	
!	the sum of these matrices should give B_i
!=====================================================================
	real(double), dimension(5,5)	:: B_1,B_2,B_3,B_4,B_5
	real(double), dimension(5)	:: loc_rhs
	real(double), dimension(5)	:: eigen_value
	real(double)			:: alpha,beta,velocity
	integer				:: i,k,m,n

	Bq	= 0.0
	alpha 	= 0.0
	beta 	= 0.0
!=====================================================================
!	according to sign of eigenvalues different matrices
!	and types of discretisation are used.
!=====================================================================

!=====================================================================
!	inflow lower corner
!=====================================================================
	i = 1
!=====================================================================
!	x-direction
!=====================================================================
	call visco_calc_evals(i, 1,alpha,beta,velocity,			&
				    eigen_value)
	call visco_calc_B_matrices(i,1,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my inflow bc for u-velocity
!	u is calculated from a parabel
!=====================================================================
	loc_rhs(1) = (cell_data(i)%unknowns%velocities(1) - 		&
		      parabel(cell_data(i)%centre(2))) /		&
                     (cell_data(i)%centre(1) - 				&
		      cell_data(i)%coordinates(1,3))
!=====================================================================
!	this is my inflow bc for v-velocity
!	v is said to be 0 in the entry region
!	additionally derivatives of v are said
!	to be zero in the entry region
!=====================================================================
	loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/		&
	                     (cell_data(i)%centre(1) -   		&
			      cell_data(i)%coordinates(1,3))
!=====================================================================
!	from boundary conditions for u and v follows that
!	sigma = 2*u_y**2*viscosity*lambda 
!       tau   = u_y*viscocity
!	gamma = 0.0
!=====================================================================
	loc_rhs(3) = (cell_data(i)%unknowns%stresses(1) -		&
		      2*parabel_y(cell_data(i)%centre(2))**2 *		&
		      viscosity*lambda)/				&
                     (cell_data(i)%centre(1) - 				&
                      cell_data(i)%coordinates(1,3))
	loc_rhs(4) = (cell_data(i)%unknowns%stresses(2) -		&
		      parabel_y(cell_data(i)%centre(2))*viscosity)/	&
                    (cell_data(i)%centre(1) - 				&
                      cell_data(i)%coordinates(1,3))
	loc_rhs(5) = cell_data(i)%unknowns%stresses(3) /		&
                    (cell_data(i)%centre(1) - 				&
                      cell_data(i)%coordinates(1,3))
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
        k = cell_data(i)%neighbours(1)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                              (cell_data(k)%unknowns%velocities(n) -  &
                               cell_data(i)%unknowns%velocities(n)) / &
                    (cell_data(k)%centre(1) - cell_data(i)%centre(1))
          end do
        end do
!=====================================================================
!	in y-direction
!=====================================================================
	call visco_calc_evals(i,2,alpha,beta,velocity,			&
				    eigen_value)
	call visco_calc_B_matrices(i,2,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my inflow bc for fixed wall u=0
!=====================================================================
	loc_rhs(1) = cell_data(i)%unknowns%velocities(1) /	&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,1))
!=====================================================================
!	this is my inflow bc for fixed wall v=0
!=====================================================================
	loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/ 		&
		     (cell_data(i)%centre(2) - 				&
		      cell_data(i)%coordinates(2,1))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_4(i,loc_rhs)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	k = cell_data(i)%neighbours(2)
	do m = 1,5
	  do n = 1,5
	    Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*		&
			(cell_data(k)%unknowns%velocities(n) -  	&
			 cell_data(i)%unknowns%velocities(n))/		&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	  end do
	end do
!=====================================================================
!	Calulation for inflow boundary
!	u	is set to a parabel
!	v 	is set to be zero
!	sigma   2*u_y**2*viscosity*lambda 
!	tau     u_y*viscosity
!       gamma   0.0
!=====================================================================
	B_INFLOW: do i = 2,inflow
!=====================================================================
!	x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)

	  call visco_calc_B_matrices(i,1,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
	  loc_rhs    = 0.0
!=====================================================================
!	this is my inflow bc for u-velocity
!=====================================================================
	  loc_rhs(1) = (cell_data(i)%unknowns%velocities(1) - 		&
		        parabel(cell_data(i)%centre(2))) /		&
                       (cell_data(i)%centre(1) -			&
		        cell_data(i)%coordinates(1,3))
!=====================================================================
!	this is my inflow bc for v-velocity
!       v is said to be 0 in the entry region
!       additionally derivatives of v are said
!       to be zero in the entry region
!=====================================================================
	loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/		&
	                     (cell_data(i)%centre(1) -   		&
			      cell_data(i)%coordinates(1,3))
!=====================================================================
!	from boundary conditions for u and v follows that
!	sigma = 2*u_y**2*viscosity*lambda 
!	tau   = u_y*viscocity
!       gamma = 0.0
!=====================================================================
	loc_rhs(3) = (cell_data(i)%unknowns%stresses(1) -		&
		      2*parabel_y(cell_data(i)%centre(2))**2 *		&
		      viscosity*lambda)/				&
                       (cell_data(i)%centre(1) -			&
		        cell_data(i)%coordinates(1,3))
	loc_rhs(4) = (cell_data(i)%unknowns%stresses(2) -		&
		      parabel_y(cell_data(i)%centre(2))*viscosity)/	&
                       (cell_data(i)%centre(1) -			&
		        cell_data(i)%coordinates(1,3))
	loc_rhs(5) = cell_data(i)%unknowns%stresses(3) /		&
                       (cell_data(i)%centre(1) -			&
		        cell_data(i)%coordinates(1,3))
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
        k = cell_data(i)%neighbours(1)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   	&
                              (cell_data(k)%unknowns%velocities(n) -  	&
                               cell_data(i)%unknowns%velocities(n)) / 	&
                    (cell_data(k)%centre(1) - cell_data(i)%centre(1))
          end do
        end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
	  call visco_calc_B_matrices(i,2,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
				    
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*		&
			(cell_data(i)%unknowns%velocities(n) -		&
			 cell_data(k)%unknowns%velocities(n)) /  	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*		&
			(cell_data(k)%unknowns%velocities(n) -  	&
			 cell_data(i)%unknowns%velocities(n))/		&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do
	end do B_INFLOW

!=====================================================================
!	inflow upper corner
!=====================================================================
	i = c_inflow_2
!=====================================================================
!	x-direction
!=====================================================================
	call visco_calc_evals(i,1,alpha,beta,velocity,			&
				    eigen_value)
	call visco_calc_B_matrices(i,1,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my inflow bc for u-velocity
!=====================================================================
	loc_rhs(1) = (cell_data(i)%unknowns%velocities(1) - 		&
		      parabel(cell_data(i)%centre(2))) /		&
                     (cell_data(i)%centre(1) - 				&
		      cell_data(i)%coordinates(1,3))
!=====================================================================
!	this is my inflow bc for v-velocity
!       v is said to be 0 in the entry region
!       additionally derivatives of v are said
!       to be zero in the entry region
!=====================================================================
	loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/		&
	                     (cell_data(i)%centre(1) -   		&
			      cell_data(i)%coordinates(1,3))
!=====================================================================
!	from boundary conditions for u and v follows that
!	sigma = 2*u_y**2*viscosity*lambda 
!       tau   = u_y*viscocity
!	gamma = 0.0
!=====================================================================
	loc_rhs(3) = (cell_data(i)%unknowns%stresses(1) -		&
		      2*parabel_y(cell_data(i)%centre(2))**2 *		&
		      viscosity*lambda)/				&
                     (cell_data(i)%centre(1) - 				&
		      cell_data(i)%coordinates(1,3))
	loc_rhs(4) = (cell_data(i)%unknowns%stresses(2) -		&
		      parabel_y(cell_data(i)%centre(2))*viscosity)/	&
                     (cell_data(i)%centre(1) - 				&
		      cell_data(i)%coordinates(1,3))
	loc_rhs(5) = cell_data(i)%unknowns%stresses(3) /		&
                     (cell_data(i)%centre(1) - 				&
		      cell_data(i)%coordinates(1,3))
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
        k = cell_data(i)%neighbours(1)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                              (cell_data(k)%unknowns%velocities(n) -  &
                               cell_data(i)%unknowns%velocities(n)) / &
                    (cell_data(k)%centre(1) - cell_data(i)%centre(1))
          end do
        end do
!=====================================================================
!	in y-direction
!=====================================================================
	call visco_calc_evals(i,2,alpha,beta,velocity,			&
				    eigen_value)
	call visco_calc_B_matrices(i,2,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	k = cell_data(i)%neighbours(4)
	do m = 1,5
	  do n = 1,5
	    Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*		&
			(cell_data(i)%unknowns%velocities(n) -		&
			 cell_data(k)%unknowns%velocities(n)) /  	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	  end do
	end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my inflow bc for u-velocity
!=====================================================================
	loc_rhs(1) = -cell_data(i)%unknowns%velocities(1) /	&
		    (cell_data(i)%coordinates(2,2) - 		&
                     cell_data(i)%centre(2))
!=====================================================================
!	this is my inflow bc for v-velocity
!=====================================================================
	loc_rhs(2) = -cell_data(i)%unknowns%velocities(2)/ 		&
		    (cell_data(i)%coordinates(2,2) - 		&
                     cell_data(i)%centre(2))
		     
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_2(i,loc_rhs)
	do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*loc_rhs(n)
          end do
        end do

!=====================================================================
!	outflow upper corner
!=====================================================================
	i = c_outflow_2
!=====================================================================
!	in x-direction
!=====================================================================
	call visco_calc_evals(i,1,alpha,beta,velocity,			&
				    eigen_value)
        call visco_calc_B_matrices(i,1,velocity,			&
                                   eigen_value,alpha,beta,		&
                                   B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(1,alpha,beta,				&
			    eigen_value,velocity,			&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
        k = cell_data(i)%neighbours(3)
        do m = 1,5
          do n = 3,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
          end do
        end do
!=====================================================================
!	pressure free outflow is assumed
!	thus no further calculation has to be done
!	take care of stress bcs!!!
!
!	I'm sure there is something missing here
!
!=====================================================================

!=====================================================================
!	in y-direction
!=====================================================================
	call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
        call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	k = cell_data(i)%neighbours(4)
	do m = 1,5
	  do n = 1,5
	    Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	  end do
	end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my bc for u-velocity
!=====================================================================
	loc_rhs(1) = -cell_data(i)%unknowns%velocities(1) /	&
		     (cell_data(i)%coordinates(2,2) -		&
                      cell_data(i)%centre(2))
!=====================================================================
!	this is my bc for v-velocity
!=====================================================================
	loc_rhs(2) = -cell_data(i)%unknowns%velocities(2)/ 		&
		     (cell_data(i)%coordinates(2,2) -		&
                      cell_data(i)%centre(2))

!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_2(i,loc_rhs)
	do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*loc_rhs(n)
          end do
        end do

	B_OUTFLOW: do i = c_outflow_2+1,outflow
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 3,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
          end do
!=====================================================================
!	pressure free outflow is assumed
!	thus no further calculation has to be done
!
!	i'm pretty sure there is something missing 
!	here still
!
!=====================================================================

!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do
	end do B_OUTFLOW

!=====================================================================
!	lower outflow corner
!=====================================================================
	i = c_outflow_4
!=====================================================================
!	in x-direction
!=====================================================================
	call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
        call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
        k = cell_data(i)%neighbours(3)
        do m = 1,5
          do n = 3,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
          end do
        end do
!=====================================================================
!	pressure free outflow is assumed
!	thus no further calculation has to be done
!
!	i'm pretty sure there is something still
!	missing here
!
!=====================================================================

!=====================================================================
!	in y-direction
!=====================================================================
	call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
        call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my outflow bc for u-velocity
!=====================================================================
	 loc_rhs(1) = cell_data(i)%unknowns%velocities(1) /	&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,1))
!=====================================================================
!	this is my outflow bc for v-velocity
!=====================================================================
	 loc_rhs(2) = (cell_data(i)%unknowns%velocities(2))/ 		&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,1))

!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_4(i,loc_rhs)
         do m = 1,5
           do n = 1,5
             Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
           end do
         end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	k = cell_data(i)%neighbours(2)
	do m = 1,5
	  do n = 1,5
	    Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	  end do
	end do

	B_WALL_2: do i = c_outflow_4+1,wall_2
!=====================================================================
!	x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
	  call visco_calc_B_matrices(i,1,velocity,			&
				   eigen_value,alpha,beta,		&
				   B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
	  end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
          k = cell_data(i)%neighbours(1)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                                (cell_data(k)%unknowns%velocities(n) -  &
                                 cell_data(i)%unknowns%velocities(n)) / &
                      (cell_data(k)%centre(1) - cell_data(i)%centre(1))
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  loc_rhs    = 0.0
!=====================================================================
!	this is my bc for u-velocity
!=====================================================================
	  loc_rhs(1) = -cell_data(i)%unknowns%velocities(1) /	&
		       (cell_data(i)%coordinates(2,2) -		&
                    	cell_data(i)%centre(2))
!=====================================================================
!	this is my bc for v-velocity
!=====================================================================
	  loc_rhs(2) = -cell_data(i)%unknowns%velocities(2)/ 		&
		       (cell_data(i)%coordinates(2,2) -		&
                    	cell_data(i)%centre(2))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_2(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*loc_rhs(n)
            end do
          end do
	end do B_WALL_2

!=====================================================================
!       all the c_wall_12 cells (corner cells)
!=====================================================================

        B_WALL_12: do i = wall_2+1, c_wall_12
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}. but (i+1,j) does not exist!
!	boundary condition is u=v=0.
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = 0.0 - cell_data(i)%unknowns%velocities(1)/	&
	  		(cell_data(i)%coordinates(1,1) -		&
			 cell_data(i)%centre(1))
          loc_rhs(2) = 0.0 - cell_data(i)%unknowns%velocities(2)/       &
	                (cell_data(i)%coordinates(1,1) -     &
			 cell_data(i)%centre(1))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_1(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)* loc_rhs(n)
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}. (i,j+1) does not exist.
!	boundary condition u=v=0
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = 0.0 - cell_data(i)%unknowns%velocities(1)/	&
	  		(cell_data(i)%coordinates(2,2) -		&
			 cell_data(i)%centre(2))
          loc_rhs(2) = 0.0 - cell_data(i)%unknowns%velocities(2)/       &
	                (cell_data(i)%coordinates(2,2) -     &
			 cell_data(i)%centre(2))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_2(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)* loc_rhs(n)
            end do
          end do
	end do B_WALL_12

!=====================================================================
!       this is a loop over all wall_1 volumes
!=====================================================================
        B_WALL_1: do i = c_wall_12+1,wall_1
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}. but (i+1,j) does not exist!
!	boundary condition is u=v=0.
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = 0.0 - cell_data(i)%unknowns%velocities(1)/	&
	  		(cell_data(i)%coordinates(1,1) -		&
			 cell_data(i)%centre(1))
          loc_rhs(2) = 0.0 - cell_data(i)%unknowns%velocities(2)/       &
	                (cell_data(i)%coordinates(1,1) -     &
			 cell_data(i)%centre(1))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_1(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)* loc_rhs(n)
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do	
	end do B_WALL_1

!=====================================================================
!       this is a loop over all c_wall_41 corners
!=====================================================================

        B_WALL_41: do i = wall_1+1,c_wall_41
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}. but (i+1,j) does not exist!
!	boundary condition is u=v=0.
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = 0.0 - cell_data(i)%unknowns%velocities(1)/	&
	  		(cell_data(i)%coordinates(1,1) -		&
			 cell_data(i)%centre(1))
          loc_rhs(2) = 0.0 - cell_data(i)%unknowns%velocities(2)/       &
	                (cell_data(i)%coordinates(1,1) -     &
			 cell_data(i)%centre(1))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_1(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)* loc_rhs(n)
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my bc for u-velocity
!=====================================================================
	loc_rhs(1) = cell_data(i)%unknowns%velocities(1) /	&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	this is my bc for v-velocity
!=====================================================================
	loc_rhs(2) = (cell_data(i)%unknowns%velocities(2))/ 		&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_4(i,loc_rhs)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do
	end do B_WALL_41

!=====================================================================
!       this is a loop over all wall_4 volumes
!=====================================================================
        B_WALL_4: do i = c_wall_41+1,wall_4
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
          k = cell_data(i)%neighbours(1)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                                (cell_data(k)%unknowns%velocities(n) -  &
                                 cell_data(i)%unknowns%velocities(n)) / &
                      (cell_data(k)%centre(1) - cell_data(i)%centre(1))
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my bc for u-velocity
!=====================================================================
	loc_rhs(1) = cell_data(i)%unknowns%velocities(1) /	&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	this is my bc for v-velocity
!=====================================================================
	loc_rhs(2) = (cell_data(i)%unknowns%velocities(2))/ 		&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_4(i,loc_rhs)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do
	end do B_WALL_4

!=====================================================================
!       this is a loop over all c_wall_34 corners
!=====================================================================
        do i = wall_4+1, c_wall_34
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}. (i-1,j) does not exist.
!	boundary condition is u=v=0
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = cell_data(i)%unknowns%velocities(1)/	&
	  	       (cell_data(i)%centre(1) -		&
		        cell_data(i)%coordinates(1,4))
	  loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/	&
	  	       (cell_data(i)%centre(1) -		&
		        cell_data(i)%coordinates(1,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_3(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)* loc_rhs(n)
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
          k = cell_data(i)%neighbours(1)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                                (cell_data(k)%unknowns%velocities(n) -  &
                                 cell_data(i)%unknowns%velocities(n)) / &
                      (cell_data(k)%centre(1) - cell_data(i)%centre(1))
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my bc for u-velocity
!=====================================================================
	loc_rhs(1) = cell_data(i)%unknowns%velocities(1) /	&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	this is my bc for v-velocity
!=====================================================================
	loc_rhs(2) = (cell_data(i)%unknowns%velocities(2))/ 		&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
        call calc_boundary_stresses_4(i,loc_rhs)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do	
	end do
!=====================================================================
!       this is a loop over all wall_3 cells
!=====================================================================
        do i = c_wall_34+1,wall_3
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}. (i-1,j) does not exist.
!	boundary condition is u=v=0
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = cell_data(i)%unknowns%velocities(1)/	&
	  	       (cell_data(i)%centre(1) -		&
		        cell_data(i)%coordinates(1,4))
	  loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/	&
	  	       (cell_data(i)%centre(1) -		&
		        cell_data(i)%coordinates(1,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_3(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)* loc_rhs(n)
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
          k = cell_data(i)%neighbours(1)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                                (cell_data(k)%unknowns%velocities(n) -  &
                                 cell_data(i)%unknowns%velocities(n)) / &
                      (cell_data(k)%centre(1) - cell_data(i)%centre(1))
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do	
	end do
!=====================================================================
!       this is a loop over all c_wall_23 corner cells
!=====================================================================
        do i = wall_3+1,c_wall_23
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}. (i-1,j) does not exist.
!	boundary condition is u=v=0
!=====================================================================
	  loc_rhs = 0.0
	  loc_rhs(1) = cell_data(i)%unknowns%velocities(1)/	&
	  	       (cell_data(i)%centre(1) -		&
		        cell_data(i)%coordinates(1,4))
	  loc_rhs(2) = cell_data(i)%unknowns%velocities(2)/	&
	  	       (cell_data(i)%centre(1) -		&
		        cell_data(i)%coordinates(1,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_3(i,loc_rhs)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)* loc_rhs(n)
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
          k = cell_data(i)%neighbours(1)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                                (cell_data(k)%unknowns%velocities(n) -  &
                                 cell_data(i)%unknowns%velocities(n)) / &
                      (cell_data(k)%centre(1) - cell_data(i)%centre(1))
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	loc_rhs    = 0.0
!=====================================================================
!	this is my bc for u-velocity
!=====================================================================
	loc_rhs(1) = cell_data(i)%unknowns%velocities(1) /	&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	this is my bc for v-velocity
!=====================================================================
	loc_rhs(2) = (cell_data(i)%unknowns%velocities(2))/ 		&
                    (cell_data(i)%centre(2) - 			&
		     cell_data(i)%coordinates(2,4))
!=====================================================================
!	for the stresses no explicit boundary conditions
!	are given. the values of the boundary volume and of
!       its two next inner neighbours are used to
!       calculate a quadratic interpolation of stress
!       values. this quadratic interpolation is used to
!       get a boundary value.
!=====================================================================
          call calc_boundary_stresses_2(i,loc_rhs)
        do m = 1,5
          do n = 1,5
            Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*loc_rhs(n)
          end do
        end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do	
	end do

!=====================================================================
!       this is a loop over all inner cells
!=====================================================================
	B_INNER: do i = c_wall_23+1,dim
!=====================================================================
!	in x-direction
!=====================================================================
	  call visco_calc_evals(i,1,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,1,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(1,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!       in case of positiv velocity a backward discretization
!       is used q_{i,j} - q_{i-1,j}
!=====================================================================
          k = cell_data(i)%neighbours(3)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*   &
                                (cell_data(i)%unknowns%velocities(n) -  &
                                 cell_data(k)%unknowns%velocities(n))/  &
                      (cell_data(i)%centre(1) - cell_data(k)%centre(1))
            end do
          end do
!=====================================================================
!       in case of negativ velocity a forward discretization
!       is used q_{i+1,j} - q_{i,j}
!=====================================================================
          k = cell_data(i)%neighbours(1)
          do m = 1,5
            do n = 1,5
              Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*   &
                                (cell_data(k)%unknowns%velocities(n) -  &
                                 cell_data(i)%unknowns%velocities(n)) / &
                      (cell_data(k)%centre(1) - cell_data(i)%centre(1))
            end do
          end do
!=====================================================================
!	in y-direction
!=====================================================================
	  call visco_calc_evals(i,2,alpha,beta,velocity,		&
				    eigen_value)
          call visco_calc_B_matrices(i,2,velocity,			&
                                     eigen_value,alpha,beta,		&
                                     B_i,B_1,B_2,B_3,B_4,B_5)
	  call visco_calc_B_m_p(2,alpha,beta,				&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	in case of positiv velocity a backward discretization
!	is used q_{i,j} - q_{i,j-1}
!=====================================================================
	  k = cell_data(i)%neighbours(4)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_p(m,n)*	&
			(cell_data(i)%unknowns%velocities(n) -  &
			 cell_data(k)%unknowns%velocities(n))/	&
                      (cell_data(i)%centre(2) - cell_data(k)%centre(2))
	    end do
	  end do
!=====================================================================
!	in case of negativ velocity a forward discretization
!	is used q_{i,j+1} - q_{i,j}
!=====================================================================
	  k = cell_data(i)%neighbours(2)
	  do m = 1,5
	    do n = 1,5
	      Bq((i-1)*5+m) = Bq((i-1)*5+m) + B_i_m(m,n)*	&
			(cell_data(k)%unknowns%velocities(n) -  &
			 cell_data(i)%unknowns%velocities(n))/	&
                      (cell_data(k)%centre(2) - cell_data(i)%centre(2))
	    end do
	  end do
	end do B_INNER

	end subroutine visco_calc_Bq

!=====================================================================
        subroutine calc_boundary_stresses_1(i,loc_rhs)

!=====================================================================
!   This subroutine calculates stress values at a boundary for
!   walls of type 1
!=====================================================================
        use visco_declaration
        use auxiliary

        integer :: i,k,kk
	real(double), dimension(5)	:: loc_rhs
        real(double) :: temp


        k = cell_data(i)%neighbours(3)
        kk = cell_data(k)%neighbours(3)
        temp = extrapolate(cell_data(i)%coordinates(1,1),               &
                      cell_data(i)%centre(1),                           &
                      cell_data(i)%unknowns%stresses(1),                &
                      cell_data(k)%centre(1),                           &
                      cell_data(k)%unknowns%stresses(1),                &
                      cell_data(kk)%centre(1),                          &
                      cell_data(kk)%unknowns%stresses(1))
        loc_rhs(3) = (temp - cell_data(i)%unknowns%stresses(1))/        &
                      (cell_data(i)%coordinates(1,1)- 			&
		      cell_data(i)%centre(1))
                      
        temp = extrapolate(cell_data(i)%coordinates(1,1),        &
                      cell_data(i)%centre(1),                           &
                      cell_data(i)%unknowns%stresses(2),                &
                      cell_data(k)%centre(1),                           &
                      cell_data(k)%unknowns%stresses(2),                &
                      cell_data(kk)%centre(1),                          &
                      cell_data(kk)%unknowns%stresses(2))
        loc_rhs(4) = (temp - cell_data(i)%unknowns%stresses(2))/        &
                      (cell_data(i)%coordinates(1,1)- 			&
		      cell_data(i)%centre(1))
                      
        temp = extrapolate(cell_data(i)%coordinates(1,1),        &
                      cell_data(i)%centre(1),                           &
                      cell_data(i)%unknowns%stresses(3),                &
                      cell_data(k)%centre(1),                           &
                      cell_data(k)%unknowns%stresses(3),                &
                      cell_data(kk)%centre(1),                          &
                      cell_data(kk)%unknowns%stresses(3))
        loc_rhs(5) = (temp - cell_data(i)%unknowns%stresses(3))/        &
                      (cell_data(i)%coordinates(1,1)- 			&
		      cell_data(i)%centre(1))

        end subroutine calc_boundary_stresses_1

!=====================================================================
        subroutine calc_boundary_stresses_2(i,loc_rhs)

!=====================================================================
!   This subroutine calculates stress values at a boundary for
!   walls of type 2
!=====================================================================
        use visco_declaration
        use auxiliary

        integer :: i,k,kk
	real(double), dimension(5)	:: loc_rhs
        real(double) :: temp


        k = cell_data(i)%neighbours(4)
        kk = cell_data(k)%neighbours(4)
        temp = extrapolate(cell_data(i)%coordinates(2,2),        &
                      cell_data(i)%centre(2),                           &
                      cell_data(i)%unknowns%stresses(1),                &
                      cell_data(k)%centre(2),                           &
                      cell_data(k)%unknowns%stresses(1),                &
                      cell_data(kk)%centre(2),                          &
                      cell_data(kk)%unknowns%stresses(1))
        loc_rhs(3) = (temp - cell_data(i)%unknowns%stresses(1))/        &
		     (cell_data(i)%coordinates(2,2) -                   &
                      cell_data(i)%centre(2))
                      
        temp = extrapolate(cell_data(i)%coordinates(2,2),        &
                      cell_data(i)%centre(2),                           &
                      cell_data(i)%unknowns%stresses(2),                &
                      cell_data(k)%centre(2),                           &
                      cell_data(k)%unknowns%stresses(2),                &
                      cell_data(kk)%centre(2),                          &
                      cell_data(kk)%unknowns%stresses(2))
        loc_rhs(4) = (temp - cell_data(i)%unknowns%stresses(2))/        &
		     (cell_data(i)%coordinates(2,2) -                   &
                      cell_data(i)%centre(2))
                      
        temp = extrapolate(cell_data(i)%coordinates(2,2),        &
                      cell_data(i)%centre(2),                           &
                      cell_data(i)%unknowns%stresses(3),                &
                      cell_data(k)%centre(2),                           &
                      cell_data(k)%unknowns%stresses(3),                &
                      cell_data(kk)%centre(2),                          &
                      cell_data(kk)%unknowns%stresses(3))
        loc_rhs(5) = (temp - cell_data(i)%unknowns%stresses(3))/        &
		     (cell_data(i)%coordinates(2,2) -                   &
                      cell_data(i)%centre(2))

        end subroutine calc_boundary_stresses_2

!=====================================================================
        subroutine calc_boundary_stresses_3(i,loc_rhs)

!=====================================================================
!   This subroutine calculates stress values at a boundary for
!   walls of type 3
!=====================================================================
        use visco_declaration
        use auxiliary

        integer :: i,k,kk
	real(double), dimension(5)	:: loc_rhs
        real(double) :: temp


        k = cell_data(i)%neighbours(1)
        kk = cell_data(k)%neighbours(1)
        temp = extrapolate(cell_data(i)%coordinates(1,3),               &
                      cell_data(i)%centre(1),                           &
                      cell_data(i)%unknowns%stresses(1),                &
                      cell_data(k)%centre(1),                           &
                      cell_data(k)%unknowns%stresses(1),                &
                      cell_data(kk)%centre(1),                          &
                      cell_data(kk)%unknowns%stresses(1))
        loc_rhs(3) = (temp - cell_data(i)%unknowns%stresses(1))/        &
		     (cell_data(i)%coordinates(1,3) -                   &
                      cell_data(i)%centre(1))
                      
        temp = extrapolate(cell_data(i)%coordinates(1,3),        &
                      cell_data(i)%centre(1),                           &
                      cell_data(i)%unknowns%stresses(2),                &
                      cell_data(k)%centre(1),                           &
                      cell_data(k)%unknowns%stresses(2),                &
                      cell_data(kk)%centre(1),                          &
                      cell_data(kk)%unknowns%stresses(2))
        loc_rhs(4) = (temp - cell_data(i)%unknowns%stresses(2))/        &
		     (cell_data(i)%coordinates(1,3) -                   &
                      cell_data(i)%centre(1))
                      
        temp = extrapolate(cell_data(i)%coordinates(1,3),        &
                      cell_data(i)%centre(1),                           &
                      cell_data(i)%unknowns%stresses(3),                &
                      cell_data(k)%centre(1),                           &
                      cell_data(k)%unknowns%stresses(3),                &
                      cell_data(kk)%centre(1),                          &
                      cell_data(kk)%unknowns%stresses(3))
        loc_rhs(5) = (temp - cell_data(i)%unknowns%stresses(3))/        &
		     (cell_data(i)%coordinates(1,3) -                   &
                      cell_data(i)%centre(1))
                      

        end subroutine calc_boundary_stresses_3

!=====================================================================
        subroutine calc_boundary_stresses_4(i,loc_rhs)

!=====================================================================
!   This subroutine calculates stress values at a boundary for
!   walls of type 4
!=====================================================================
        use visco_declaration
        use auxiliary

        integer :: i,k,kk
	real(double), dimension(5)	:: loc_rhs
        real(double) :: temp


        k = cell_data(i)%neighbours(2)
        kk = cell_data(k)%neighbours(2)
        temp = extrapolate(cell_data(i)%coordinates(2,1),        &
                      cell_data(i)%centre(2),                           &
                      cell_data(i)%unknowns%stresses(1),                &
                      cell_data(k)%centre(2),                           &
                      cell_data(k)%unknowns%stresses(1),                &
                      cell_data(kk)%centre(2),                          &
                      cell_data(kk)%unknowns%stresses(1))
        loc_rhs(3) = (temp - cell_data(i)%unknowns%stresses(1))/        &
		     (cell_data(i)%coordinates(2,1) -                   &
                      cell_data(i)%centre(2))
                      
        temp = extrapolate(cell_data(i)%coordinates(2,1),        &
                      cell_data(i)%centre(2),                           &
                      cell_data(i)%unknowns%stresses(2),                &
                      cell_data(k)%centre(2),                           &
                      cell_data(k)%unknowns%stresses(2),                &
                      cell_data(kk)%centre(2),                          &
                      cell_data(kk)%unknowns%stresses(2))
        loc_rhs(4) = (temp - cell_data(i)%unknowns%stresses(2))/        &
		     (cell_data(i)%coordinates(2,1) -                   &
                      cell_data(i)%centre(2))
                      
        temp = extrapolate(cell_data(i)%coordinates(2,1),        &
                      cell_data(i)%centre(2),                           &
                      cell_data(i)%unknowns%stresses(3),                &
                      cell_data(k)%centre(2),                           &
                      cell_data(k)%unknowns%stresses(3),                &
                      cell_data(kk)%centre(2),                          &
                      cell_data(kk)%unknowns%stresses(3))
        loc_rhs(5) = (temp - cell_data(i)%unknowns%stresses(3))/        &
		     (cell_data(i)%coordinates(2,1) -                   &
                      cell_data(i)%centre(2))
                      

        end subroutine calc_boundary_stresses_4

	end module visco_bq
