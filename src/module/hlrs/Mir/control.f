!=====================================================================
!	This program is intended to provide some
!	basic routines to control handling
!	of data and data flow
!	Original author: Michael M. Resch (MR)
!	First released: 1. November 1994
!	First revision:
!			date: 15. December 1994
!			author: (MR)
!			reason: output files for wolfgang lohr's
!				triangulation programm
!
!	Second revision:
!			date: 30. December 1994
!			author: (MR)
!			reason: subroutine for testing the
!				divergence of the velocity
!				field and subroutine for
!				testing the pressure gradient
!
!	Third revision:
!			date: 8. March 1995
!			author: (MR)
!			reason: adding a short information
!				on development of variables
!
!=====================================================================
	module visco_control
	contains

!=====================================================================
        subroutine visco_check_B_i(B_i,B_i_p,B_i_m,info)
!=====================================================================

        use declaration
!	use visco_declaration

        implicit none
        real(double), dimension(5,5)    ::B_i
        real(double), dimension(5,5)    ::B_i_p
        real(double), dimension(5,5)    ::B_i_m

        integer                         :: i,j,info

        info = 0
        do i = 1,5
          do j = 1,5
            if(abs(B_i(i,j)-B_i_p(i,j)-B_i_m(i,j)) .gt. 1e-08) then
	      info = 1
!	      write(*,*) 'i,j =',i,j, B_i(i,j), B_i_p(i,j), B_i_m(i,j)
	    end if
          end do
        end do

        end subroutine visco_check_B_i

!=====================================================================
	subroutine visco_get_neighbours()
!=====================================================================
!	this subroutine writes all the neighbourhood relations
!	of the mesh given
!=====================================================================

	use visco_declaration
	
	implicit none

	integer		:: i,j
        
        open(60,file='neighbours.in')

	do j = 1,dim
	  write(60,*) 'cell  ',j
	  write(60,*) 'neighbours' , (cell_data(j)%neighbours(i),i=1,4)
	 !!write(60,'(I5)') (cell_data(j)%neighbours(i),i=1,4)
          write (60,*)
	end do

        close(60)

	end subroutine visco_get_neighbours

!=====================================================================
	subroutine visco_get_neighbours_1()
!=====================================================================
!	this subroutine writes all the neighbourhood relations
!	of the mesh given
!=====================================================================

	use visco_declaration
	
	implicit none

	integer		:: i,j

	do j = 1,dim
	  write(*,*) 'cell  ',j
	  write(*,*) 'neighbours'
	  write(*,'(4I4)') (cell_data_1(j)%neighbours(i),i=1,4)
	end do

	end subroutine visco_get_neighbours_1

!=====================================================================
	subroutine visco_get_mesh()
!=====================================================================
!       This subroutine writes out all information
!=====================================================================
!       This subroutine writes out all information
!	for the mesh:
!	neighbours of a cell
!	coordinates of points
!	normal vectors
!=====================================================================

	use visco_declaration
	
	implicit none

	integer		:: i,j

	do j = 1,dim
	  write(*,*) 'cell  ',j
	  write(*,*) 'neighbours'
	  write(*,'(4I4)') (cell_data(j)%neighbours(i),i=1,4)
	  write(*,*) 'x-ccordinates of points'
	  write(*,'(4E16.10)') (cell_data(j)%coordinates(1,i),i=1,4)
	  write(*,*) 'y-ccordinates of points'
	  write(*,'(4E16.10)') (cell_data(j)%coordinates(2,i),i=1,4)
	  write(*,*) 'centre point'
	  write(*,'(2E16.10)') (cell_data(j)%centre(i),i=1,2)
	  write(*,*) 'the normal vectors'
	  write(*,'(2E16.10)') (cell_data(j)%n_vectors(i,1),i=1,2)
	  write(*,'(2E16.10)') (cell_data(j)%n_vectors(i,2),i=1,2)
	  write(*,'(2E16.10)') (cell_data(j)%n_vectors(i,3),i=1,2)
	  write(*,'(2E16.10)') (cell_data(j)%n_vectors(i,4),i=1,2)
	end do

	end subroutine visco_get_mesh

!=====================================================================
	subroutine visco_write_outflow()
!=====================================================================
!	This subroutine writes the outflow values
!       of the computed results.
!=====================================================================
	use visco_declaration
        use arithmetic

	implicit none

	integer		:: i

        i = c_outflow_4 - c_outflow_2 -1
        write(*,*) i
        do i = c_outflow_2,c_outflow_4
          write(*,*) cell_data(i)%centre(2),cell_data(i)%unknowns%velocities(1)
        end do

	end subroutine visco_write_outflow

!=====================================================================
	subroutine visco_write_A()
!=====================================================================
!	This subroutine writes A
!=====================================================================
	use visco_declaration
        use arithmetic

	implicit none

	write(*,*) 'Matrix A'
        call print_matrix(A2)

	end subroutine visco_write_A

!=====================================================================
	subroutine visco_write_C()
!=====================================================================
!	This subroutine writes C
!=====================================================================
	use visco_declaration
        use arithmetic

	implicit none

	write(*,*) 'Matrix C'
        call print_matrix(C2)

	end subroutine visco_write_C

!=====================================================================
	subroutine visco_write_AC()
!=====================================================================
!	This subroutine writes AC
!=====================================================================
	use visco_declaration

	implicit none

	write(*,*) 'Matrix AC'
        call print_matrix(AC2)

	end subroutine visco_write_AC

!=====================================================================
	subroutine visco_write_q()
!=====================================================================
!	This subroutine writes q
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'results q'
	do i = 1,5*dim
	  write(*,'(E16.10)') q(i)
	end do

	end subroutine visco_write_q

!=====================================================================
	subroutine visco_write_r()
!=====================================================================
!	This subroutine writes r
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'right hand of equation r'
	do i = 1,5*dim
	  write(*,'(E16.10)') r(i)
	end do

	end subroutine visco_write_r

!=====================================================================
	subroutine visco_write_T()
!=====================================================================
!	This subroutine writes T
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) '===================================='
	write(*,*) 'right hand side of pressure equation'
	write(*,*) '===================================='
	do i = 1,dim
	  write(*,'(E16.10)') T(i)
	end do
	write(*,*) '===================================='
	write(*,*) ' '
	write(*,*) '===================================='

	end subroutine visco_write_T

!=====================================================================
	subroutine visco_write_vec_U()
!=====================================================================
!	This subroutine writes vector U
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) '============================'
	write(*,*) 'results of pressure equation'
	write(*,*) '============================'
	do i = 1,dim
	  write(*,'(E16.10)') U(i)
	end do
	write(*,*) '============================'
	write(*,*) ' '
	write(*,*) '============================'

	end subroutine visco_write_vec_U

!=====================================================================
	subroutine visco_write_Bq()
!=====================================================================
!	This subroutine writes Bq
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'Bq'
	do i = 1,5*dim
	  write(*,'(E16.10)') Bq(i)
	end do

	end subroutine visco_write_Bq

!=====================================================================
	subroutine visco_write_results()
!=====================================================================
!	This subroutine writes all results required by the user
!=====================================================================
        use visco_declaration
        use auxiliary

	implicit none

	integer		:: i,j,counter
	integer		:: sum
	character*25	:: locstring
	character*200	:: outstr

	sum = 0
        do i = 1,7
	  sum = sum+output_flags(i)
	end do
	if (sum .gt. 0) then
!=====================================================================
!	Write all results to a file 'values.out'
!   first 12 lines of old values.out file
!=====================================================================
	  open(50,file='values.out')

	  write(50,*)  '# values file'
	  write(50,*) 200/(x_max - x_min)
	  write(50,*) today
	  write(50,*) problem
	  write(50,*) sum
	  if (output_flags(1) .eq. 1) then
	    write(50,'(3(E21.11),a)') 					&
			p_min,p_max, (p_max-p_min)/10,' pressure'
	  end if
	  if (output_flags(2) .eq. 1) then
            write(50,'(3(E21.11),a)') 					&
			u_min,u_max, (u_max-u_min)/10,' u-velocity'
	  end if
	  if (output_flags(3) .eq. 1) then
            write(50,'(3(E21.11),a)') 					&
			v_min,v_max, (v_max-v_min)/10,' v-velocity'
	  end if
	  if (output_flags(4) .eq. 1) then
            write(50,'(3(E21.11),a)') 					&
			sigma_min,sigma_max, (sigma_max-sigma_min)/10, ' sigma'
	  end if
	  if (output_flags(5) .eq. 1) then
            write(50,'(3(E21.11),a)') 					&
			tau_min,tau_max, (tau_max-tau_min)/10,' tau'
	  end if
	  if (output_flags(6) .eq. 1) then
            write(50,'(3(E21.11),a)') 					&
                         gamma_min,gamma_max, (gamma_max-gamma_min)/10,' gamma'
          end if
!!          cot_min = 0.01
!!          cot_max = 0.02
          if (output_flags(7) .eq. 1) then
            write(50,'(3(E21.11),a)') 					&
              cot_min,cot_max, (cot_max-cot_min)/10,' cot'
	  end if

      close(50)

!=====================================================================
!   Write all results to a file 'result.out'
!
!=====================================================================
      open(55,file='result.out')

	  write(55,*) dim
	  do i = 1,dim
	    write(locstring,*) i-1
	    outstr = locstring(1:10)
	    counter = 10
	    if (output_flags(1) .eq. 1) then
                write(locstring,'(a,E21.11)')'  ',cell_data(i)%pressure
                outstr = outstr(1:counter)//locstring(1:25)
                counter = counter+25
	    end if
            do j = 2,6
	      if (output_flags(j) .eq. 1) then
		write(locstring,'(a,E21.11)')'  ', 			&
			cell_data(i)%unknowns%velocities(j-1)
		outstr = outstr(1:counter)//locstring(1:25)
		counter = counter+25
	      end if
            end do
            if (output_flags(7).eq. 1) then
              write(locstring,'(a,E21.11)')'  ',                      &
!	       (q_old((i-1)*5+1)-q((i-1)*5+1))**2

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!! write change of type to file result.out
!!
!! heikex 980206
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

              type_change(i)
              outstr = outstr(1:counter)//locstring(1:25)
              counter = counter+25
            end if
	    write(55,*) outstr
	  end do
	  close(55)


          deallocate(type_change)
!=====================================================================
!	Write all residua to a file 'res.out'
!=====================================================================
	  open(60,file='res.out')
	  write(60,*) 200/(x_max - x_min)
	  write(60,*) today
	  write(60,*) problem
          write(60,*) sum - output_flags(1) - output_flags(7)
	  if (output_flags(2) .eq. 1) then
            write(60,'(3(E21.11),a)') 					&
			res1_min,res1_max, (res1_max-res1_min)/10,' res_u'
	  end if
	  if (output_flags(3) .eq. 1) then
            write(60,'(3(E21.11),a)') 					&
			res2_min,res2_max, (res2_max-res2_min)/10,' res_v'
	  end if
	  if (output_flags(4) .eq. 1) then
            write(60,'(3(E21.11),a)') 					&
			res3_min,res3_max, (res3_max-res3_min)/10,' res_s'
	  end if
	  if (output_flags(5) .eq. 1) then
            write(60,'(3(E21.11),a)') 					&
			res4_min,res4_max, (res4_max-res4_min)/10,' res_t'
	  end if
	  if (output_flags(6) .eq. 1) then
            write(60,'(3(E21.11),a)') 					&
			res5_min,res5_max, (res5_max-res5_min)/10,' res_g'
	  end if

	  write(60,*) dim
	  do i = 1,dim
	    write(locstring,*) i-1
	    outstr = locstring(1:10)
	    counter = 10
	    do j = 2,6
	      if (output_flags(j) .eq. 1) then
		write(locstring,'(a,E21.11)')'  ', 			&
			q(i+j-2) - q_old(i+j-2)
		outstr = outstr(1:counter)//locstring(1:25)
		counter = counter+25
	      end if
	    end do
	    write(60,*) outstr
	  end do
	  close(60)
	end if

	end subroutine visco_write_results

!=====================================================================
	subroutine visco_write_vectors()
!=====================================================================
!	This subroutine writes all velocity vectors
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i
	
	open(99,file='vectors.out')
	write(99,*) 10.0/sqrt(u_max**2+v_max**2)
	write(99,*) dim
	do i = 1,dim
	  write(99,*) cell_data(i)%centre(1),			&
	  	      cell_data(i)%centre(2),			&
		      cell_data(i)%unknowns%velocities(1),	&
		      cell_data(i)%unknowns%velocities(2)
	end do
	close(99)
	
	end subroutine visco_write_vectors
	  

!=====================================================================
	subroutine visco_write_u()
!=====================================================================
!	This subroutine writes u and centre_values
!	so to allow 3D-plotting
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'u'
	do i = 1,dim
	  write(*,*) cell_data(i)%centre(1),			&
		     cell_data(i)%centre(2),			&
		     cell_data(i)%unknowns%velocities(1)
	end do

	end subroutine visco_write_u
!=====================================================================
	subroutine visco_write_v()
!=====================================================================
!	This subroutine writes v and centre_values
!	so to allow 3D-plotting
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'v'
	do i = 1,dim
	  write(*,*) cell_data(i)%centre(1),			&
		     cell_data(i)%centre(2),			&
		     cell_data(i)%unknowns%velocities(2)
	end do

	end subroutine visco_write_v

!=====================================================================
	subroutine visco_write_tau()
!=====================================================================
!	This subroutine writes tau and centre_values
!	so to allow 3D-plotting
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'tau'
	do i = 1,dim
	  write(*,*) cell_data(i)%centre(1),			&
		     cell_data(i)%centre(2),			&
		     cell_data(i)%unknowns%stresses(2)
	end do

	end subroutine visco_write_tau

!=====================================================================
	subroutine visco_write_gamma()
!=====================================================================
!	This subroutine writes gamma and centre_values
!	so to allow 3D-plotting
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'gamma'
	do i = 1,dim
	  write(*,*) cell_data(i)%centre(1),			&
		     cell_data(i)%centre(2),			&
		     cell_data(i)%unknowns%stresses(3)
	end do

	end subroutine visco_write_gamma

!=====================================================================
	subroutine visco_write_sigma()
!=====================================================================
!	This subroutine writes sigma and centre_values
!	so to allow 3D-plotting
!=====================================================================
	use visco_declaration

	implicit none

	integer		:: i

	write(*,*) 'sigma'
	do i = 1,dim
	  write(*,*) cell_data(i)%centre(1),			&
		     cell_data(i)%centre(2),			&
		     cell_data(i)%unknowns%stresses(1)
	end do

	end subroutine visco_write_sigma

!=====================================================================
	subroutine visco_calc_Aq()
!=====================================================================
!	This subroutine checks whether Aq = r
!	Only for inflow cells there should
!	be non-zero elements in r
!=====================================================================
	use visco_declaration
        use auxiliary

        implicit none

        integer :: i,inco

        inco = 0
        T2 = A2*q2

       !! -------------------------------------
       !!heikex c_b(1) do not longer exist
       !! ----------------------------------
!!        T = T2%c_b(1)%container%field

        T => T2%container%field

        do i = 1,dim
          if (T(i) > 1e-8) then
            if (inco == 0) then
              write(*,*) '==============================================='
              write(*,*) 'Incompressibility hurt in'
              write(*,*) '==============================================='
              inco = 1
            end if
            if (i <= c_inflow_2) then
              T(i) = T(i) + parabel(cell_data(i)%centre(2))*                &
                            cell_data(i)%n_vectors(1,3)/cell_data(i)%volume
              if (T(i) > 1e-8) write(*,*) i,T(i)
            else
              write(*,*) i,T(i)
	    end if
          end if
        end do
        
        end subroutine visco_calc_Aq
        
!=====================================================================
	subroutine visco_calc_Cp()
!=====================================================================
!	This subroutine checks whether Cp is constant
!=====================================================================
	use visco_declaration

        implicit none

        integer :: i

        S2 = C2*p2
        !! ---------------------------------------------
        !! heikex :: same here c_b(1) do not exist
        !!S = S2%c_b(1)%container%field
        S => S2%container%field
       
        write(*,*) '==============================================='
        write(*,*) 'Pressure gradient '
        write(*,*) '==============================================='
        do i = 1,dim*5,5
          write(*,*) (i+4)/5,' p_x = ',S(i)
          write(*,*) (i+4)/5,' p_y = ',S(i+1)
        end do
        
	end subroutine visco_calc_Cp

!=====================================================================
	subroutine visco_calc_div_u()
!=====================================================================
!	This subroutine calculates the divergence of the
!	velocity field
!	
!	Attention: This subroutine works only on rectangular
!		   grids since it makes use of neighbours.
!=====================================================================
        use declaration
	use visco_declaration

	implicit none

	real(double)	:: v_y,delta_y
	real(double) 	:: u_x,delta_x

	integer		:: i

!=====================================================================
!	Inflow lower corner
!=====================================================================
	i = 1
	u_x = 0.0
	v_y = 0.0
!=====================================================================
!	calculate u_x
!=====================================================================
	delta_x = 							     &
	cell_data(cell_data(i)%neighbours(1))%centre(1) -		     &
	cell_data(i)%centre(1)
	u_x =							     	& 
        (cell_data(cell_data(i)%neighbours(1))%unknowns%velocities(1) - &
        cell_data(i)%unknowns%velocities(1))/delta_x
!=====================================================================
!	calculate v_y
!=====================================================================
	delta_y = 							&
	cell_data(cell_data(i)%neighbours(2))%centre(2) -		&
	cell_data(i)%centre(2)
	v_y =							     	& 
        (cell_data(cell_data(i)%neighbours(2))%unknowns%velocities(2) - &
        cell_data(i)%unknowns%velocities(2))/delta_y
	write(*,*) 'Inflow lower corner'
	write(*,*) i,u_x+v_y

	write(*,*) 'Inflow boundary'
	do i = 2,inflow
	  u_x = 0.0
	  v_y = 0.0
!=====================================================================
!	calculate u_x
!=====================================================================
	  delta_x = 							     &
	  cell_data(cell_data(i)%neighbours(1))%centre(1) -		     &
	  cell_data(i)%centre(1)
	  u_x =							     	& 
          (cell_data(cell_data(i)%neighbours(1))%unknowns%velocities(1) - &
          cell_data(i)%unknowns%velocities(1)) &
	  /delta_x
!=====================================================================
!	calculate v_y
!=====================================================================
	  delta_y = 							&
	  cell_data(cell_data(i)%neighbours(2))%centre(2) -		&
	  cell_data(cell_data(i)%neighbours(4))%centre(2)
	  v_y =							     	& 
          (cell_data(cell_data(i)%neighbours(2))%unknowns%velocities(2) - &
          cell_data(cell_data(i)%neighbours(4))%unknowns%velocities(2))  &
	  /delta_y
	  write(*,*) i,u_x+v_y
	end do

	write(*,*) 'inner cells'
!=====================================================================
!	Loop over all inner cells to calculate the values
!	u_x and v_y
!=====================================================================
	do i = c_wall_23,dim
	  u_x = 0.0
	  v_y = 0.0
!=====================================================================
!	calculate u_x
!=====================================================================
	  delta_x = 							     &
	  cell_data(cell_data(i)%neighbours(1))%centre(1) -		     &
	  cell_data(cell_data(i)%neighbours(3))%centre(1)
	  u_x =							     	& 
          (cell_data(cell_data(i)%neighbours(1))%unknowns%velocities(1) - &
          cell_data(cell_data(i)%neighbours(3))%unknowns%velocities(1)) &
	  /delta_x
	  
!=====================================================================
!	calculate v_y
!=====================================================================
	  delta_y = 							&
	  cell_data(cell_data(i)%neighbours(2))%centre(2) -		&
	  cell_data(cell_data(i)%neighbours(4))%centre(2)
	  v_y =							     	& 
          (cell_data(cell_data(i)%neighbours(2))%unknowns%velocities(2) - &
          cell_data(cell_data(i)%neighbours(4))%unknowns%velocities(2))  &
	  /delta_y
	  write(*,*) i,u_x+v_y
	end do
	end subroutine visco_calc_div_u

!=====================================================================
	subroutine visco_check_alpha()
!=====================================================================
!	This subroutine checks whether alphas have changed
!=====================================================================
        use declaration
        use visco_declaration
        use auxiliary
        use visco_init_1

	implicit none

        integer	:: i,j,num_sides
        real(double), dimension(8) :: alpha
	real(double) :: factor

        check = 0
        do i = 1,dim
          factor = 0.0d0
          num_sides = cell_data(i)%num_sides
          do j = 1,num_sides
            factor    = cell_data(i)%n_vectors(1,j)*                     &
                        cell_data(i)%unknowns%velocities(1) +            &
                        cell_data(i)%n_vectors(2,j)*                     &
                        cell_data(i)%unknowns%velocities(2)
            if (abs(factor) > 0.0) then
              alpha(j)  = int(sign(1.0d0,factor))
            else
	      if (j .eq. 2) then
	        alpha(j) =  1.0
	      else if (j .eq. 4) then
	        alpha(j) = -1.0
	      else if (j .eq. 1) then
	        alpha(j) = -1.0
	      else if (j .eq. 3) then
	        alpha(j) = 1.0
              end if
            end if
!=========================================================================
!	if we want to have central differences
!=========================================================================
!	    alpha(j) = 0.0d0
          end do
          do j = 1,num_sides
            if (cell_data(i)%alphas(j) .ne. alpha(j)) check = 1
          end do
        end do

        write(*,*) 'check necessary'
	end subroutine

	end module visco_control
