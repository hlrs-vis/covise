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
!			date: 1.7.1995
!			author: (MR)
!			reason: when distributing results onto
!				cells calculate minima and maxima
!				of values.
!       Second revision:
!                       date: 20.October 1995
!                       author: (MR)
!                       reason: calculate correctly boundary
!                               conditions for stresses
!                               from velocities
!=====================================================================
	module visco_kernel
	contains

!=====================================================================
	subroutine visco_correct_q()
!=====================================================================
!    this subroutine takes an existing q and corrects it
!    to get a divergence free vector field.
!=====================================================================

	use visco_declaration
        use mod_ma28
	use auxiliary
        use visco_control

	implicit none

        integer :: i

        T2=A2*q2

        !! --------------------------
        !! heikex 970918
        !! --------------------------
        !!T=>T2%c_b(1)%container%field
 
        T => T2%container%field

	do i = 1,c_inflow_2
	  T(i) = T(i) + parabel(cell_data(i)%centre(2))*		&
			cell_data(i)%n_vectors(1,3)/cell_data(i)%volume
        end do
        T2 = T
        call ma28c(dim,AC2,T,pivot,info)
        T2 = T
        q_old2 = q2-C2*T2

        !! -------------------------
        !! heikex 970918
        !! ------------------------
        !!        q_old=>q_old2%c_b(1)%container%field
        q_old=>q_old2%container%field

        q=q_old
        call visco_calc_Aq()

        end subroutine visco_correct_q
      
!=====================================================================
	subroutine visco_calc_R()
!=====================================================================
!	This subroutine calculates the right hand side
!	of the second equation, which consists of
!	only stresses
!=====================================================================

	use visco_declaration

	implicit none

	integer		:: i

        do i = 1,5*dim,5
	  r(i)		= 0.0
	  r(i+1)	= 0.0
	  r(i+2)	= -cell_data((i+4)/5)%unknowns%stresses(1)/lambda
	  r(i+3)	= -cell_data((i+4)/5)%unknowns%stresses(2)/lambda
          r(i+4)	= -cell_data((i+4)/5)%unknowns%stresses(3)/lambda
	end do

	end subroutine visco_calc_r

!=====================================================================
	subroutine visco_calc_S()
!=====================================================================
!	This subroutine calculates the auxiliary vector S
!       for the method with correction.
!
!	S = q+dtR-dtBq
!
!=====================================================================

	use visco_declaration

	implicit none

	integer		:: i

	do i = 1,5*dim
	  S(i) =  q(i) + delta_t*r(i) - delta_t*Bq(i)
	end do
        S2 = S

      end subroutine visco_calc_S
      
!=====================================================================
      subroutine visco_calc_S_1()
!=====================================================================
!	This subroutine calculates the auxiliary vector S
!       for the method without correction.
!
!	S = dtR-dtBq
!
!=====================================================================

	use visco_declaration

	implicit none

	integer		:: i


	do i = 1,5*dim
	  S(i) =  delta_t*r(i) - delta_t*Bq(i)
        end do
        S2 = S

      end subroutine visco_calc_S_1

!=====================================================================
	subroutine visco_calc_T()
!=====================================================================
!	This subroutine calculates the auxiliary vector T
!
!	T = A S - r
!
!=====================================================================

	use visco_declaration
        use arithmetic
	use auxiliary

	implicit none

	integer		:: i

! Uwe Laun: Die Matrix-Vektor-Multiplikation wird durch das Arithmetic-Modul
!           erledigt.

        T2=A2*S2
!        deallocate(T)

! Uwe Laun: Um Elemente des Vektors wie gehabt aendern zu koennen, wird
!           der Pointer T aud das Elementfeld des sparse-Vektors T2 gesetzt.

        !! ----------------------
        !! heikex  970918
        !! ----------------------
        !!T=>T2%c_b(1)%container%field
        T=>T2%container%field

!	call visco_write_T()
!=====================================================================
!	calculation of boundary-values-vector
!=====================================================================
	do i = 1,c_inflow_2
	  T(i) = T(i) + parabel(cell_data(i)%centre(2))*		&
			cell_data(i)%n_vectors(1,3)/cell_data(i)%volume
        end do
        T2 = T
        
	end subroutine visco_calc_T
!=====================================================================
        subroutine visco_calc_T_1()
!=====================================================================
!	This subroutine calculates the auxiliary vector T
!       for the version without correction.
!
!	T = A S
!
!=====================================================================

	use visco_declaration
        use arithmetic
	use auxiliary
!============================= remove urgently
	use visco_control

	implicit none

! Uwe Laun: Die Matrix-Vektor-Multiplikation wird durch das Arithmetic-Modul
!           erledigt.

        T2=A2*S2
!        deallocate(T)

! Uwe Laun: Um Elemente des Vektors wie gehabt aendern zu koennen, wird
!           der Pointer T aud das Elementfeld des sparse-Vektors T2 gesetzt.

        !! --------------------------
        !! heikex  970918
        !! --------------------------
        !!T=>T2%c_b(1)%container%field
        T=>T2%container%field


      end subroutine visco_calc_T_1

!=====================================================================
	subroutine visco_solve_AC()
!=====================================================================
!	This subroutine solves for AC
!=====================================================================

	use visco_declaration

	implicit none

	end subroutine visco_solve_AC

!=====================================================================
	subroutine visco_calc_q()
!=====================================================================
!	This subroutine calculates the new q
!       for the version with correction of q.
!=====================================================================

	use visco_declaration
        use arithmetic
	use visco_smooth

	implicit none

	integer	        :: i,inorm
        real(double)	:: norm
        
!=====================================================================
!	store old time_step
!=====================================================================
        !! -----------------------
        !! heikex  970918
        !! -----------------------
        !! q=>q2%c_b(1)%container%field
        q=>q2%container%field
        q_old = q
	norm = 0.0d0
	q = 0.0d0
!!==========================================================================
! Uwe Laun: Die Matrix-Vektor-Multiplikation wird durch das Arithmetic-Modul
!           erledigt.
!!==========================================================================
        
        q2=C2*U2
        
!!==========================================================================
! Uwe Laun: Um Elemente des Vektors wie gehabt aendern zu koennen, wird
!           der Pointer T aud das Elementfeld des sparse-Vektors T2 gesetzt.
!!==========================================================================

        !! ------------------------
        !! heikex  970918
        !! ------------------------
        !!q=>q2%c_b(1)%container%field        
        q=>q2%container%field        
        
        q =  S - q
        q2 = q

        norm = 0.0d0
        inorm = 0
        do i = 1,5*dim
          if (norm .lt. abs(q_old(i)-q(i))) then
            norm = abs(q_old(i)-q(i))
            inorm = i
          end if
        end do
        write(*,*) 'norm = ',norm,' in ',inorm

!        call smooth_corners()

        do i = 1,5*dim
          q(i) = ((1+relax)*q(i)+(1-relax)*q_old(i))/2.0d0
        end do
        q2 = q

      end subroutine visco_calc_q
      
!=====================================================================
      subroutine visco_calc_q_1()
!=====================================================================
!	This subroutine calculates the new q
!       for the verson without correction of q.
!=====================================================================

	use visco_declaration
        use arithmetic

	implicit none

	integer	:: i
        real(double)	:: norm
        real(double), dimension(C2%maxrow)  :: yy
        integer :: k,ii
!=====================================================================
!	store old time_step
!=====================================================================
	q_old = q
	
	norm = 0.0d0
	q = 0.0d0

!!==========================================================================
! Uwe Laun: Die Matrix-Vektor-Multiplikation wird durch das Arithmetic-Modul
!           erledigt.
!!==========================================================================

!!==========================================================================
!! this part is replaced by the original library code since
!! the library has problems handling a dynamic function result
!!==========================================================================

        !! -------------------------
        !! heikex  970918
        !! -------------------------
        !!call intern_add(U2)

        yy=0.
        do k=1,C2%maxng
          do i=1,C2%container%ng(k)
            ii=i+C2%container%offset(k)
            yy(i)=yy(i)+C2%container%element(ii)*&
            !! ----------------------
            !! heikex 970918
            !! ----------------------
            !! U2%c_b(1)%container%field(C2%container%column_number(ii))
             U2%container%field(C2%container%column_number(ii))
          end do
        end do

        do i=1,q2%dim
        !! --------------------
        !! heikex  970918
        !! --------------------
        !!  q2%c_b(1)%container%field(i)=yy(C2%container%permutation(i))* &
        !!  U2%c_b(1)%scal
        q2%container%field(i)=yy(C2%container%permutation(i))

        end do

        !! ----------------------
        !! heikex  970918
        !! ----------------------       
        !!q2%c_b(1)%scal=1.0
        !!q2%charge=1
        
        
!        q2=C2*U2
        
!!==========================================================================
! Uwe Laun: Um Elemente des Vektors wie gehabt aendern zu koennen, wird
!           der Pointer T aud das Elementfeld des sparse-Vektors T2 gesetzt.
!!==========================================================================
        !! -------------------
        !! heikex  970918
        !! ------------------
        !!q=>q2%c_b(1)%container%field
        q=>q2%container%field
        
        
        q = (-1)*q + S + q_old

        do i = 1,5*dim
          if (q_old(i) < 1E-10) then
            if (q(i) /= 0.0) norm = norm + dabs(q(i))
          else
!           norm = norm + ((q_old(i)-q(i))/q_old(i))**2
            norm = norm + (q_old(i)-q(i))**2
          end if
        end do
        norm = sqrt(norm)/dim
        write(*,*) 'norm = ',norm
        

      end subroutine visco_calc_q_1

!=====================================================================
	subroutine visco_distribute_results()
!=====================================================================
!	This subroutine distributes results
!=====================================================================

        use auxiliary
        use visco_declaration

	implicit none

	integer		:: i

!=====================================================================
!	calculate minimum and maxium pressure
!=====================================================================
	p_min 	= 1d+100
	p_max 	= -1d100
	do i = 1,dim
	  cell_data(i)%pressure	= p(i)
	  p_max = max(p_max,p(i))
	  p_min = min(p_min,p(i))
	end do
	u_min 	= 1d100
	u_max 	= -1d100
	v_min 	= 1d100
	v_max 	= -1d100
	sigma_min 	= 1d100
	sigma_max 	= -1d100
	tau_min 	= 1d100
	tau_max 	= -1d100
	gamma_min 	= 1d100
        gamma_max 	= -1d100
        cot_min         = 1d100
        cot_max         = -1d100
	res1_min	= 1d100
	res1_max	= -1d100
	res2_min	= 1d100
	res2_max	= -1d100
	res3_min	= 1d100
	res3_max	= -1d100
	res4_min	= 1d100
	res4_max	= -1d100
	res5_min	= 1d100
	res5_max	= -1d100

        !! ----------------------
        !! heikex  970918
        !! ----------------------
        !!q=>q2%c_b(1)%container%field
        q=>q2%container%field
	do i = 1,5*dim,5
	  cell_data((i+4)/5)%unknowns%velocities(1) = q(i)
	  cell_data((i+4)/5)%unknowns%velocities(2) = q(i+1)
	  cell_data((i+4)/5)%unknowns%stresses(1)   = q(i+2)
	  cell_data((i+4)/5)%unknowns%stresses(2)   = q(i+3)
	  cell_data((i+4)/5)%unknowns%stresses(3)   = q(i+4)
	  u_min     = min(u_min,q(i))
	  u_max     = max(u_max,q(i))
	  v_min     = min(v_min,q(i+1))
	  v_max     = max(v_max,q(i+1))
	  sigma_min = min(sigma_min,q(i+2))
	  sigma_max = max(sigma_max,q(i+2))
	  tau_min   = min(tau_min,q(i+3))
	  tau_max   = max(tau_max,q(i+3))
	  gamma_min = min(gamma_min,q(i+4))
          gamma_max = max(gamma_max,q(i+4))

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!
!! compute change of type and select min value and max value
!!
!! heikex 980206
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
          type_change((i+4)/5) =                           &
                                  (    q(i+3)**2                                     &      
                                     - 2 * q(i+3) * q(i) * q(i+1)                    &
                                     - ( viscosity +q(i+4) )*( viscosity + q(i+2)  ) &
                                     + q(i+1)**2 *( q(i+2) + viscosity )             &
                                     + q(i)**2 * ( viscosity + q(i+4) )              & 
                                  )                        &
                                /                          &
                                 (     ( q(i+2) + viscosity - q(i)**2 )**2           &
                                 ) 

          cot_min   = min(cot_max,type_change((i+4)/5))
          cot_max   = max(cot_max,type_change((i+4)/5))

	  res1_min  = min(res1_min,(q_old(i)-q(i)))
	  res1_max  = max(res1_min,(q_old(i)-q(i)))
	  res2_min  = min(res2_min,(q_old(i+1)-q(i+1)))
	  res2_max  = max(res2_min,(q_old(i+1)-q(i+1)))
	  res3_min  = min(res3_min,(q_old(i+2)-q(i+2)))
	  res3_max  = max(res3_min,(q_old(i+2)-q(i+2)))
	  res4_min  = min(res4_min,(q_old(i+3)-q(i+3)))
	  res4_max  = max(res4_min,(q_old(i+3)-q(i+3)))
	  res5_min  = min(res5_min,(q_old(i+4)-q(i+4)))
	  res5_max  = max(res5_min,(q_old(i+4)-q(i+4)))
	end do

	end subroutine visco_distribute_results

!=====================================================================
	subroutine visco_calc_evals(i, direction,alpha,beta,velocity,	&
				    eigen_value)
!=====================================================================
!	This subroutine calculates the eigenvalues of the B
!	matrices used in the discretisation.
!	direction	integer indicating whether x (1) or y(2)
!			direction is to be calculated
!=====================================================================

	use declaration
	use visco_declaration

	implicit none
	integer		:: i,direction
	real(double)	:: alpha, beta,velocity
	real(double), dimension(5)	:: eigen_value

!=====================================================================
!	x-direction
!=====================================================================
	eigen_value = 0.0
	if (direction .eq. 1) then
	  velocity = cell_data(i)%unknowns%velocities(1)
          alpha = viscosity/lambda + cell_data(i)%unknowns%stresses(1)
	  if (alpha .ge. 0.0) then
            eigen_value(1) = velocity
            eigen_value(2) = velocity - sqrt(2*alpha)
            eigen_value(3) = velocity + sqrt(2*alpha)
            eigen_value(4) = velocity - sqrt(alpha)
            eigen_value(5) = velocity + sqrt(alpha)
	  else
	    write(*,*) 'alpha < 0 in ',i
	  end if
	else
          velocity = cell_data(i)%unknowns%velocities(2)
          beta = viscosity/lambda + cell_data(i)%unknowns%stresses(3)
	  if (beta .ge. 0.0) then
            eigen_value(1) = velocity
            eigen_value(2) = velocity + sqrt(2*beta)
            eigen_value(3) = velocity - sqrt(2*beta)
            eigen_value(4) = velocity + sqrt(beta)
            eigen_value(5) = velocity - sqrt(beta)
	  else
	    write(*,*) 'beta < 0 in ',i
	  end if
	end if

	end subroutine visco_calc_evals

!=====================================================================
	subroutine visco_calc_B_matrices(i,direction,velocity,		&
					 eigen_value,alpha,beta,	&
					 B_i,B_1,B_2,B_3,B_4,B_5)
!=====================================================================
!	This subroutine calculates B-matrices
!=====================================================================

	use visco_declaration

	implicit none

	integer				:: i,direction
	real(double)			:: velocity
        real(double), dimension(5)      :: eigen_value
	real(double)			:: alpha,beta
        real(double), dimension(5,5)    :: B_i
        real(double), dimension(5,5)    :: B_1,B_2,B_3,B_4,B_5

!=====================================================================
!	direction = 1 	=>	u-direction
!=====================================================================
	if (direction .eq. 1) then
!=====================================================================
!       calculation of overall matrix B_i need not to be done
!       in any case. however here it is done to check for
!       the correctness of B+ and B-.
!=====================================================================
          B_i      = 0.0
          B_i(1,1) = velocity
          B_i(1,3) = -1.0
          B_i(2,2) = velocity
          B_i(2,4) = -1.0
          B_i(3,1) = -2*alpha
          B_i(3,3) = velocity
          B_i(4,2) = -alpha
          B_i(4,4) = velocity
          B_i(5,2) = -2.0*cell_data(i)%unknowns%stresses(2)
          B_i(5,5) = velocity

	  if (alpha .gt. 0.0) then
!=====================================================================
!       caluclate B_1
!=====================================================================
            B_1 = 0.0
            B_1(5,4) = -eigen_value(1)*                           &
                         2.0*cell_data(i)%unknowns%stresses(2)/alpha
            B_1(5,5) = eigen_value(1)
!=====================================================================
!       caluclate B_2
!=====================================================================
            B_2 = 0.0
            B_2(1,1) = eigen_value(2)/2.0
            B_2(1,3) = eigen_value(2)/(2.0*sqrt(2*alpha))
            B_2(3,1) = eigen_value(2)*sqrt(alpha)/(sqrt(2.0))
            B_2(3,3) = eigen_value(2)/2.0

!=====================================================================
!       caluclate B_3
!=====================================================================
            B_3 = 0.0
            B_3(1,1) = eigen_value(3)/2.0
            B_3(1,3) = -eigen_value(3)/(2.0*sqrt(2*alpha))
            B_3(3,1) = -eigen_value(3)*sqrt(alpha)/(sqrt(2.0))
            B_3(3,3) = eigen_value(3)/2.0

!=====================================================================
!       caluclate B_4
!=====================================================================
            B_4 = 0.0
            B_4(2,2) = eigen_value(4)/2.0
            B_4(2,4) = eigen_value(4)/(2.0*sqrt(alpha))
            B_4(4,2) = eigen_value(4)*sqrt(alpha)/2.0
            B_4(4,4) = eigen_value(4)/2.0
            B_4(5,2) = eigen_value(4)*cell_data(i)%unknowns%stresses(2)/ &
                         sqrt(alpha)
            B_4(5,4) = eigen_value(4)*cell_data(i)%unknowns%stresses(2)/ &
                         alpha

!=====================================================================
!       caluclate B_5
!=====================================================================
            B_5 = 0.0
            B_5(2,2) = eigen_value(5)/2.0
            B_5(2,4) = -eigen_value(5)/(2.0*sqrt(alpha))
            B_5(4,2) = -eigen_value(5)*sqrt(alpha)/2.0
            B_5(4,4) = eigen_value(5)/2.0
            B_5(5,2) = -eigen_value(5)*cell_data(i)%unknowns%stresses(2)/ &
                         sqrt(alpha)
            B_5(5,4) = eigen_value(5)*cell_data(i)%unknowns%stresses(2)/ &
                         alpha
	  end if
	else

          B_i = 0.0
          B_i(1,1) = velocity
          B_i(1,4) = -1.0
          B_i(2,2) = velocity
          B_i(2,5) = -1.0
          B_i(3,1) = -2*cell_data(i)%unknowns%stresses(2)
          B_i(3,3) = velocity
          B_i(4,1) = -beta
          B_i(4,4) = velocity
          B_i(5,2) = -2*beta
          B_i(5,5) = velocity

	  if (beta .gt. 0.0) then
!=====================================================================
!       caluclate B_1
!=====================================================================
            B_1 = 0.0
            B_1(3,3) = eigen_value(1)
            B_1(3,4) = -eigen_value(1)*                               &
                         2.0*cell_data(i)%unknowns%stresses(2)/beta
!=====================================================================
!       caluclate B_2
!=====================================================================
            B_2 = 0.0
            B_2(2,2) = eigen_value(2)/2.0
            B_2(2,5) = -eigen_value(2)/(2*sqrt(2*beta))
            B_2(5,2) = -eigen_value(2)*sqrt(beta)/sqrt(2.0)
            B_2(5,5) = eigen_value(2)/2.0

!=====================================================================
!       caluclate B_3
!=====================================================================
            B_3 = 0.0
            B_3(2,2) = eigen_value(3)/2.0
            B_3(2,5) = eigen_value(3)/(2*sqrt(2*beta))
            B_3(5,2) = eigen_value(3)*sqrt(beta)/sqrt(2.0)
            B_3(5,5) = eigen_value(3)/2.0

!=====================================================================
!       caluclate B_4
!=====================================================================
            B_4 = 0.0
            B_4(1,1) = eigen_value(4)/2.0
            B_4(1,4) = -eigen_value(4)/(2*sqrt(beta))
            B_4(3,1) = -eigen_value(4)*cell_data(i)%unknowns%stresses(2)/ &
                                        sqrt(beta)
            B_4(3,4) = eigen_value(4)*cell_data(i)%unknowns%stresses(2)/ &
                                        beta
            B_4(4,1) = -eigen_value(4)*sqrt(beta)/2.0
            B_4(4,4) = eigen_value(4)/2.0

!=====================================================================
!       caluclate B_5
!=====================================================================
            B_5 = 0.0
            B_5(1,1) = eigen_value(5)/2.0
            B_5(1,4) = eigen_value(5)/(2*sqrt(beta))
            B_5(3,1) = eigen_value(5)*cell_data(i)%unknowns%stresses(2)/ &
                                        sqrt(beta)
            B_5(3,4) = eigen_value(5)*cell_data(i)%unknowns%stresses(2)/ &
                                        beta
            B_5(4,1) = eigen_value(5)*sqrt(beta)/2.0
            B_5(4,4) = eigen_value(5)/2.0

	  end if
	end if

	end subroutine visco_calc_B_matrices

!=====================================================================
	subroutine visco_calc_B_m_p(direction,alpha,beta,		&
				    eigen_value,velocity,		&
				    B_1,B_2,B_3,B_4,B_5,		&
				    B_i_m, B_i_p,B_i)
!=====================================================================
!	This subroutine calculates the matrices
!	B_i_m, B_i_p for splitting of discretization
!=====================================================================

	use visco_declaration
	use visco_control

	implicit none

	integer			:: direction
	real(double)		:: alpha,beta
	real(double), dimension(5)	:: eigen_value
	real(double)			:: velocity
	real(double), dimension(5,5)	:: B_1,B_2,B_3,B_4,B_5
	real(double), dimension(5,5)	:: B_i_m,B_i_p
	real(double), dimension(5,5)	:: B_i

	B_i_m = 0.0
	B_i_p = 0.0

	if (direction .eq. 1) then
          if (alpha .le. 0.0) then
!=====================================================================
!       in case of complex eigenvalues the real parts
!       of the eigenvalues all have the same sign (the sign of the
!       velocity u).
!=====================================================================
            if (velocity .ge. 0.0) then
              B_i_p = B_i
            else
              B_i_m = B_i
            end if
          else
            if (velocity .ge. 0.0) then
              if (eigen_value(2) .ge. 0.0) then
                B_i_p = B_1+B_2+B_3+B_4+B_5
              else
                if (eigen_value(4) .ge. 0.0) then
                  B_i_m = B_2
                  B_i_p = B_1+B_3+B_4+B_5
                else
                  B_i_m = B_2+B_4
                  B_i_p = B_1+B_3+B_5
                end if
              end if
            else
              if (eigen_value(5) .ge. 0.0) then
                B_i_p = B_3+B_5
                B_i_m = B_1+B_2+B_4
              else
                if (eigen_value(3) .ge. 0) then
                  B_i_p = B_3
                  B_i_m = B_1+B_2+B_4+B_5
                else
                  B_i_m = B_1+B_2+B_3+B_4+B_5
                end if
              end if
            end if
          end if
	else
	  if (beta .le. 0.0) then
	    if (velocity .ge. 0.0) then
	      B_i_p = B_i
	    else
	      B_i_m = B_i
	    end if
	  else
            if (velocity .ge. 0.0) then
              if (eigen_value(3) .ge. 0.0) then
                B_i_p = B_1+B_2+B_3+B_4+B_5
              else
                if (eigen_value(5) .ge. 0.0) then
                  B_i_m = B_3
                  B_i_p = B_1+B_3+B_4+B_5
                else
                  B_i_m = B_3+B_5
                  B_i_p = B_1+B_2+B_4
	        end if
	      end if
	    else
              if (eigen_value(4) .ge. 0.0) then
	        B_i_p = B_2+B_4
                B_i_m = B_1+B_3+B_5
	      else
	        if (eigen_value(2) .ge. 0.0) then
                  B_i_p = B_2
                  B_i_m = B_1+B_3+B_4+B_5
	        else
		  B_i_m = B_1+B_2+B_3+B_4+B_5
	        end if
	      end if
	    end if
	  end if
	end if

!=====================================================================
!	check whether calculation of B+ and B- was correct
!=====================================================================
        call visco_check_B_i(B_i,B_i_p,B_i_m,info)

	end subroutine visco_calc_B_m_p
	end module visco_kernel
