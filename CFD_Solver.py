"""Programmed by Binay Limbu
University of Southampton
1-d CFD solver for airflow through a convergent-divergent nozzle"""

import math
import numpy
import pylab

def nozzle_area(x):
    """Function to calculate A/A* of the nozzle"""
    return 1 + 2.2*(x-1.5)**2

def initial_variable(x):
    """Function to calculate the initial primitive variables"""
    if x>=0 and x<=0.5:
        rho_p = 1.0
        temp_p = 1.0
    elif x>0.5 and x<=1.5:
        rho_p = 1.0 - 0.366*(x-0.5)
        temp_p = 1.0-0.167*(x-0.5)
    elif x>1.5 and x<=3.5:
        rho_p = 0.634 - 0.3879*(x-1.5)
        temp_p = 0.833-0.3507*(x-1.5)
    return rho_p,temp_p

def initial_shock_variables(x):
    """Function to calculate the initial primitive variables when there is a shock in nozzle"""
    if x>=0 and x<=0.5:
        rho_p = 1.0
        temp_p = 1.0
    elif x>0.5 and x<=1.5:
        rho_p = 1.0 - 0.366*(x-0.5)
        temp_p = 1.0-0.167*(x-0.5)
    elif x>1.5 and x<=2.1:
        rho_p = 0.634 - 0.702*(x-1.5)
        temp_p = 0.833-0.4908*(x-1.5)
    elif x>2.1 and x<=3.0:
        rho_p = 0.5982 + 0.10228*(x-2.1)
        temp_p = 0.93968 + 0.0622*(x-2.1)
    return rho_p,temp_p



def solver(grid_points,time_step,case):
    """Execute this function to perform calculation
    grid points ~ no of grid points in the axis
    time step ~ No of iterations or time step
    case 1 ~ subsonic subsonic flow with a shock in the divergent section LAX method also included
    output -1. initial values of rho, temp, vel , U1, U2, U3.
            2. graph of (rho/rho_o vs grid point), graph of (P/P_o vs grid points), graph of Mach Number vs grid points and Mach number
            convergence histroy vs time steps for Mac Cormacks method
            3. graph of Mach number vs grid points and Mach convergence vs time steps for LAX method
            4. Final values of primitive variables and U1 U2 AND U3 in table form.
    case 0 ~ subsonic supersonic flow
    output - 1. initial values of rho, temp, vel , U1, U2, U3.
             2. Graph of (rho/rho_o) vs grid points
             3. Final values of primitive variables and U1 U2 AND U3 in table form."""

    C_x = 0.6
    back_pressure = 0.6784
    grid_size = grid_points
    courant = 0.5
    gamma = 1.4
    area = []
    grid = []
    rho = []
    temp = []
    vel = []
    pres = []
    Mach = []
    LAX_rho = []
    LAX_vel = []
    LAX_temp = []
    LAX_Mach = []
    LAX_pres = []
    U1 = []
    U2 = []
    U3 = []
    LAX_U1 = []
    LAX_U2 = []
    LAX_U3 = []
    F1 = []
    F2 = []
    F3 = []
    LAX_F1 = []
    LAX_F2 = []
    LAX_F3 = []
    J2 = []
    LAX_J2 = []
    S_1 = []
    S_2 = []
    S_3 = []
    dU1_dt = []
    dU2_dt = []
    dU3_dt = []
    pred_U1 = []
    pred_U2 = []
    pred_U3 = []
    pred_rho = []
    pred_temp = []
    pred_vel = []
    pred_pres = []
    pred_F1 = []
    pred_F2 = []
    pred_F3 = []
    p_S_1 = []
    p_S_2 = []
    p_S_3 = []
    dU1_dt_bar = []
    dU2_dt_bar = []
    dU3_dt_bar = []
    avg_dU1_dt = []
    avg_dU2_dt = []
    avg_dU3_dt = []
    U1_new = []
    U2_new = []
    U3_new = []
    LAX_plust_U1 = []
    LAX_plust_U2 = []
    LAX_plust_U3 = []
    convergence_history = []
    LAX_convergence_history = []
    steps = []
    #Working out the initial conditions
    x = 0.0
    delta_x = (3.0/(grid_size - 1))
    for i in range(grid_size):
        """For loop to find the initial conditions of the primitive variables and U1 U2 U3"""
        a = nozzle_area(x)
        if case == 0: #subsonic_supersonic
            r, t = initial_variable(x)
        elif case == 1:#subsonic_subsonic with a shock
            r, t = initial_shock_variables(x)
        area.append(a)
        grid.append(x)
        x+= delta_x
        v = 0.59/(r*a)
        pressure = r*t
        m = v/math.sqrt(t)
        u_1 = (r*a)
        u_2 = (r*a*v)
        u_3 = u_1*((t/(gamma - 1))+(gamma*0.5*(v**2)))
        U1.append(u_1)
        U2.append(u_2)
        U3.append(u_3)
        rho.append(r)
        temp.append(t)
        vel.append(v)
        pres.append(pressure)
        Mach.append(m)
        if case == 1:
            #for subsonic subsonic case using the LAX method
            LAX_U1.append(u_1)
            LAX_U2.append(u_2)
            LAX_U3.append(u_3)
            LAX_rho.append(r)
            LAX_temp.append(t)
            LAX_vel.append(v)
            LAX_pres.append(pressure)
            LAX_Mach.append(m)
    """Starting the time steps"""
    for t in range(time_step):
        steps.append(t)
        outflow = Mach[grid_size -1]
        convergence_history.append(outflow) #calculating the convergence history for q4
        lowest = 10
        for j in range(grid_size):
            """For loop to find the minimum delta t"""
            delta_t = courant*(delta_x/(math.sqrt(temp[j])+ vel[j]))
            if delta_t<lowest:
                lowest = delta_t

        """Working out the initial value of the fluxes i.e. F1, F2 and F3 from the initial
        values of U1, U2, U3"""

        for j in range(grid_size):
            f_1 = U2[j]
            const_2 = ((U2[j]**2)/U1[j])
            f_2 = const_2 +((0.4/gamma)*(U3[j]-(0.7*const_2)))
            f_3 = (1.4*U2[j]*U3[j]/U1[j]) - ((0.7*0.4)*(U2[j]**3)/(U1[j]**2))
            if j==(grid_size - 1):
                j_2 = 0
            else:
                delta_a = area[j+1] - area[j]
                j_2 = rho[j]*temp[j]*(delta_a/delta_x)*(1/1.4)
            if t==0:
                F3.append(f_3)
                F1.append(f_1)
                F2.append(f_2)
                J2.append(j_2)
            else:
                F1[j]=f_1
                F2[j]=f_2
                F3[j]= f_3
                J2[j]=j_2

        """Carrying out the forward differencing"""
        for j in range(grid_size):
            # calculation of j ignored at last grid point
            if j == (grid_size - 1):
                d_U1 = 0
                d_U2 = 0
                d_U3 = 0
            else:
                d_U1 = - (F1[j+1] - F1[j])/delta_x
                d_U2 = - ((F2[j+1] - F2[j])/delta_x )+ J2[j]
                d_U3 = - (F3[j+1] - F3[j])/delta_x
            if t==0:
                dU1_dt.append(d_U1)
                dU2_dt.append(d_U2)
                dU3_dt.append(d_U3)
            else:
                dU1_dt[j] = d_U1
                dU2_dt[j] = d_U2
                dU3_dt[j] = d_U3

        """Only does this if user has specified subsonic subsonic flow"""
        #calculating artifical viscosity
        if case == 1:
            for i in range(grid_size):
                if j == 0 or j == (grid_size - 1):
                    s_1 = 0
                    s_2 = 0
                    s_3 = 0
                else:
                    switch = C_x*abs(pres[j+1] - 2*pres[j] + pres[j-1])/(pres[j+1] + 2*pres[j] + pres[j-1])
                    s_1 = switch *(U1[j+1] - 2*U1[j] + U1[j-1])
                    s_2 = switch *(U2[j+1] - 2*U2[j] + U2[j-1])
                    s_3 = switch *(U3[j+1] - 2*U3[j] + U3[j-1])
                if t==0:
                    S_1.append(s_1)
                    S_2.append(s_2)
                    S_3.append(s_3)
                else:
                    S_1[j] = s_1
                    S_2[j] = s_2
                    S_3[j] = s_3

        """Calculating the predicted values"""
        for j in range(grid_size):
            if case == 1: #if subsonic_subsonic include the pressure_switch term
                p_U1 = U1[j] + (dU1_dt[j]*lowest) + S_1[j]
                p_U2 = U2[j] + (dU2_dt[j]*lowest) + S_2[j]
                p_U3 = U3[j] + (dU3_dt[j]*lowest) + S_3[j]
            elif case==0:
                p_U1 = U1[j] + (dU1_dt[j]*lowest)
                p_U2 = U2[j] + (dU2_dt[j]*lowest)
                p_U3 = U3[j] + (dU3_dt[j]*lowest)
            if t==0:
                pred_U1.append(p_U1)
                pred_U2.append(p_U2)
                pred_U3.append(p_U3)
            else:
                pred_U1[j] = p_U1
                pred_U2[j] = p_U2
                pred_U3[j] = p_U3

        """finding the values of the predicted primitive variables i.e rho(t+delta), temp(t+delta)"""
        for j in range(grid_size):
            p_rho = pred_U1[j]/area[j]
            p_Temp = (0.4)*((pred_U3[j]/pred_U1[j]) - (0.7*(pred_U2[j]/pred_U1[j])**2))
            p_Vel = pred_U2[j]/pred_U1[j]
            p_pres = p_rho*p_Temp
            if t == 0:
                pred_rho.append(p_rho)
                pred_temp.append(p_Temp)
                pred_vel.append(p_Vel)
                pred_pres.append(p_pres)
            else:
                pred_rho[j] = p_rho
                pred_temp[j] = p_Temp
                pred_vel[j] = p_Vel
                pred_pres[j] = p_pres

        """Evaluating the predicted value of the fluxes"""
        for j in range(grid_size):
            p_F1 = pred_U2[j]
            const = (pred_U2[j]**2)/pred_U1[j]
            p_F2 = const + ((0.4/1.4)*(pred_U3[j] - 0.7*const))
            p_F3 = (1.4*pred_U3[j]*pred_U2[j]/pred_U1[j]) - (0.7*0.4*((pred_U2[j]**3)/pred_U1[j]**2))
            if t==0:
                pred_F1.append(p_F1)
                pred_F2.append(p_F2)
                pred_F3.append(p_F3)
            else:
                pred_F1[j] = p_F1
                pred_F2[j] = p_F2
                pred_F3[j] = p_F3


        """Carrying out the corrector step"""
        for j in range(grid_size):
            if j==0:
                p_d_U1 = 0
                p_d_U2 = 0
                p_d_U3 = 0
            else:
                p_d_U1= -(pred_F1[j]-pred_F1[j-1])/delta_x
                p_d_U2 = -((pred_F2[j] - pred_F2[j-1])/delta_x) + ((1/1.4)*pred_rho[j]*pred_temp[j]*(area[j] - area[j-1])/delta_x)
                p_d_U3 = -(pred_F3[j]-pred_F3[j-1])/delta_x
            if t == 0:
                dU1_dt_bar.append(p_d_U1)
                dU2_dt_bar.append(p_d_U2)
                dU3_dt_bar.append(p_d_U3)
            else:
                dU1_dt_bar[j] = p_d_U1
                dU2_dt_bar[j] = p_d_U2
                dU3_dt_bar[j] = p_d_U3


        """calculating the average time derivatives"""
        for j in range(grid_size):
            average_U1 = 0.5 *(dU1_dt[j] + dU1_dt_bar[j])
            average_U2 = 0.5 *(dU2_dt[j] + dU2_dt_bar[j])
            average_U3 = 0.5 *(dU3_dt[j] + dU3_dt_bar[j])
            if t == 0:
                avg_dU1_dt.append(average_U1)
                avg_dU2_dt.append(average_U2)
                avg_dU3_dt.append(average_U3)
            else:
                avg_dU1_dt[j] = average_U1
                avg_dU2_dt[j] = average_U2
                avg_dU3_dt[j] = average_U3


        """Calculating the artificial viscosity at t + delta t"""
        if case == 1:
            for j in range(grid_size):
                if j ==0 or j == (grid_size -1):
                    plust_s1 = 0
                    plust_s2 = 0
                    plust_s3 = 0
                else:
                    plust_switch = C_x * abs(pred_pres[j+1] - 2*pred_pres[j] + pred_pres[j-1])/(pred_pres[j+1] + 2*pred_pres[j] + pred_pres[j-1])
                    plust_s1 = plust_switch*(pred_U1[j+1] -2*pred_U1[j] + pred_U1[j-1])
                    plust_s2 = plust_switch*(pred_U2[j+1] -2*pred_U2[j] + pred_U2[j-1])
                    plust_s3 = plust_switch*(pred_U3[j+1] -2*pred_U3[j] + pred_U3[j-1])
                if t == 0:
                    p_S_1.append(plust_s1)
                    p_S_2.append(plust_s2)
                    p_S_3.append(plust_s3)
                else:
                    p_S_1[j] = plust_s1
                    p_S_2[j] = plust_s2
                    p_S_3[j] = plust_s3


        """Carrying out the calculation for final corrected version of U1,U2,U3 i.e. at time = delta t """
        for j in range(grid_size):
            if case ==1:
                U1_n = U1[j] + (lowest*avg_dU1_dt[j]) + p_S_1[j]
                U2_n = U2[j] + (lowest*avg_dU2_dt[j]) + p_S_2[j]
                U3_n = U3[j] + (lowest*avg_dU3_dt[j]) + p_S_3[j]
            elif case == 0:
                U1_n = U1[j] + (lowest*avg_dU1_dt[j])
                U2_n = U2[j] + (lowest*avg_dU2_dt[j])
                U3_n = U3[j] + (lowest*avg_dU3_dt[j])
            if t == 0:
                U1_new.append(U1_n)
                U2_new.append(U2_n)
                U3_new.append(U3_n)
            else:
                U1_new[j] = U1_n
                U2_new[j] = U2_n
                U3_new[j] = U3_n

        """Apply the boundary conditions now."""
        #finding the value of U1_new, U2_new and U3_new at the 0 index
        U1_new[0] = 5.950 # U1 = rho * A. rho is fixed variable = 1 and A doesn't change so U1 is constant independent of time
        U2_new[0] = 2*U2_new[1] - U2_new[2]
        v_dash = U2_new[0]/U1_new[0]
        U3_new[0] = U1_new[0] * ((1/0.4)+(0.7*(v_dash**2)))

        #finding the value of U1_new, U2_new and U3_new at the last grid point
        U1_new[grid_size-1] = 2*U1_new[grid_size-2] - U1_new[grid_size-3]
        U2_new[grid_size-1] = 2*U2_new[grid_size-2] - U2_new[grid_size-3]
        if case == 0:
            U3_new[grid_size-1] = 2*U3_new[grid_size-2] - U3_new[grid_size-3]
        elif case == 1:
            U3_new[grid_size - 1] = (back_pressure*area[grid_size-1]/0.4) + (0.7*U2_new[grid_size-1]*(U2_new[grid_size-1]/U1_new[grid_size -1]))


        #printing the initial conditions for the first instance
        if t==0:
            print "index ..grid..A/A*....rho...Vel...Temp....U1....U2....U3...F1.....F2.....F3...j2"
            for i in range(grid_size):
                d,a,o,l,p,one,two,three,flux1,flux2,flux3,jval= (grid[i], area[i], rho[i], vel[i],temp[i],U1[i],U2[i],U3[i],F1[i],F2[i],F3[i],J2[i])
                print ('%3d %6.3f  %6.3f  %6.3f  %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f %6.3f  %6.3f'%((i+1),d,a,o,l,p,one,two,three,flux1,flux2,flux3,jval))





        #substituting the old value of U1 with the new calculated value
        for j in range(grid_size):
            U1[j] = U1_new[j]
            U2[j] = U2_new[j]
            U3[j] = U3_new[j]

        for j in range(grid_size):
            """For loop to calculate the new primitive variables"""
            rho[j] = U1[j]/area[j]
            temp[j] = (0.4)*((U3[j]/U1[j]) - (0.7*(U2[j]/U1[j])**2))
            vel[j] = U2[j]/U1[j]
            pres[j] = rho[j]*temp[j]
            Mach[j] = vel[j]/math.sqrt(temp[j])

    """Plotting the required graphs"""
    density_graph(rho,grid)
    if case == 1:
        pressure_graph(pres,grid)
        mach_graph(Mach,grid)
        convergence_graph(convergence_history,steps)

    """Carrying out the calculation using the lax method for question 4"""
    if case == 1:
        for t in range(time_step):
            smallest = 100
            Lax_outflow = LAX_Mach[grid_size-1]
            LAX_convergence_history.append(Lax_outflow)
            for i in range(grid_size):
                """Finding the smalles time for the LAX method """
                LAX_delta_t = courant*(delta_x/(math.sqrt(LAX_temp[i])+ LAX_vel[i]))
                if LAX_delta_t<smallest:
                    smallest = LAX_delta_t


            for i in range(grid_size):
                """Calculating F1 F2 and F3 for the lax method"""
                f_1 = LAX_U2[i]
                const_2 = ((LAX_U2[i]**2)/LAX_U1[i])
                f_2 = const_2 +((0.4/gamma)*(LAX_U3[i]-(0.7*const_2)))
                f_3 = (1.4*LAX_U2[i]*LAX_U3[i]/LAX_U1[i]) - ((0.7*0.4)*(LAX_U2[i]**3)/(LAX_U1[i]**2))
                if i==(grid_size - 1):
                    j_2 = 0
                else:
                    delta_a = (area[i+1] - area[i-1])*0.5 #central differencing
                    j_2 = LAX_rho[i]*LAX_temp[i]*(delta_a/delta_x)*(1/1.4)
                if t==0:
                    LAX_F3.append(f_3)
                    LAX_F1.append(f_1)
                    LAX_F2.append(f_2)
                    LAX_J2.append(j_2)
                else:
                    LAX_F1[i]=f_1
                    LAX_F2[i]=f_2
                    LAX_F3[i]= f_3
                    LAX_J2[i]=j_2

            for i in range(grid_size):
                """Calculating U1 U2 and U3 at t + delta t"""
                if i == 0 or i == (grid_size-1):
                    L_plust_U1 = 0
                    L_plust_U2 = 0
                    L_plust_U3 = 0
                else:
                    L_plust_U1 = 0.5*(LAX_U1[i+1] + LAX_U1[i-1]) - ((smallest/delta_x)*(LAX_F1[i+1] - LAX_F1[i-1])*0.5)
                    L_plust_U2 = (0.5*(LAX_U1[i+1] + LAX_U1[i-1]) - ((smallest/delta_x)*(LAX_F1[i+1] - LAX_F1[i-1])*0.5))+LAX_J2[i]
                    L_plust_U3 = 0.5*(LAX_U1[i+1] + LAX_U1[i-1]) - ((smallest/delta_x)*(LAX_F1[i+1] - LAX_F1[i-1])*0.5)
                if t==0:
                    LAX_plust_U1.append(L_plust_U1)
                    LAX_plust_U2.append(L_plust_U2)
                    LAX_plust_U3.append(L_plust_U3)
                else:
                    LAX_plust_U1[i] = L_plust_U1
                    LAX_plust_U2[i] = L_plust_U2
                    LAX_plust_U3[i] = L_plust_U3

            """Applying the boundary conditions"""
            LAX_plust_U1[0] = 5.950 # U1 = rho * A. rho is fixed variable = 1 and A doesn't change so U1 is constant independent of time
            LAX_plust_U2[0] = 2*LAX_plust_U2[1] - LAX_plust_U2[2]
            v_dash = LAX_plust_U2[0]/LAX_plust_U1[0]
            LAX_plust_U3[0] = LAX_plust_U1[0] * ((1/0.4)+(0.7*(v_dash**2)))
            #finding the value of U1_new, U2_new and U3_new at the 31 index
            LAX_plust_U1[grid_size-1] = 2*LAX_plust_U1[grid_size-2] - LAX_plust_U1[grid_size-3]
            LAX_plust_U2[grid_size-1] = 2*LAX_plust_U2[grid_size-2] - LAX_plust_U2[grid_size-3]
            LAX_plust_U3[grid_size - 1] = (back_pressure*area[grid_size-1]/0.4) + (0.7*LAX_plust_U2[grid_size-1]*(LAX_plust_U2[grid_size-1]/LAX_plust_U1[grid_size -1]))

            for i in range(grid_size):
                """Updating the old solution with the new one"""
                LAX_U1[i] = LAX_plust_U1[i]
                LAX_U2[i] = LAX_plust_U1[i]
                LAX_U3[i] = LAX_plust_U1[i]
                LAX_rho[i] = U1[i]/area[i]
                LAX_temp[i] = (0.4)*((U3[i]/U1[i]) - (0.7*(U2[i]/U1[i])**2))
                LAX_vel[i] = U2[i]/U1[i]
                LAX_pres[i] = rho[i]*temp[i]
                LAX_Mach[i] = LAX_vel[i]/math.sqrt(LAX_temp[i])

        mach_graph(LAX_Mach,grid)
        convergence_graph(LAX_convergence_history,steps)


    for i in range(3):
        print ""

    for i in range(grid_size):
        """This for loop prints the value of the primitive variables along with values of Mach Number and U1 U2 and U3 obtained at the last time step"""
        d,a,o,l,p,s,h,one,two,three,= (grid[i], area[i], rho[i], vel[i],temp[i],pres[i],Mach[i],U1[i],U2[i],U3[i])
        print ('%3d %8.3f  %8.3f  %8.3f  %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f %8.3f'%((i+1),d,a,o,l,p,s,h,one,two,three))


def mach_graph(y,x):
    """This function plots the Mach number against grid"""
    pylab.plot(x,y,label = 'mach number')
    pylab.grid()
    pylab.xlabel('grid')
    pylab.ylabel('mach number')
    pylab.show()

def convergence_graph(y,x):
    """This function plots the convergence history of the Mach Number at the outflow for both methods"""
    pylab.plot(x,y,label = 'Mach number at outflow')
    pylab.grid()
    pylab.xlabel('time step')
    pylab.ylabel('mach number')
    pylab.show()

def density_graph(r,g):
    """Function to plot the graph of density vs grid"""
    pylab.plot(g,r,label = 'density vs grid')
    pylab.grid()
    pylab.xlabel('grid')
    pylab.ylabel('density')
    pylab.show()

def pressure_graph(p,g):
    """Function to plot the graph of pressure vs grid"""
    pylab.plot(g,p,label= 'pressure vs grid')
    pylab.grid()
    pylab.xlabel('grid')
    pylab.ylabel('pressure')
    pylab.show()
