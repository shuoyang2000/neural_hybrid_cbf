param1 = {
        # vehicle body dimensions
        'length': 4.298,  # vehicle length [m]
        'width': 1.674,  # vehicle width [m]

        # steering constraints
        's_min': -0.910,  # minimum steering angle [rad]
        's_max': 0.910,  # maximum steering angle [rad]
        'sv_min': -0.4,  # minimum steering velocity [rad/s]
        'sv_max': 0.4,  # maximum steering velocity [rad/s]

        # longitudinal constraints
        'v_min': -13.9,  # minimum velocity [m/s]
        'v_max': 45.8,  # minimum velocity [m/s]
        'v_switch': 4.755,  # switching velocity [m/s]
        'a_max': 3.5,  # maximum absolute acceleration [m/s^2]

        # masses
        'm': 1225.887,  # vehicle mass [kg]  MASS
        'm_s': 1094.542,  # sprung mass [kg]  SMASS
        'm_uf': 65.672,  # unsprung mass front [kg]  UMASSF
        'm_ur': 65.672,  # unsprung mass rear [kg]  UMASSR

        # axes distances
        'lf': 0.88392,  # distance from spring mass center of gravity to front axle [m]  LENA
        'lr': 1.50876,  # distance from spring mass center of gravity to rear axle [m]  LENB

        # moments of inertia of sprung mass
        'I_Phi_s': 244.0472306,  # moment of inertia for sprung mass in roll [kg m^2]  IXS
        'I_y_s': 1342.259768,  # moment of inertia for sprung mass in pitch [kg m^2]  IYS
        'I_z': 1538.853371,  # moment of inertia for sprung mass in yaw [kg m^2]  IZZ
        'I_xz_s': 0.0,  # moment of inertia cross product [kg m^2]  IXZ

        # suspension parameters
        'K_sf': 21898.332429,  # suspension spring rate (front) [N/m]  KSF
        'K_sdf': 1459.390293,  # suspension damping rate (front) [N s/m]  KSDF
        'K_sr': 21898.332429,  # suspension spring rate (rear) [N/m]  KSR
        'K_sdr': 1459.390293,  # suspension damping rate (rear) [N s/m]  KSDR

        # geometric parameters
        'T_f': 1.389888,  # track width front [m]  TRWF
        'T_r': 1.423416,  # track width rear [m]  TRWB
        'K_ras': 175186.659437,  # lateral spring rate at compliant compliant pin joint between M_s and M_u [N/m]  KRAS

        'K_tsf': -12880.270509,  # auxiliary torsion roll stiffness per axle (normally negative) (front) [N m/rad]  KTSF
        'K_tsr': 0.0,  # auxiliary torsion roll stiffness per axle (normally negative) (rear) [N m/rad]  KTSR
        'K_rad': 10215.732056,  # damping rate at compliant compliant pin joint between M_s and M_u [N s/m]  KRADP
        'K_zt': 189785.547723,  # vertical spring rate of tire [N/m]  TSPRINGR

        'h_cg': 0.557784,  # center of gravity height of total mass [m]  HCG (mainly required for conversion to other vehicle models)
        'h_raf': 0.0,  # height of roll axis above ground (front) [m]  HRAF
        'h_rar': 0.0,  # height of roll axis above ground (rear) [m]  HRAR

        'h_s': 0.59436,  # M_s center of gravity above ground [m]  HS

        'I_uf': 32.539630,  # moment of inertia for unsprung mass about x-axis (front) [kg m^2]  IXUF
        'I_ur': 32.539630,  # moment of inertia for unsprung mass about x-axis (rear) [kg m^2]  IXUR
        'I_y_w': 1.7,  # wheel inertia, from internet forum for 235/65 R 17 [kg m^2]

        'K_lt': 1.0278264878518764e-05,  # lateral compliance rate of tire, wheel, and suspension, per tire [m/N]  KLT
        'R_w': 0.344,  # effective wheel/tire radius  chosen as tire rolling radius RR  taken from ADAMS documentation [m]

        # split of brake and engine torque
        'T_sb': 0.76,
        'T_se': 1,

        # suspension parameters
        'D_f': -0.623359580, # [rad/m]  DF
        'D_r': -0.209973753,  # [rad/m]  DR
        'E_f': 0,  # [needs conversion if nonzero]  EF
        'E_r': 0,  # [needs conversion if nonzero]  ER

        # tire parameters from ADAMS handbook
        # longitudinal coefficients
        'tire_p_cx1': 1.6411,  # Shape factor Cfx for longitudinal force
        'tire_p_dx1': 1.1739,  # Longitudinal friction Mux at Fznom
        'tire_p_dx3': 0,  # Variation of friction Mux with camber
        'tire_p_ex1': 0.46403,  # Longitudinal curvature Efx at Fznom
        'tire_p_kx1': 22.303,  # Longitudinal slip stiffness Kfx/Fz at Fznom
        'tire_p_hx1': 0.0012297,  # Horizontal shift Shx at Fznom
        'tire_p_vx1': -8.8098e-006,  # Vertical shift Svx/Fz at Fznom
        'tire_r_bx1': 13.276,  # Slope factor for combined slip Fx reduction
        'tire_r_bx2': -13.778,  # Variation of slope Fx reduction with kappa
        'tire_r_cx1': 1.2568,  # Shape factor for combined slip Fx reduction
        'tire_r_ex1': 0.65225,  # Curvature factor of combined Fx
        'tire_r_hx1': 0.0050722,  # Shift factor for combined slip Fx reduction

        # lateral coefficients
        'tire_p_cy1': 1.3507,  # Shape factor Cfy for lateral forces
        'tire_p_dy1': 1.0489,  # Lateral friction Muy
        'tire_p_dy3': -2.8821,  # Variation of friction Muy with squared camber
        'tire_p_ey1': -0.0074722,  # Lateral curvature Efy at Fznom
        'tire_p_ky1': -21.92,  # Maximum value of stiffness Kfy/Fznom
        'tire_p_hy1': 0.0026747,  # Horizontal shift Shy at Fznom
        'tire_p_hy3': 0.031415,  # Variation of shift Shy with camber
        'tire_p_vy1': 0.037318,  # Vertical shift in Svy/Fz at Fznom
        'tire_p_vy3': -0.32931,  # Variation of shift Svy/Fz with camber
        'tire_r_by1': 7.1433,  # Slope factor for combined Fy reduction
        'tire_r_by2': 9.1916,  # Variation of slope Fy reduction with alpha
        'tire_r_by3': -0.027856,  # Shift term for alpha in slope Fy reduction
        'tire_r_cy1': 1.0719,  # Shape factor for combined Fy reduction
        'tire_r_ey1': -0.27572,  # Curvature factor of combined Fy
        'tire_r_hy1': 5.7448e-006,  # Shift factor for combined Fy reduction
        'tire_r_vy1': -0.027825,  # Kappa induced side force Svyk/Muy*Fz at Fznom
        'tire_r_vy3': -0.27568,  # Variation of Svyk/Muy*Fz with camber
        'tire_r_vy4': 12.12,  # Variation of Svyk/Muy*Fz with alpha
        'tire_r_vy5': 1.9,  # Variation of Svyk/Muy*Fz with kappa
        'tire_r_vy6': -10.704,  # Variation of Svyk/Muy*Fz with atan(kappa)
    }