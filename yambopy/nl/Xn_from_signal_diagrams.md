# `Xn_from_signal`: documentation (version 1.0)

#### Myrta Grüning, Claudio Attaccalite, Mike Pointeck, Anna Romani, Mao Yuncheng

This document describes the `Xn_from_signal` abstract python class and a set of derived classes, part of the `YamboPy` code, for extracting nonlinear susceptibilities and conductivities from the macroscopic time-dependent polarization $P$ and current $J$.  An extended theoretical background  can be found in Nonlinear Optics textbooks, see e.g. Sec. 2 of “The Elements of Nonlinear Optics” by Butcher and Cotter and the other sources listed in the [Bibliography](## Bibliography). A [minimal background](## 0. Theory ) is given in the next session to help understanding the code and facilitate further development. The rest of the code is dedicated to describe the code structure, key workflows, main functions and to provide an essential guide of the code use.  

## 0. Theory 

The problem solved is algebraic:

$$ M_{kj} S_j = P_k,$$

where $P_k$ is the time-dependent polarization (or current) sampled on $N_t$ times $\{t_k\}$ which is output by the `yambo`/`lumen`code; the resulting $S_j$ is proportional to the susceptibility (conductivity) of nonlinear order $j$. The matrix of coefficients $M_{kj}$, of dimensions $N_t \times N_\text{nl}$ contains the time dependence to the applied electric field. So far three physical situations are implemented:
1. a single monochromatic electric field: $ {\bf E}_0 \sin(\omega_0 t)$
2. two monochromatic electric fields: $ {\bf E}_0 (\sin(\omega_1 t) + \sin(\omega_2 t)) $
3. a pulse-shaped electric field: $ {\bf E}(t) \sin(\omega_0 t)$. Here, it is assumed that the shape of the pulse ${\bf E}(t)$ varies slowly with respect to the period $2\pi/\omega_0$. So far, only a Gaussian pulse (${\bf E}(t) = {\bf E}_0 \exp(-(t-t_0)^2/(2\sigma^2))/(\sqrt{2}\sigma)$) has been implemented. 

Four solvers are available:

1. the standard solver for full, well-determined matrix:  calls [`numpy.linalg.solve`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.solve.html)
2. the least square solver, when $N_t \gg N_\text{nl}$ : calls  [`numpy.linalg.lstsq`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html#numpy.linalg.lstsq)
3. the single value decomposition, using the Moore-Penrose pseudoinverse,  when $N_t \gg N_\text{nl}$: calls [`numpy.linalg.pinv`](https://numpy.org/doc/stable/reference/generated/numpy.linalg.pinv.html#numpy.linalg.pinv)
4. the least square solver with an initial guess, when $N_t \gg N_\text{nl}$ : calls  [`scipy.optimize.least_squares`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)

From $S_j$ the susceptibilities (or conductivities) $\chi^{(n)}(-\omega_\sigma, \omega_1, \dots, \omega_n)$  are obtained using the following expression:

$$ S_j = C_0 K (-\omega_\sigma, \omega_1, \dots, \omega_n)\chi^{(n)}(-\omega_\sigma, \omega_1, \dots, \omega_n) $$

where $K(-\omega_\sigma; \omega_1, \dots, \omega_n)$ is a numerical factor that accounts for the intrinsic permutation symmetry depending on the nonlinear order and frequency arguments of $\chi$. $C_0$ is a further numerical factor depending on the applied electric field.

Details on the implementation can be found in the sources listed in the [Bibliography](## Bibliography)

---

## 1) Class structure (`classDiagram`)
```mermaid
classDiagram
    class Xn_from_signal {
        <<abstract>>
        %% --- Attributes ---
        +time : np.ndarray
        +T_step : float
        +T_deph : float
        +efield : dict
        +pumps : list
        +efields : list
        +n_runs : int
        +polarization : np.ndarray
        +current : np.ndarray
        +l_eval_current : bool
        +l_out_current : bool
        +X_order : int
        +solver : str
        +samp_mod : str
        +nsamp : int
        +T_urange : list[float]
        +freqs : np.ndarray
        +prefix : str
        +out_dim : int
        +tol : float

        %% --- Constructor ---
        +__init__(nldb, X_order=4, T_range=[-1,-1], l_out_current=False, nsamp=-1, samp_mod='', solver='', tol=1e-10)

        %% --- Special ---
        +__str__() str

        %% --- Abstract hooks ---
        #set_defaults()
        #get_sampling(idir, ifrq)
        #define_matrix(samp, ifrq)
        #update_time_range()
        #get_Unit_of_Measure(i_order)   
        #output_analysis(out, to_file)
        #reconstruct_signal(out, to_file)

        %% --- Concrete methods ---
        +solve_lin_system(mat, samp, init=None) np.ndarray
        +perform_analysis() np.ndarray
        +get_Unit_of_Measure(i_order) float
    }

    %% Auxiliary (module-level) functions for completeness
    class AuxMath {
        +IsSquare(m) bool
        +IsWellConditioned(m) bool
        +residuals_func(x, M, S_i) float
    }
```

---

## 2) Workflow: `perform_analysis` (`flowchart`)
```mermaid
graph TD
  A[perform_analysis] --> B[set_defaults]
  B --> C[allocate_out]
  C --> D{loop_i_f}
  D --> E{loop_i_d}
  E --> F[get_sampling]
  F --> G[define_matrix]
  G --> H[solve_lin_system]
  H --> I[assign_out]
  I --> J{more_i_d}
  J -- Yes --> E
  J -- No --> K{more_i_f}
  K -- Yes --> D
  K -- No --> L[return_out]
```

---

## 3) Workflow: `solve_lin_system` (`flowchart`)
```mermaid
graph TD
  A[solve_lin_system] --> B[init_out]
  B --> C{solver_full}
  C -- Yes --> D{square_and_well_conditioned}
  D -- No --> E[set_solver_lstsq]
  D -- Yes --> F[linalg_solve]
  C -- No --> G{solver_lstsq}
  G -- Yes --> H[lstsq]
  G -- No --> I{solver_svd}
  I -- Yes --> J[pinv]
  J --> K[accumulate_inv_times_samp]
  I -- No --> L{solver_lstsq_opt}
  L -- Yes --> M{has_init}
  M -- Yes --> O[use_init]
  M -- No --> N[lstsq_for_x0]
  N --> P[concat_real_and_imag]
  O --> P
  P --> Q[least_squares]
  Q --> R[compose_complex_out]
  L -- No --> S[return_out]
  F --> T[return_out]
  H --> T
  K --> T
  R --> T
  E --> H
```

---

## 4) Sequence view (optional)
```mermaid
sequenceDiagram
    participant Analyzer as Xn_from_signal
    participant Impl as Subclass (implements abstract hooks)

    Analyzer->>Impl: set_defaults()
    loop for each run i_f and direction i_d
        Analyzer->>Impl: get_sampling(i_d, i_f)
        Impl-->>Analyzer: (samp_time, samp_sig)
        Analyzer->>Impl: define_matrix(samp_time, i_f)
        Impl-->>Analyzer: matrix
        Analyzer->>Analyzer: solve_lin_system(matrix, samp_sig)
        Analyzer->>Analyzer: out[:, i_f, i_d] = raw[:out_dim]
    end
    Analyzer-->>Impl: output_analysis(out, to_file)
    Analyzer-->>Impl: reconstruct_signal(out, to_file)
```

## Bibliography
