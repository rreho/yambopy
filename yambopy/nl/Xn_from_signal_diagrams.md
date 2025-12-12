# `Xn_from_signal` – Mermaid Diagrams

This document contains Mermaid diagrams derived from `nl_analysis.py` for the abstract class **`Xn_from_signal`** and its key workflows.

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
