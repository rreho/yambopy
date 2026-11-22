# Anti-resonant term of the one-phonon Raman tensor

Notation of `excph_raman.py`. `G_m[a, b] = exc_ph_mat_el[m, a, b] = <b|dV_m|a>`, with a = in and b = out.
`D_S = exc_dipoles[:, S] = <0|r|S>` is the emission dipole. ω is the laser energy, ω_s = ω − Ω_m, and every
exciton energy carries −iη.

## Phonon vertex

The displacement is u = Σ_qν √(1/2MΩ) [e_qν e^{iq·R} b_qν + e*_qν e^{−iq·R} b†_qν]. With
dV_qν = Σ e_qν e^{iq·R} ∂V/∂u (the operator inside the LetzElPhC/yambopy G), this gives

    H_ep = Σ_qν [ dV_qν b_qν + dV_qν^† b†_qν ].

Stokes scattering at q = 0 emits the phonon through dV_m^†. For any pair of excitons A → B:

    <B, 1_m| H_ep |A> = <B|dV_m^†|A> = <A|dV_m|B>^* = conj(G_m[B, A]).

## The two time orderings (phonon vertex in the middle)

Initial state |0; ω>, final state |0; ω_s, 1_m>. Third-order amplitude: Σ V V V / ((E_i − E_1)(E_i − E_2)).

1. **Resonant.** Absorb ω (0 → S), emit the phonon (S → S'), emit ω_s (S' → 0).
   The denominators are (ω − E_S)(ω − E_S' − Ω).

       R_res = Σ conj(D_iS) · conj(G_m[S', S]) · D_jS' / ((ω − E_S)(ω − E_S' − Ω))

   With A = S and B = S' the vertex is conj(G_m[S', S]). This is what the code has.

2. **Anti-resonant.** Emit ω_s first (0 → A + γ_s, intermediate energy ω + ω_s + E_A, denominator −(ω − Ω + E_A)).
   Then emit the phonon (A → B, denominator −(ω + E_B)). Then absorb ω (B → 0).
   The code labels S = B (factor 1/(ω + E_S)) and S' = A (factor 1/(ω + E_S' − Ω)), so

       R_ares = Σ D_iS · conj(G_m[S, S']) · conj(D_jS') / ((ω + E_S)(ω + E_S' − Ω))
              = Σ D_iS · G^{m*}_{S'S} · conj(D_jS') / (...)        with G^m_{S'S} = <S'|dV_m|S>.

The port (and PhdScripts) uses `G_m[S', S] = <S|dV_m|S'>` in the anti-resonant term. That equals
conj(G_m[S, S']) only if dV_m = dV_m^†, which holds when the Γ eigenvector e_m is real.
A complex e_m breaks the equality. Examples: a global phase e^{iφ} (which is arbitrary), or circular
combinations of degenerate E′/E″ modes.

Under e_m → e^{iφ} e_m:
- the resonant term picks up e^{−iφ};
- the anti-resonant term of the port picks up e^{+iφ}, so |R|² depends on φ, which is unphysical;
- the alternative, conj(G_m[S, S']), picks up e^{−iφ}, so |R|² is gauge invariant.

## Numerical comparison (MoTe2 1L, Γ, 9641 excitons, broadening 0.1 eV, laser 0.9–2.7 eV)

Script: `/scratch/users/rreho/paper_runs_v2/zpl_raman/raman_ar.py`; log `raman_ar.log`. Results are in the report.
`excph_raman.py` is unchanged.

| check | value |
|---|---|
| the replica of the port in the script vs `exc_ph_raman` (A1′) | 4e-8 |
| hermiticity defect ‖G−G^H‖/‖G‖ of the stored modes | A1′ 9e-7, E′ 1e-5 (the LetzElPhC Γ eigenvectors are real) |
| stored A1′, E′a, E′b: max \|I_cur − I_alt\| / max I (xx, xy, σ+σ+, σ+σ−) | ≤ 5e-6 (identical) |
| E′± = (E′a ± i E′b)/√2: xx, xy, σ+σ+, σ+σ− | ≤ 1.3e-3; ratio \|AR\|/\|R\| 0.24 (port) vs 0.20 (alt) |
| E′+ in the forbidden σ+σ− channel (TCAM forbidden: j_in + j_ph = 1 + 1 = −1 = j_out does hold for σ+ → σ−, so this channel is σ+σ− with e_out = σ−) | port 2.2e-5 at 0.9 eV; alt integral 4 % of the port's |
| A1′ × e^{0.7i} (a pure gauge change) | port changes by 2.2e-3 at the peak (\|AR\|/\|R\| 0.33 → 0.56); alt is invariant |

With the present real Γ eigenvectors the two forms agree to round-off. The port becomes gauge dependent once
complex (e.g. circular) phonon modes are used. The effect is small near resonance and largest off resonance,
where the anti-resonant term matters.
