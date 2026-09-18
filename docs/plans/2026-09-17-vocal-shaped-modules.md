# Vocal-shaped modules — 2026-09-17

**What Kennith gets:** three new switches on the DeBleed panel, each off by default so
every existing session loads exactly as it does today. LOOKAHEAD lets the expander see
1 ms ahead so the first consonant of a phrase is never clipped (the only one that adds
latency, and he ruled it must be switchable). CONSONANT keeps the top end open during
s/sh/f/t/k/breath instead of letting the wideband expander dull them. COMB, with a DEPTH
knob (6–12 dB), thins the bleed between the vocal's own harmonics while a note is held
and backs off by itself whenever the pitch is unclear. Nothing derives its shape from the
noise; everything derives it from the vocal. Brief: brain
`projects/debleed/procedures/vocal-shaped-modules-direction.md`.

His words (2026-09-17): *"Let's do #1. And the lookahead needs to be toggle-able. I don't
want it always there."*

## Settled decisions (technical, mine)

### Parameters (APVTS, all version 1)
| id | name | type | default |
|---|---|---|---|
| `lookahead` | Lookahead | bool | off |
| `consonant` | Consonant | bool | off |
| `comb` | Comb | bool | off |
| `combDepth` | Comb Depth | float 6–12 dB, step 0.1, text "9.0 dB" | 9.0 |

### Signal chain (processBlock)
```
input ──► mono sidechain (undelayed) ──► SpectralVAD ──► vadConf[n]
                       │                     └──► expander gain law ──► mainGain[n]
                       ├─ CONSONANT: LR4 HP 4k on sidechain ──► hfExpander ──► hfGain[n]
                       │             unvoiced[n] from VAD band ratio
                       │             hiGain[n] = mainGain + unvoiced·max(0, hfGain − mainGain)
                       └─ COMB: PitchTracker (decimated MPM) ──► (period, confidence) per update
audio ──► [LOOKAHEAD: per-channel delay of round(fs/1000) samples] ──►
       if (consonant || comb):  LR4 split @ 4 kHz
              low  · mainGain ──► [COMB: fractional feedback comb, g(n)] ─┐
              high · (consonant ? hiGain : mainGain) ───────────────────────┴─► sum
       else:  audio · mainGain                                   (bit-identical to today)
──► output gain ──► mix ──► meter
```
- **Expander gain is computed ONCE from the mono sidechain and applied to every channel.**
  Today `SimpleExpander::processBlock` is called per channel with one shared envelope,
  so stereo runs the second channel off the first channel's end state. Mono is unchanged.
  `SimpleExpander` gets `float computeGain(float sidechainSample, float vad)` (the current
  law, returning the linear gain) and `processBlock(float* audio, const float* gains, n)`
  becomes a multiply; keep `processSample` for compatibility.
- **Lookahead** = `juce::dsp::DelayLine`-free ring buffer per channel, length
  `round(sampleRate * 0.001)` (96 @ 96k, 48 @ 48k, 44 @ 44.1k). Detectors read the
  undelayed input, so their gains lead the audio by 1 ms. `setLatencySamples(on ? L : 0)`
  from `parameterChanged` and from `prepareToPlay`. Toggling mid-stream may click; it is
  a setup switch.
- **One crossover for both modules:** Linkwitz-Riley 4th order (two cascaded Butterworth
  2nd-order biquads per band) at 4 kHz, per channel, on the delayed audio. Sums flat when
  nothing acts on the bands. Only instantiated in the path when consonant or comb is on.
- **Consonant = the highs judge themselves.** A second `SimpleExpander` (VAD gating off,
  same threshold/ratio/range as the main, attack 0.5 ms, release 40 ms) is fed the
  high band of the sidechain. `unvoiced` = smoothed (1 ms up / 15 ms down) ramp of
  hf/(hf+lf) from SpectralVAD's existing band envelopes (hf = bands 5–7 = 4k/6k/8k,
  lf = bands 0–3 = 200–1600 Hz), ramp 0.55→0.80 maps 0→1. During an 's' the high band
  is loud on its own so hfGain is open; cymbal bleed with no vocal is quiet in the high
  band so hfGain stays closed; a vowel has unvoiced ≈ 0 so nothing changes. SpectralVAD
  exposes `getUnvoicedRatio()` computed in `processSample`.
- **Pitch tracker** (`PitchTracker`, new, JUCE-free): decimate the sidechain by
  `D = max(1, round(fs / 24000))` after a 4th-order Butterworth LP at 8 kHz; 1024-sample
  window at the decimated rate; McLeod NSDF over lags for 80–1000 Hz
  (`nsdf[τ] = 2·Σx[i]x[i+τ] / Σ(x[i]²+x[i+τ]²)`); pick the first local max ≥ 0.9·global
  max; parabolic interpolation; confidence = that peak value; recompute every 256
  decimated samples. Output `getPeriodSamples()` at the FULL rate and `getConfidence()`.
- **Comb** (`HarmonicComb`, new, JUCE-free, one per channel): normalised feedback comb
  `y[n] = (1−g)·x[n] + g·y[n−T]` with linear-interpolated fractional T. Harmonic gain is
  exactly 1; between harmonics `(1−g)/(1+g)`. Depth D dB → `g = (1−r)/(1+r)`,
  `r = 10^(−D/20)` (6 dB → 0.333, 12 dB → 0.6). Per-sample `g = gDepth · env`; `env`
  rises with a 25 ms one-pole while `confidence ≥ 0.85 && vad ≥ 0.5 && 80 ≤ f0 ≤ 1000`,
  falls with a 5 ms one-pole otherwise. T slews to the tracker's period with a 5 ms
  one-pole only while confident (holds otherwise; g is fading to 0 anyway). Buffer sized
  `fs/60 + 4`. Failure mode is therefore "cleans less", never "robotic". Depth 6–12 as
  the brief said; the switch is the off.
- **Files:** `plugin/Source/PitchTracker.{h,cpp}`, `HarmonicComb.{h,cpp}`,
  `LinkwitzRiley4.{h,cpp}` (JUCE-free so a CLI probe can compile them), edits to
  `SimpleExpander`, `SpectralVAD`, `PluginProcessor`, `ControlPanel`,
  `DeBleedLookAndFeel`, and the `.jucer` (new files added, then Projucer `--resave`).
- **Probe, not a test:** `tools/probe/comb_probe.cpp`, built with plain clang++ against
  the three JUCE-free files. It synthesises a 220 Hz harmonic tone plus white noise,
  runs tracker + comb at 96k, and prints: tracked f0, confidence, gain at harmonics and
  at the midpoints, and the LR4 sum error. Numbers are read and checked by me; it is
  not a committed unit test suite.

### Visual (his; decided on assumption while he is away — flagged on the rail)
- A third row on the control panel: `LOOKAHEAD` `CONSONANT` `COMB` capsule switches and a
  `DEPTH` knob, same 14 px label-above format as the knobs. Panel 200 → 280 px, window
  440×460 → 440×540. Nothing above it moves.
- Switch = capsule, green `#39FF14` on / `#3A3A3A` off, white thumb. Drawn by the
  LookAndFeel when the button carries the `switchStyle` property (checked before the
  existing width>40 branch). DEPTH knob arc is the same green: one family, one colour.
- DEPTH stays enabled when COMB is off (constant geometry; nothing greys or moves).

## Tasks
1. Parameters + processor atomics + latency reporting (`lookahead` wired to a per-channel
   delay, `setLatencySamples`). Observable: plugin builds, Standalone reports 96-sample
   latency at 96k when the switch is on, 0 when off.
2. Expander refactor to gain-once-apply-everywhere. Observable: mono output identical
   to before (probe: same gains for same input).
3. `LinkwitzRiley4` + consonant module. Observable: probe shows LR4 sum flat within
   0.05 dB; a 6 kHz noise burst at −25 dB with the module on keeps the high band open
   while the wideband gain is closed.
4. `PitchTracker` + `HarmonicComb` + comb wiring + DEPTH. Observable: probe tracks
   220 Hz within 0.5 Hz, confidence > 0.9, harmonic gain 0 dB ± 0.2, midpoint gain
   −D ± 0.5 dB, and g → 0 within 10 ms of the tone stopping.
5. UI row 3 + LookAndFeel capsule switch. Observable: Standalone shows the row.
6. Build Debug (AU/VST3/Standalone), cold review by a fresh Fable subagent (Astra
   builds), fix, commit, push, rail rows.
