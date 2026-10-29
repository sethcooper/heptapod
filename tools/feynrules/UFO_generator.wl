#!/usr/bin/env wolframscript
(*
   UFO_generator.wl is a part of the HEPTAPOD package.
   Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
   HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
   Please respect the MCnet Guidelines, see GUIDELINES for details.
*)

(* ============ FeynRules model -> UFO, with consistency checks ============ *)
(*
   Driver behind FeynRulesToUFOTool and ValidateLagrangianTool.

   Usage:
     wolframscript -f UFO_generator.wl \
        ModelPath=<PATH_TO_MODEL.fr> \
        FeynRulesPath=<PATH_TO_FeynRules> \
        OutputDir=<PATH_TO_OUTPUT> \
        Lagrangian="LSM + LBSM" \
        [LoadSM=true|false] [ChecksOnly=true|false] \
        [Checks=true|false|all|<name,name,...>] [BaselineCompare=true|false] \
        [StrictChecks=true|false] [AddDecays=true|false] \
        [FeynmanGauge=true|false] [ExpandIndices=true|false]

   Nothing about the model is assumed.  The Lagrangian expression is supplied
   by the caller rather than guessed from a naming convention, and whether the
   Standard Model is loaded alongside the model file is a separate decision
   from what that expression contains -- a pure dark sector needs neither.
   Omit Lagrangian entirely and the script loads the model, reports the
   Lagrangian-like symbols it actually defines, and exits; discovering the
   name is cheaper than guessing it wrong.

   Exit codes:
     0  success
     1  bad invocation, unusable environment, or undefined Lagrangian symbol
     2  WriteUFO reported success but produced an incomplete UFO
     3  a consistency check failed and StrictChecks=true
*)
(* ======================================================================== *)

ClearAll["Global`*"];

(* ---- Parse command-line args as key=value pairs. ---- *)
(* StringSplit is limited to two parts so a value may itself contain "=",
   which a Lagrangian expression legitimately can. *)
heptapodParams = Association @ Map[
    Rule @@ StringSplit[#, "=", 2] &,
    Rest[$ScriptCommandLine]
];

heptapodBoolParam[key_String, default_] :=
  Switch[ToLowerCase[ToString[Lookup[heptapodParams, key, default]]],
    "true", True, "false", False, _, default];

(* ---- Set paths from args, falling back to env vars. ---- *)
If[!ValueQ[$FeynRulesPath],
  $FeynRulesPath = Lookup[heptapodParams, "FeynRulesPath",
    With[{env = Environment["FEYNRULES_PATH"]}, If[env === $Failed, Missing["nf"], env]]
  ];
];
If[!ValueQ[$ModelPath],
  $ModelPath = Lookup[heptapodParams, "ModelPath",
    With[{env = Environment["FR_MODEL_PATH"]}, If[env === $Failed, Missing["nf"], env]]
  ];
];
If[!ValueQ[$OutputDir], $OutputDir = Lookup[heptapodParams, "OutputDir", "UFO_Output"]];

$LagrangianSpec = Lookup[heptapodParams, "Lagrangian", None];
$LoadSM         = heptapodBoolParam["LoadSM", True];
(* Tri-state. Unset leaves whatever the model files set, which is what every
   caller before this parameter existed got. *)
$FeynmanGaugeArg = Lookup[heptapodParams, "FeynmanGauge", None];
$ExpandIndices  = heptapodBoolParam["ExpandIndices", True];
(* Which consistency checks to run.  Two of the six are excluded by default:
   CheckDiagonalQuadraticTerms operates on GetQuadraticTerms without the
   scalar-vector DeleteCases that makes CheckDiagonalKineticTerms usable, so it
   reports the R_xi gauge-Goldstone mixing term for any spontaneously broken
   gauge theory; and CheckKineticTermNormalisation expects a +-1 coefficient
   that does not apply to Weyl fermions, so it trips on the SM neutrinos.  Both
   fail for the Standard Model alone and would fail for essentially every model
   built on it.  Pass Checks=all, or a comma-separated list, to run them. *)
$DefaultChecks = {"hermiticity", "kinetic_diagonal", "mass_diagonal",
                  "mass_spectrum"};
$AllChecks     = {"hermiticity", "kinetic_diagonal", "quadratic_diagonal",
                  "mass_diagonal", "kinetic_norm", "mass_spectrum"};

$ChecksRaw = ToString[Lookup[heptapodParams, "Checks", "true"]];
$SelectedChecks = Switch[ToLowerCase[$ChecksRaw],
  "false", {},
  "true",  $DefaultChecks,
  "all",   $AllChecks,
  _,       StringTrim /@ StringSplit[$ChecksRaw, ","]
];
$UnknownChecks  = Complement[$SelectedChecks, $AllChecks];
(* Select, not Intersection: Intersection sorts, and the running order is
   deliberate -- cheapest and most fundamental first. *)
$SelectedChecks = Select[$SelectedChecks, MemberQ[$AllChecks, #] &];
$RunChecks      = Length[$SelectedChecks] > 0;
$ChecksOnly     = heptapodBoolParam["ChecksOnly", False];
$StrictChecks   = heptapodBoolParam["StrictChecks", False];
(* Comparing against an SM-only run is only meaningful if the SM is loaded, and
   only worth its cost -- a second pass of every check -- when running checks
   the SM itself fails.  Off by default now that those are opt-in. *)
$BaselineCompare = heptapodBoolParam["BaselineCompare", False] && $LoadSM;

(* ---- Wolfram version gate. ---- *)
(* FeynRules 2.3.49 (2021) predates Wolfram 14, which made Commutator and
   MatrixSymbol Protected built-ins.  FeynRules redefines both, so on an
   unpatched install under >=14 WriteUFO throws SetDelayed::write and then
   cascades into malformed output rather than failing outright.  Upstream fix:
   FeynRules PR #6.  Under >=15 the 1->2 partial-width routine is broken as
   well (IndexRange[Index[Spin]] no longer iterates), so decays default off
   there and MadGraph computes widths from the UFO instead. *)
$LegacyKernel = $VersionNumber < 14;
$AddDecays    = heptapodBoolParam["AddDecays", $LegacyKernel];

Print["[INFO] Wolfram $VersionNumber = ", $VersionNumber];
If[!$LegacyKernel,
  Print["[WARN] Wolfram >= 14 detected. FeynRules 2.3.49 requires the ",
        "Commutator/MatrixSymbol unprotect patch (FeynRules PR #6) on this ",
        "kernel; without it the UFO export produces malformed output. Use a ",
        "kernel older than 14 (WOLFRAMSCRIPT_KERNELPATH=<its WolframKernel>)."];
  If[$AddDecays,
    Print["[WARN] AddDecays=true on Wolfram >= 14: the partial-width routine ",
          "is unreliable here. Let MadGraph compute widths instead."]];
];

(* ---- Validate inputs (existence only). ---- *)
If[$FeynRulesPath === Missing["nf"] || !DirectoryQ[$FeynRulesPath],
  Print["Error: FeynRulesPath must be a directory. Got: ", $FeynRulesPath]; Quit[1];
];
If[$ModelPath === Missing["nf"] || !FileExistsQ[$ModelPath],
  Print["Error: ModelPath not found: ", $ModelPath]; Quit[1];
];

(* ---- Load FeynRules. ---- *)
(* FeynRules.m reads Global`$FeynRulesPath at load time, so it must already be
   set above; loading it without that silently fails to find its subroutines. *)
heptapodFrm = FileNameJoin[{$FeynRulesPath, "FeynRules.m"}];
If[FileExistsQ[heptapodFrm],
  Print["[INFO] Loading via explicit file: ", heptapodFrm];
  Get[heptapodFrm],
  Print["[INFO] Adding to $Path and using Needs[]. Dir: ", $FeynRulesPath];
  If[!MemberQ[$Path, $FeynRulesPath], AppendTo[$Path, $FeynRulesPath]];
  Needs["FeynRules`"]
];

(* ---- Load the model, with or without the SM. ---- *)
heptapodSmFR = FileNameJoin[{$FeynRulesPath, "Models", "SM", "SM.fr"}];
If[$LoadSM,
  If[FileExistsQ[heptapodSmFR],
    Print["[INFO] Loading SM from: ", heptapodSmFR];
    LoadModel[heptapodSmFR, $ModelPath],
    Print["[INFO] Loading SM from search path + add-on: ", $ModelPath];
    LoadModel["SM.fr", $ModelPath]
  ],
  Print["[INFO] Loading model standalone (LoadSM=false): ", $ModelPath];
  LoadModel[$ModelPath]
];

(* Gauge override, applied AFTER the load and BEFORE the Lagrangian is used.
   Both halves matter, and so does being before `$LTot = ToExpression[...]`
   below: that is a Set, so it evaluates the SetDelayed Lagrangians on the
   spot and reads FeynmanGauge as it does.

   SM.fr assigns `FeynmanGauge = True;` itself, so a value
   set before LoadModel is simply overwritten; and SM.fr's Lagrangians are all
   SetDelayed, reading FeynmanGauge when they are evaluated rather than when
   they are read, so setting it here still reaches them. LHiggs and LYukawa
   drop the Goldstones through their own feynmangaugerules, and LGhost reduces
   to the QCD ghost term, so the whole SM side follows from this one symbol. *)
If[$FeynmanGaugeArg =!= None,
  FeynmanGauge = (ToLowerCase[$FeynmanGaugeArg] === "true");
  Print["[INFO] FeynmanGauge set by caller: ", ToString[FeynmanGauge]]];

(* ---- Lagrangian resolution. ---- *)
(* heptapodHasValue tests a symbol named by a string without evaluating it; the naive
   ValueQ[Symbol[name]] would evaluate the symbol to its value first and then
   test that instead. *)
heptapodHasValue[name_String] := ToExpression[name, InputForm, Function[s, ValueQ[s], HoldFirst]];
heptapodSymContext[name_String] := ToExpression[name, InputForm, Function[s, Context[s], HoldFirst]];

(* Symbols a model defines that plausibly name a Lagrangian. Reported, never
   chosen: picking one silently is the failure this script exists to prevent. *)
heptapodLagrangianCandidates[] := Select[Names["Global`L*"], heptapodHasValue];

If[$LagrangianSpec === None,
  Print["Error: no Lagrangian given. Pass Lagrangian=<expression>, e.g. ",
        "Lagrangian=\"LSM + LBSM\" or Lagrangian=\"LNP\"."];
  Print["[INFO] Lagrangian-like symbols defined by this model: ",
        StringRiffle[heptapodLagrangianCandidates[], ", "]];
  Quit[1];
];

(* Preflight: every Global` symbol named in the expression must be defined.
   Built-ins are skipped by context so Lagrangian="Expand[LSM + LBSM]" works. *)
$LagSymbols = DeleteDuplicates @ StringCases[
  $LagrangianSpec, RegularExpression["[A-Za-z$][A-Za-z0-9$]*"]];
$Undefined = Select[$LagSymbols,
  heptapodSymContext[#] === "Global`" && !heptapodHasValue[#] &];

If[Length[$Undefined] > 0,
  Print["Error: undefined symbol(s) in Lagrangian \"", $LagrangianSpec, "\": ",
        StringRiffle[$Undefined, ", "]];
  Print["[INFO] Lagrangian-like symbols defined by this model: ",
        StringRiffle[heptapodLagrangianCandidates[], ", "]];
  Quit[1];
];

(* ExpandIndices BEFORE WriteUFO, and by default.

   WriteUFO given a Lagrangian that still carries unexpanded SU(2) index
   structure emits `P.<>CreateObjectParticleName[PartNameMG[Wi]]` for the
   gauge self-couplings -- Mathematica source where a Python particle name
   belongs. Those vertex blocks are then unparseable, and every tool that
   reads the UFO simply does not see them. The model compiles, reports
   success, and is missing its triple- and quartic-gauge couplings.

   Measured on the published Hidden Abelian Higgs implementation: compiled
   straight, 0 Zp-W-W vertices; compiled as ExpandIndices[LHAHM], 105
   vertices including 4 Zp-W-W. The loss is silent in both the tool's own
   postflight and in MadGraph, which just sees a smaller model.

   FeynmanRules expands Wi correctly on its own, so the defect is in WriteUFO
   re-processing the Lagrangian rather than in the index expansion. Expanding
   first sidesteps it. On a Lagrangian that is already expanded -- which
   SM.fr's own pieces are, since they call ExpandIndices internally -- this is
   redundant rather than harmful, which is why it is safe as a default.

   Set ExpandIndices=false to skip it, for a model where the expansion is too
   expensive or the caller has already done it in their own expression. *)
$LTot = If[$ExpandIndices,
  ExpandIndices[ToExpression[$LagrangianSpec]],
  ToExpression[$LagrangianSpec]];
Print["[INFO] Lagrangian: ", $LagrangianSpec];

(* Which gauge the model is being built in. FeynmanGauge is a global set inside
   SM.fr -- True by default there since FeynRules v1.4.2 -- so an add-on
   inherits Feynman gauge without deciding to, and picks up a Goldstone and
   ghost sector that shows up as tens of extra vertices. One model built this
   way carried 138 gauge-artefact vertices against 122 physical ones and its
   author only discovered the gauge afterwards. Nothing in FeynRules announces
   it, so announce it here. *)
Print["HEPTAPOD-GAUGE|feynman=",
      ToString[TrueQ[FeynmanGauge]],
      "|source=", Which[
        $FeynmanGaugeArg =!= None, "caller (FeynmanGauge argument)",
        $LoadSM, "SM.fr (inherited unless the model overrides it)",
        True, "model file"]];

(* Report what the expression actually resolved to. A caller who writes the
   wrong symbol, or omits one, otherwise ships a model missing a whole sector
   with nothing in any output to show it -- an agent noted that passing a
   total that excluded the Higgs potential would have looked identical to
   passing the right one. Term count is after expansion, so "LSM + LBSM"
   resolving to a handful of terms is visible as wrong. *)
Print["HEPTAPOD-LAGRANGIAN|expression=", $LagrangianSpec,
      "|symbols=", StringRiffle[$LagSymbols, ","],
      "|terms=", ToString[Length[Expand[$LTot]]]];

(* Unphysical fields that survived expansion.

   THE SILENT FAILURE this catches. An add-on re-expresses an SM gauge
   eigenstate by adding a rule to MR$Definitions. If the pattern does not match
   what FeynRules actually stored -- and it often does not, because each rule is
   kept twice, bare (Phi[2]) and index-wrapped (Phi[Index[SU2D,2]]) -- the rule
   is simply never applied. The unphysical field stays in the expanded
   Lagrangian, vertex extraction drops every term containing it, and nothing
   anywhere says so. One model lost its entire |D Phi_SM|^2 term this way,
   along with the W and Z masses and every SM Higgs coupling, and still
   compiled and passed hermiticity. The author spent forty minutes finding it.

   The class list is read through MR$ClassesList / MR$ClassesRules, which is
   the API FeynRules 2.3.49 actually exposes. An earlier version used
   MR$ClassNameList -- a symbol that appears in Core/Loop.m but is never
   defined at top level, so it stayed an inert symbol, Select returned
   unevaluated, ListQ was False, and the whole diagnostic emitted NOTHING while
   looking like it had run. Verified against a live kernel: for the SM the
   unphysical classes are {B, Wi, ghB, ghWi, LL, lR, QL, uR, dR, Phi, theta,
   eps0..eps9}, LSM leaves none of them behind, and LSM + Phibar[1] Phi[1]
   reports {Phi}.

   Wrapped defensively: a FeynRules version that shapes these differently must
   not take a run down over a diagnostic. *)
Module[{unphys, names, residual},
  residual = Quiet[Check[
    (unphys = Select[MR$ClassesList,
       ((Unphysical /. MR$ClassesRules[#]) === True) &];
     names = (ClassName /. MR$ClassesRules[#]) & /@ unphys;
     Select[names, (! FreeQ[$LTot, #]) &]),
    $Failed]];
  If[ListQ[residual],
    WriteString[$Output,
      "HEPTAPOD-RESIDUAL|count=", ToString[Length[residual]],
      "|fields=", StringRiffle[ToString /@ residual, ","], "\n"]]];

(* ---- FeynRules consistency checks. ---- *)
(* Each verdict comes from the check's RETURN VALUE, not its printed prose --
   see tools/feynrules/lagrangian_checks.py for why, and for what each one returns.  The
   prose is captured anyway and passed through as supporting detail. *)

heptapodSanitise[s_String] := StringTake[
  StringReplace[s, {"|" -> "/", "\n" -> " ", "\r" -> " "}], UpTo[400]];

(* HoldRest matters: without it the check would evaluate at the call site,
   outside the Block, and its Print output would escape uncaptured. *)
SetAttributes[heptapodRunCheck, HoldRest];
heptapodRunCheck[scope_String, name_String, verdictFn_, expr_] := Module[
  {captured = {}, value, verdict, detail},
  value = Block[
    {Print = (AppendTo[captured, StringJoin[ToString /@ {##}]] &)},
    expr
  ];
  verdict = verdictFn[value];
  detail  = heptapodSanitise[StringRiffle[captured, " "]];
  WriteString[$Output,
    "HEPTAPOD-CHECK|scope=", scope, "|name=", name,
    "|verdict=", verdict, "|detail=", detail, "\n"];
  verdict
];

(* Dispatch one check by name.  Switch evaluates only the matching branch, and
   heptapodRunCheck holds its expression, so nothing runs that was not asked for. *)
heptapodRunNamedCheck[scope_String, name_String, lag_] := Switch[name,
  "hermiticity",        heptapodRunCheck[scope, name, heptapodVHermiticity,   CheckHermiticity[lag]],
  "kinetic_diagonal",   heptapodRunCheck[scope, name, heptapodVBoolean,       CheckDiagonalKineticTerms[lag]],
  "quadratic_diagonal", heptapodRunCheck[scope, name, heptapodVBoolean,       CheckDiagonalQuadraticTerms[lag]],
  "mass_diagonal",      heptapodRunCheck[scope, name, heptapodVBoolean,       CheckDiagonalMassTerms[lag]],
  "kinetic_norm",       heptapodRunCheck[scope, name, heptapodVNormalisation, CheckKineticTermNormalisation[lag]],
  "mass_spectrum",      With[{v = CheckMassSpectrum[lag]},
                          If[scope === "model", heptapodEmitMassRows[v]];
                          heptapodRunCheck[scope, name, heptapodVSpectrum, v]],
  _,                    "inconclusive"
];

(* Run the selected checks against one Lagrangian, returning name -> verdict. *)
heptapodRunAllChecks[scope_String, lag_, names_List] :=
  Association[# -> heptapodRunNamedCheck[scope, #, lag] & /@ names];

(* CheckHermiticity returns the Feynman rules of L - HC[L]; {} means hermitian. *)
heptapodVHermiticity[v_] := Which[v === {}, "pass", ListQ[v], "fail", True, "inconclusive"];

(* The three diagonality checks return True/False, and Null when there are no
   terms of that kind to examine -- which is undecided, not passing. *)
heptapodVBoolean[v_] := Which[v === True, "pass", v === False, "fail", True, "inconclusive"];

(* CheckKineticTermNormalisation has no else-branch on its final If, so failure
   returns Null.  Null is also what a non-diagonal kinetic sector produces, so
   the two cases are separated in checks.py using the diagonality verdict. *)
heptapodVNormalisation[v_] := If[v === True, "pass", "inconclusive"];

(* CheckMassSpectrum returns a TableForm of
   {Particle, Analytic value, Numerical value, Model-file value} rows, with
   "OK" stripped from the ones that agree, so a surviving StyleForm marks a
   discrepancy.

   The numbers are worth more than the verdict. A model can be hermitian, have
   diagonal kinetic and mass terms and exactly the right interactions, and
   still declare a mass the Lagrangian does not produce -- a sign error in a
   mass-matrix inversion does precisely that. "fail" does not say which state
   or by how much; "125.0 declared against 164.2 derived" does.

   InputForm, not the default. ToString on a machine real renders six
   significant figures, which rounded a true residual of 1.8*10^-8 to a
   reported relative difference of exactly 0.0 -- a check whose whole job is
   numerical comparison discarding the comparison. FeynRules' own tolerance is
   0.1% (ToolBox.m: agreement is model value * 0.999 to * 1.001), so six
   figures is not even enough to see where a state sits inside it. *)
heptapodEmitMassRows[v_] := Module[{rows},
  If[Head[v] =!= TableForm, Return[]];
  rows = First[v];
  If[!ListQ[rows] || Length[rows] < 2, Return[]];
  Do[
    With[{r = rows[[i]]},
      If[ListQ[r] && Length[r] >= 4,
        WriteString[$Output,
          "HEPTAPOD-MASS|particle=", heptapodSanitise[ToString[r[[1]]]],
          "|derived=", heptapodSanitise[ToString[N[r[[3]]], InputForm]],
          "|declared=", heptapodSanitise[ToString[N[r[[4]]], InputForm]],
          "|agrees=", ToString[Length[r] < 5], "\n"]]],
    {i, 2, Length[rows]}];
];
heptapodVSpectrum[v_] := Which[
  Head[v] =!= TableForm, "inconclusive",
  Cases[v, _StyleForm, Infinity] === {}, "pass",
  True, "fail"];

If[Length[$UnknownChecks] > 0,
  Print["[WARN] Unknown check name(s) ignored: ",
        StringRiffle[$UnknownChecks, ", "], ". Known checks: ",
        StringRiffle[$AllChecks, ", "]];
];

$CheckVerdicts = {};
If[$RunChecks,
  (* Several checks fail for the Standard Model on its own, for reasons that
     have nothing to do with the model under test: CheckDiagonalQuadraticTerms
     reports the R_xi gauge-Goldstone mixing term (del[G+,mu] W-[mu]), which
     CheckDiagonalKineticTerms deliberately excludes but this one does not, and
     CheckKineticTermNormalisation warns about the massless neutrinos.  Both
     fire for every SM extension ever written.  Running the same battery over
     LSM alone tells the caller which failures the model actually introduced,
     which is the only version of this question worth answering. *)
  $BaselineVerdicts = If[$BaselineCompare,
    Print["[INFO] Running consistency checks on the SM baseline."];
    heptapodRunAllChecks["baseline", LSM, $SelectedChecks],
    Association[]
  ];

  Print["[INFO] Running consistency checks: ", StringRiffle[$SelectedChecks, ", "]];
  $CheckVerdicts = heptapodRunAllChecks["model", $LTot, $SelectedChecks];

  (* Only failures the model itself introduces are worth stopping for. *)
  $NewFailures = Select[Keys[$CheckVerdicts],
    $CheckVerdicts[#] === "fail" && Lookup[$BaselineVerdicts, #, None] =!= "fail" &];

  If[$StrictChecks && Length[$NewFailures] > 0,
    Print["Error: consistency check(s) failed that the SM baseline passes (",
          StringRiffle[$NewFailures, ", "], ") and StrictChecks=true; ",
          "no UFO was written."];
    Quit[3];
  ];
];

If[$ChecksOnly,
  Print["[INFO] ChecksOnly=true; skipping UFO generation."];
  Print["[INFO] Done."];
  Quit[0];
];

(* ---- Write UFO. ---- *)
Print["[INFO] UFO output: ", $OutputDir, "  (AddDecays -> ", $AddDecays, ")"];
WriteUFO[$LTot, Output -> $OutputDir, AddDecays -> $AddDecays];

(* ---- Postflight: a UFO that lacks these is not usable. ---- *)
(* WriteUFO writes its boilerplate even when vertex extraction produced
   nothing, so "it finished" is not evidence that it worked.  Without this the
   caller sees exit 0 and a directory of empty scaffolding. *)
$Required = {"particles.py", "vertices.py", "couplings.py", "parameters.py",
             "coupling_orders.py"};
$Missing = Select[$Required, !FileExistsQ[FileNameJoin[{$OutputDir, #}]] &];

If[Length[$Missing] > 0,
  Print["Error: UFO is incomplete; missing ", StringRiffle[$Missing, ", "],
        " in ", $OutputDir, ". The Lagrangian may contain no interaction terms, ",
        "or vertex extraction may have failed above."];
  Quit[2];
];

Print["[INFO] UFO complete: ", StringRiffle[$Required, ", "], " all present."];
Print["[INFO] Done."];
Quit[0];
