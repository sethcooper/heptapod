#!/usr/bin/env wolframscript
(*
   FeynArts_generator.wl is a part of the HEPTAPOD package.
   Copyright (C) 2026 HEPTAPOD authors (see AUTHORS for details).
   HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
   Please respect the MCnet Guidelines, see GUIDELINES for details.
*)

(* ============ FeynRules model -> FeynArts model ========================= *)
(*
   Driver behind WriteFeynArtsModel.

   Usage:
     wolframscript -f FeynArts_generator.wl \
        ModelPath=<PATH_TO_MODEL.fr> FeynRulesPath=<PATH_TO_FeynRules> \
        OutputDir=<DIR> ModelName=<NAME> Lagrangian="LSM + LBSM" \
        [LoadSM=true|false] [FeynmanGauge=true|false] [ExpandIndices=true|false] \
        [UFOCheck=true|false] [RenameCouplings=true|false]

   Writes <OutputDir>/<NAME>/<NAME>.mod, .gen and .pars (FeynRules'
   WriteFeynArtsOutput), and with UFOCheck=true also a UFO of the same
   Lagrangian in <OutputDir>/<NAME>_UFO, which the linter compares vertex by
   vertex. Loading, the gauge override and the Lagrangian are handled exactly
   as in UFO_generator.wl, whose comments explain why.

   Prints, for the caller to parse:
     HEPTAPOD-GAUGE|feynman=True|source=...
     HEPTAPOD-LAGRANGIAN|expression=...|symbols=...|terms=N
     HEPTAPOD-FACLASS|field=F[2]|class=l|names=e-,mu-,ta-|antinames=e+,mu+,ta+|
                     pdg=11,13,15|ghost=|goldstone=|indices=Generation|selfconjugate=False
     HEPTAPOD-FAOUT|dir=...|mod=...|gen=...
     HEPTAPOD-UFOOUT|dir=...

   Exit codes: 0 success; 1 bad invocation or undefined Lagrangian symbol;
   2 WriteFeynArtsOutput produced no model file.
*)
(* ======================================================================== *)

ClearAll["Global`*"];

heptapodParams = Association @ Map[
    Rule @@ StringSplit[#, "=", 2] &,
    Rest[$ScriptCommandLine]
];
heptapodBoolParam[key_String, default_] :=
  Switch[ToLowerCase[ToString[Lookup[heptapodParams, key, default]]],
    "true", True, "false", False, _, default];

$FeynRulesPath  = Lookup[heptapodParams, "FeynRulesPath",
  With[{env = Environment["FEYNRULES_PATH"]}, If[env === $Failed, Missing["nf"], env]]];
$ModelPath      = Lookup[heptapodParams, "ModelPath", Missing["nf"]];
$OutputDir      = Lookup[heptapodParams, "OutputDir", "FeynArts_Output"];
$ModelName      = Lookup[heptapodParams, "ModelName", "Model_FA"];
$LagrangianSpec = Lookup[heptapodParams, "Lagrangian", None];
$LoadSM         = heptapodBoolParam["LoadSM", True];
$FeynmanGaugeArg = Lookup[heptapodParams, "FeynmanGauge", None];
(* Off by default, unlike for the UFO: WriteFeynArtsOutput wants the
   Lagrangian in classes, and ExpandIndices writes out class members, which
   it refuses ("Class members in the Lagrangian! Not supported by FeynArts").
   The UFO for the vertex check is written from the same expression. *)
$ExpandIndices  = heptapodBoolParam["ExpandIndices", False];
$UFOCheck       = heptapodBoolParam["UFOCheck", True];
(* FeynRules names every coupling gc1, gc2, ... by default; off here, so
   amplitudes come out in the model's own parameters (ee, cw, yt, ...). *)
$RenameCouplings = heptapodBoolParam["RenameCouplings", False];

If[$FeynRulesPath === Missing["nf"] || !DirectoryQ[$FeynRulesPath],
  Print["Error: FeynRulesPath must be a directory. Got: ", $FeynRulesPath]; Quit[1]];
If[$ModelPath === Missing["nf"] || !FileExistsQ[$ModelPath],
  Print["Error: ModelPath not found: ", $ModelPath]; Quit[1]];
If[!StringMatchQ[$ModelName, RegularExpression["[A-Za-z][A-Za-z0-9_]*"]],
  Print["Error: ModelName must be letters, digits and _, starting with a letter: ", $ModelName]; Quit[1]];

Get[FileNameJoin[{$FeynRulesPath, "FeynRules.m"}]];

heptapodSmFR = FileNameJoin[{$FeynRulesPath, "Models", "SM", "SM.fr"}];
If[$LoadSM, LoadModel[heptapodSmFR, $ModelPath], LoadModel[$ModelPath]];

If[$FeynmanGaugeArg =!= None,
  FeynmanGauge = (ToLowerCase[$FeynmanGaugeArg] === "true")];

heptapodHasValue[name_String] := ToExpression[name, InputForm, Function[s, ValueQ[s], HoldFirst]];
heptapodSymContext[name_String] := ToExpression[name, InputForm, Function[s, Context[s], HoldFirst]];
heptapodLagrangianCandidates[] := Select[Names["Global`L*"], heptapodHasValue];

If[$LagrangianSpec === None,
  Print["Error: no Lagrangian given. Pass Lagrangian=<expression>, e.g. Lagrangian=\"LSM + LBSM\"."];
  Print["[INFO] Lagrangian-like symbols defined by this model: ", StringRiffle[heptapodLagrangianCandidates[], ", "]];
  Quit[1]];
$LagSymbols = DeleteDuplicates @ StringCases[$LagrangianSpec, RegularExpression["[A-Za-z$][A-Za-z0-9$]*"]];
$Undefined = Select[$LagSymbols, heptapodSymContext[#] === "Global`" && !heptapodHasValue[#] &];
If[Length[$Undefined] > 0,
  Print["Error: undefined symbol(s) in Lagrangian \"", $LagrangianSpec, "\": ", StringRiffle[$Undefined, ", "]];
  Print["[INFO] Lagrangian-like symbols defined by this model: ", StringRiffle[heptapodLagrangianCandidates[], ", "]];
  Quit[1]];

$LTot = If[$ExpandIndices, ExpandIndices[ToExpression[$LagrangianSpec]], ToExpression[$LagrangianSpec]];

Print["HEPTAPOD-GAUGE|feynman=", ToString[TrueQ[FeynmanGauge]], "|source=", Which[
  $FeynmanGaugeArg =!= None, "caller (FeynmanGauge argument)",
  $LoadSM, "SM.fr (inherited unless the model overrides it)", True, "model file"]];
Print["HEPTAPOD-LAGRANGIAN|expression=", $LagrangianSpec, "|symbols=", StringRiffle[$LagSymbols, ","],
      "|terms=", ToString[Length[Expand[$LTot]]]];

(* ---- The UFO of the same Lagrangian, for the vertex-by-vertex check. ---- *)
(* Written first: after WriteFeynArtsOutput, FeynRules' state is the FeynArts
   interface's, and WriteUFO then complains (Inner::incom) about generation
   indices it would otherwise handle. *)
If[$UFOCheck,
  Print["HEPTAPOD-STAGE|ufo"];
  $UFODir = FileNameJoin[{$OutputDir, $ModelName <> "_UFO"}];
  (* WriteUFO wants indices expanded (see UFO_generator.wl), FeynArts the
     opposite; each gets the form it needs of the same Lagrangian. *)
  WriteUFO[If[$ExpandIndices, $LTot, ExpandIndices[$LTot]], Output -> $UFODir, AddDecays -> False];
  Print["HEPTAPOD-UFOOUT|dir=", $UFODir]];

(* ---- The FeynArts model. ---- *)
Print["HEPTAPOD-STAGE|feynarts"];
(* WriteFeynArtsOutput writes into a directory named after Output, relative to
   the current directory, and changes into it; so go to OutputDir first and
   come back after. *)
If[!DirectoryQ[$OutputDir], CreateDirectory[$OutputDir, CreateIntermediateDirectories -> True]];
$Here = Directory[];
SetDirectory[$OutputDir];
WriteFeynArtsOutput[$LTot, Output -> $ModelName, CouplingRename -> $RenameCouplings];
SetDirectory[$OutputDir];
$ModDir = FileNameJoin[{$OutputDir, $ModelName}];
$Mod = FileNameJoin[{$ModDir, $ModelName <> ".mod"}];
$Gen = FileNameJoin[{$ModDir, $ModelName <> ".gen"}];
If[!FileExistsQ[$Mod],
  Print["Error: WriteFeynArtsOutput wrote no ", $Mod, ". The Lagrangian may have no interaction terms."];
  Quit[2]];
Print["HEPTAPOD-FAOUT|dir=", $ModDir, "|mod=", $Mod, "|gen=", If[FileExistsQ[$Gen], $Gen, ""]];

(* ---- What each FeynArts class is, in UFO terms. ---- *)
(* MR$ClassesDescription is keyed by the FeynArts field itself (F[2] -> {...}),
   with the names WriteUFO uses for the members. Ghosts carry no ParticleName:
   WriteUFO names them by ClassName, antiparticle ClassName~. *)
(* Local names are heptapod-prefixed: a model's own symbols are Global` too,
   and the SM's down quark is literally d -- an iterator called d makes
   MR$ClassesDescription refer to itself. *)
heptapodStrList[x_List] := StringRiffle[ToString /@ x, ","];
heptapodStrList[x_] := ToString[x];
Do[
  With[{f = heptapodClass[[1]], r = heptapodClass[[2]]},
    If[!TrueQ[Unphysical /. r] && MatchQ[f, (F | S | V | U | T)[_Integer]],
      Module[{cls = ClassName /. r, names, anti, sc = TrueQ[SelfConjugate /. r]},
        names = ParticleName /. r /. ParticleName -> cls;
        anti = AntiParticleName /. r /. AntiParticleName :>
          If[sc, names, If[ListQ[names], # <> "~" & /@ (ToString /@ names), ToString[names] <> "~"]];
        Print["HEPTAPOD-FACLASS|field=", ToString[f, InputForm], "|class=", ToString[cls],
          "|names=", heptapodStrList[names], "|antinames=", heptapodStrList[anti],
          "|pdg=", heptapodStrList[PDG /. r /. PDG -> ""],
          "|ghost=", ToString[Ghost /. r /. Ghost -> ""],
          "|goldstone=", ToString[Goldstone /. r /. Goldstone -> ""],
          "|indices=", heptapodStrList[(Indices /. r /. Indices -> {}) /. Index[i_] :> i],
          "|selfconjugate=", ToString[sc]]]]],
  {heptapodClass, MR$ClassesDescription}];

(* ---- Each particle on its own, for models FeynRules flavour-expands. ---- *)
(* A Lagrangian written with class members (one flavour of a leptoquark
   coupling, say) makes WriteFeynArtsOutput redefine the classes, one particle
   each, and record the new ones in FR$ClassesTranslation in the .mod
   (e -> F[4], ...). The class lines above then no longer describe the file,
   so every member is printed by name too. *)
heptapodAsList[x_List] := x;
heptapodAsList[x_] := {x};
Do[
  With[{f = heptapodClass[[1]], r = heptapodClass[[2]]},
    If[!TrueQ[Unphysical /. r] && MatchQ[f, (F | S | V | U | T)[_Integer]],
      Module[{cls = ClassName /. r, mem, names, anti, pdg, sc = TrueQ[SelfConjugate /. r]},
        mem = heptapodAsList[ClassMembers /. r /. ClassMembers -> cls];
        names = heptapodAsList[ParticleName /. r /. ParticleName -> cls];
        anti = heptapodAsList[AntiParticleName /. r /. AntiParticleName :>
          If[sc, names, (ToString[#] <> "~") & /@ names]];
        pdg = heptapodAsList[PDG /. r /. PDG -> ""];
        Do[Print["HEPTAPOD-FRMEMBER|name=", ToString[mem[[k]]], "|particle=", ToString[names[[Min[k, Length[names]]]]],
            "|anti=", ToString[anti[[Min[k, Length[anti]]]]], "|pdg=", ToString[pdg[[Min[k, Length[pdg]]]]],
            "|class=", ToString[cls], "|ghost=", ToString[Ghost /. r /. Ghost -> ""],
            "|goldstone=", ToString[Goldstone /. r /. Goldstone -> ""], "|selfconjugate=", ToString[sc]],
          {k, Length[mem]}]]]],
  {heptapodClass, MR$ClassesDescription}];

Print["[INFO] Done."];
Quit[0];
