"""
# settings_topics.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.

Curated topic and alias layer over the Pythia settings database.

The xmldoc gives every setting a name, a type and a default, but nothing that
maps the *physics* vocabulary a user actually types onto those names. Searching
the raw manual for "beam energy" ranks `Beams:allowVariableEnergy` above
`Beams:eCM`, because the former has the word in its name and the latter only in
its prose. This module supplies that missing layer for the settings a run card
is actually built out of:

  * ``TOPICS``   -- the sections of a run card, in the order they belong
                    in one.
  * ``ALIASES``  -- the phrases a physicist would use for a given setting.

Only run-card-relevant settings are curated. The other ~1900 remain fully
searchable through the xmldoc-derived name and description index; they simply
do not get synonym support.
"""

from __future__ import annotations

from typing import Dict, List

# Ordered so that `topic` can also drive the section order of a generated card.
TOPICS: Dict[str, str] = {
    "run": "Event count, seeding, and printout verbosity",
    "beams": "What is colliding, at what energy, in what frame",
    "lhe": "Reading Les Houches events from an external matrix-element "
    "generator",
    "process": "Which hard process to generate internally",
    "cuts": "Phase-space cuts on the hard process",
    "pdf": "Parton distribution choice",
    "shower": "Initial- and final-state parton showers",
    "mpi": "Multiparton interactions and the underlying event",
    "hadronization": "String fragmentation and hadron formation",
    "decays": "Particle decays, lifetimes, and stability",
    "tune": "Global tunes that set many parameters at once",
    "matching": "ME/PS matching and merging",
    "bsm": "Beyond-the-Standard-Model process groups",
}

# setting name -> phrases a user might search for. Keep these to genuine
# synonyms; a phrase listed here scores as strongly as the setting's own name.
ALIASES: Dict[str, List[str]] = {
    # ---- run control -------------------------------------------------
    "Main:numberOfEvents": ["number of events", "nevents", "how many events"],
    "Random:seed": ["random seed", "rng seed", "reproducibility"],
    "Random:setSeed": ["enable seed", "fix seed", "use random seed"],
    "Print:quiet": ["silence output", "suppress printout", "quiet mode"],
    "Init:showChangedSettings": ["show changed settings", "echo settings"],
    "Init:showAllSettings": ["dump all settings", "list every setting"],
    "Next:numberCount": ["progress printout", "print every n events"],
    "Next:numberShowEvent": ["print event record", "show event listing"],
    "Stat:showProcessLevel": ["cross section summary", "statistics printout"],
    # ---- beams -------------------------------------------------------
    "Beams:idA": ["first beam particle", "beam a", "projectile"],
    "Beams:idB": ["second beam particle", "beam b", "target"],
    "Beams:eCM": [
        "beam energy",
        "collision energy",
        "centre of mass energy",
        "center of mass energy",
        "cm energy",
        "sqrt s",
        "root s",
        "13 tev",
        "14 tev",
    ],
    "Beams:frameType": [
        "frame",
        "collision frame",
        "fixed target",
        "lhe input mode",
    ],
    "Beams:eA": ["energy of first beam", "asymmetric beam energy"],
    "Beams:eB": ["energy of second beam", "asymmetric beam energy"],
    "Beams:allowMomentumSpread": [
        "beam momentum spread",
        "beam energy smearing",
    ],
    "Beams:allowVertexSpread": [
        "beam spot",
        "vertex smearing",
        "interaction point",
    ],
    # ---- Les Houches / external events -------------------------------
    "Beams:LHEF": [
        "les houches file",
        "lhe file",
        "lhef path",
        "madgraph output",
        "shower lhe",
        "external events",
    ],
    "Beams:readLHEFheaders": ["lhe header", "read lhef headers"],
    "Beams:setProductionScalesFromLHEF": [
        "lhe scales",
        "starting scale from lhe",
    ],
    "LesHouches:setLifetime": [
        "lifetime from lhe",
        "displaced vertex from lhe",
    ],
    # ---- processes ---------------------------------------------------
    "SoftQCD:all": ["soft qcd", "minimum bias", "min bias", "inelastic"],
    "SoftQCD:nonDiffractive": ["non diffractive", "minimum bias"],
    "SoftQCD:inelastic": ["inelastic scattering"],
    "SoftQCD:elastic": ["elastic scattering"],
    "SoftQCD:singleDiffractive": ["single diffraction"],
    "SoftQCD:doubleDiffractive": ["double diffraction"],
    "HardQCD:all": ["hard qcd", "dijet", "jet production", "qcd jets"],
    "HardQCD:gg2gg": ["gluon gluon to gluons"],
    "HardQCD:hardccbar": ["charm production", "ccbar"],
    "HardQCD:hardbbbar": ["bottom production", "bbbar", "b quark production"],
    "Top:all": ["top production", "ttbar", "top quark"],
    "Top:gg2ttbar": ["gluon fusion ttbar", "ttbar production"],
    "Top:qqbar2ttbar": ["quark annihilation ttbar", "ttbar production"],
    "WeakSingleBoson:all": [
        "w production",
        "z production",
        "drell yan",
        "single boson",
    ],
    "WeakSingleBoson:ffbar2gmZ": [
        "drell yan",
        "z boson",
        "dilepton",
        "gamma star z",
    ],
    "WeakSingleBoson:ffbar2W": ["w boson production", "w plus w minus"],
    "WeakDoubleBoson:all": ["diboson", "ww", "wz", "zz"],
    "WeakBosonAndParton:all": [
        "v plus jet",
        "w plus jet",
        "z plus jet",
        "boson plus jet",
    ],
    "PromptPhoton:all": [
        "prompt photon",
        "direct photon",
        "photon production",
    ],
    "HiggsSM:all": ["higgs production", "standard model higgs"],
    "HiggsSM:gg2H": ["gluon fusion higgs", "ggf"],
    "HiggsSM:ff2Hff(t:WW)": ["vector boson fusion", "vbf higgs"],
    "Charmonium:all": ["jpsi", "charmonium", "quarkonium"],
    "Bottomonium:all": ["upsilon", "bottomonium", "quarkonium"],
    # ---- phase space cuts --------------------------------------------
    "PhaseSpace:pTHatMin": [
        "pt cut",
        "minimum pt",
        "pthat cut",
        "hard process pt cut",
        "jet pt threshold",
    ],
    "PhaseSpace:pTHatMax": ["maximum pt", "pt upper cut", "pthat max"],
    "PhaseSpace:mHatMin": [
        "invariant mass cut",
        "minimum mass",
        "mass window low",
    ],
    "PhaseSpace:mHatMax": ["maximum invariant mass", "mass window high"],
    "PhaseSpace:pTHatMinDiverge": ["divergence cut", "regularize low pt"],
    "PhaseSpace:bias2Selection": [
        "bias sampling",
        "enhance high pt",
        "weighted sampling",
    ],
    "PhaseSpace:bias2SelectionPow": ["bias exponent", "sampling power"],
    # ---- pdfs --------------------------------------------------------
    "PDF:pSet": ["pdf set", "parton distribution", "lhapdf set", "proton pdf"],
    "PDF:pHardSet": ["hard process pdf", "separate pdf for hard process"],
    "PDF:useHard": ["separate hard pdf"],
    # ---- showers -----------------------------------------------------
    "PartonLevel:ISR": ["initial state radiation", "isr on off"],
    "PartonLevel:FSR": ["final state radiation", "fsr on off"],
    "PartonLevel:MPI": [
        "multiparton interactions",
        "underlying event",
        "mpi on off",
    ],
    "PartonLevel:Remnants": ["beam remnants"],
    "PartonLevel:all": ["disable parton level", "hard process only"],
    "SpaceShower:alphaSvalue": ["isr alphas", "initial state coupling"],
    "SpaceShower:pTmin": ["isr cutoff", "initial state shower cutoff"],
    "SpaceShower:pTmaxMatch": [
        "isr starting scale",
        "power shower",
        "wimpy shower",
    ],
    "TimeShower:alphaSvalue": ["fsr alphas", "final state coupling"],
    "TimeShower:pTmin": ["fsr cutoff", "final state shower cutoff"],
    "TimeShower:pTmaxMatch": [
        "fsr starting scale",
        "power shower",
        "wimpy shower",
    ],
    "TimeShower:globalRecoil": ["global recoil", "shower recoil scheme"],
    # ---- multiparton interactions ------------------------------------
    "MultipartonInteractions:pT0Ref": [
        "mpi regularization scale",
        "pt0 reference",
    ],
    "MultipartonInteractions:ecmRef": ["mpi reference energy"],
    "MultipartonInteractions:bProfile": [
        "impact parameter profile",
        "overlap profile",
    ],
    # ---- hadronization -----------------------------------------------
    "HadronLevel:all": [
        "turn off hadronization",
        "parton level only",
        "hadronization switch",
    ],
    "HadronLevel:Hadronize": ["hadronization on off", "fragmentation switch"],
    "HadronLevel:Decay": ["particle decays on off", "decay switch"],
    "StringZ:aLund": ["lund a parameter", "fragmentation function a"],
    "StringZ:bLund": ["lund b parameter", "fragmentation function b"],
    "StringPT:sigma": ["fragmentation pt width", "string pt"],
    "StringFlav:probStoUD": ["strangeness suppression", "s quark suppression"],
    "ColourReconnection:reconnect": [
        "colour reconnection",
        "color reconnection",
    ],
    # ---- decays ------------------------------------------------------
    "ParticleDecays:limitTau0": [
        "lifetime cut",
        "limit proper lifetime",
        "long lived cut",
    ],
    "ParticleDecays:tau0Max": ["maximum proper lifetime", "ctau cut"],
    "ParticleDecays:limitRadius": ["decay radius cut", "detector radius"],
    "ParticleDecays:rMax": ["maximum decay radius"],
    "ParticleDecays:FSRinDecays": [
        "radiation in decays",
        "shower decay products",
    ],
    "SLHA:file": ["slha file", "susy spectrum", "param card", "spectrum file"],
    "SLHA:useDecayTable": ["slha decay table", "external decay widths"],
    # ---- tunes -------------------------------------------------------
    "Tune:pp": ["pp tune", "monash", "underlying event tune", "cmc tune"],
    "Tune:ee": ["ee tune", "lep tune", "fragmentation tune"],
    # ---- matching and merging ----------------------------------------
    "Merging:doMerging": ["enable merging", "turn on merging"],
    "Merging:doMGMerging": ["madgraph merging", "mlm style merging"],
    "Merging:doKTMerging": ["ckkw-l merging", "kt merging"],
    "Merging:TMS": ["merging scale", "merging cut"],
    "Merging:Process": ["merging hard process"],
    "JetMatching:doVeto": ["mlm jet matching", "jet veto"],
    "JetMatching:qCut": ["matching scale", "qcut"],
    "JetMatching:nJetMax": ["maximum extra jets"],
    # ---- BSM ---------------------------------------------------------
    "HiddenValley:all": ["hidden valley", "dark shower", "dark sector"],
    "HiddenValley:fragment": [
        "dark hadronization",
        "hidden valley fragmentation",
    ],
    "HiddenValley:ffbar2Zv": ["dark z prime", "zv production"],
    "DM:ffbar2Zp2XX": [
        "dark matter production",
        "mono z prime",
        "dm mediator",
    ],
    "DM:gg2S2XX": ["scalar mediator dark matter"],
    "SUSY:all": ["susy production", "sparticle production"],
    "ExtraDimensionsG*:all": [
        "graviton",
        "randall sundrum",
        "extra dimensions",
    ],
    "LeptoQuark:all": ["leptoquark production"],
    "NewGaugeBoson:ffbar2gmZZprime": ["z prime", "new neutral gauge boson"],
    "NewGaugeBoson:ffbar2Wprime": ["w prime", "new charged gauge boson"],
    "ExcitedFermion:all": [
        "excited fermion",
        "compositeness",
        "excited quark",
    ],
}

# Prefix -> topic. Longest prefix wins, so a specific setting can be assigned a
# different topic than its group by listing its full name.
_TOPIC_PREFIXES: List[tuple] = [
    ("Main:", "run"),
    ("Random:", "run"),
    ("Init:", "run"),
    ("Next:", "run"),
    ("Stat:", "run"),
    ("Print:", "run"),
    ("Parallelism:", "run"),
    ("Beams:LHEF", "lhe"),
    ("Beams:readLHEF", "lhe"),
    ("Beams:setProductionScalesFromLHEF", "lhe"),
    ("Beams:", "beams"),
    ("LesHouches:", "lhe"),
    ("LHEF:", "lhe"),
    ("PhaseSpace:", "cuts"),
    ("PDF:", "pdf"),
    ("PartonLevel:", "shower"),
    ("SpaceShower:", "shower"),
    ("TimeShower:", "shower"),
    ("Vincia:", "shower"),
    ("Dire", "shower"),
    ("SimpleSpaceShower:", "shower"),
    ("SimpleTimeShower:", "shower"),
    ("WeakShower:", "shower"),
    ("UncertaintyBands:", "shower"),
    ("MultipartonInteractions:", "mpi"),
    ("BeamRemnants:", "mpi"),
    ("Diffraction:", "process"),
    ("HadronLevel:", "hadronization"),
    ("StringZ:", "hadronization"),
    ("StringPT:", "hadronization"),
    ("StringFlav:", "hadronization"),
    ("StringFragmentation:", "hadronization"),
    ("HadronScatter:", "hadronization"),
    ("ColourReconnection:", "hadronization"),
    ("Ropewalk:", "hadronization"),
    ("ClosePacking:", "hadronization"),
    ("FragmentationSystems:", "hadronization"),
    ("BoseEinstein:", "hadronization"),
    ("DeuteronProduction:", "hadronization"),
    ("ParticleDecays:", "decays"),
    ("SLHA:", "decays"),
    ("ResonanceWidths:", "decays"),
    ("Tune:", "tune"),
    ("Merging:", "matching"),
    ("JetMatching:", "matching"),
    ("Matching:", "matching"),
    ("SoftQCD:", "process"),
    ("HardQCD:", "process"),
    ("Top:", "process"),
    ("Weak", "process"),
    ("PromptPhoton:", "process"),
    ("Photon", "process"),
    ("Charmonium:", "process"),
    ("Bottomonium:", "process"),
    ("Onia:", "process"),
    ("HiggsSM:", "process"),
    ("Higgs", "bsm"),
    ("SUSY", "bsm"),
    ("HiddenValley:", "bsm"),
    ("DM:", "bsm"),
    ("ExtraDimensions", "bsm"),
    # Pythia's own setting names spell it with three m's.
    ("LeftRightSymmmetry:", "bsm"),
    ("LeptoQuark:", "bsm"),
    ("NewGaugeBoson:", "bsm"),
    ("Compositeness:", "bsm"),
    ("ExcitedFermion:", "bsm"),
    ("Zprime:", "bsm"),
    ("Wprime:", "bsm"),
    ("FourthGeneration:", "bsm"),
    ("ContactInteractions:", "bsm"),
    ("TopThreshold:", "process"),
]


def topic_for(name: str) -> str:
    """Best-effort topic for a setting name; '' when nothing matches."""
    best_topic = ""
    best_length = 0
    for prefix, topic in _TOPIC_PREFIXES:
        if name.startswith(prefix) and len(prefix) > best_length:
            best_topic, best_length = topic, len(prefix)
    return best_topic
