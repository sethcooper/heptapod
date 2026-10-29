#!/usr/bin/env python3
"""main_template.py -- a minimal, complete Pythia 8 program in Python.

Use a standalone program (rather than a .cmnd card through
PythiaFromRunCardTool) when you need something the card interface cannot
express: UserHooks, per-event analysis inside the generation loop, custom
reweighting, or multiple subruns.

Run:
    python main_template.py [run_card.cmnd]

TWO BINDINGS EXIST, AND THEY ARE NOT INTERCHANGEABLE
----------------------------------------------------
* ``pythia8mc``  -- the pip wheel. This is what heptapod's event_gen bundle
  installs and what PythiaFromRunCardTool imports. Use it.
* ``pythia8``    -- the binding built by Pythia's own ``./configure
  --with-python``. The official examples (main29N.py) import this one, and
  they prepend ``../lib`` to ``sys.path`` to find it.

The API is otherwise the same, with one difference that breaks a direct
translation of any C++ example:

    C++     pythia.info.sigmaGen()
    Python  pythia.infoPython().sigmaGen()

``pythia.info`` does not exist on the Python object at all -- it raises
AttributeError rather than returning something empty.
"""

import sys

try:
    import pythia8mc as pythia8
except ImportError:  # fall back to a source-built binding
    import pythia8


def main(argv):
    # printBanner=False keeps the startup banner out of the output; drop it if
    # you want the citation block.
    pythia = pythia8.Pythia("", printBanner=False)

    # Settings. Either read a card ...
    if len(argv) > 1:
        if not pythia.readFile(argv[1]):
            # readFile returns False on an unknown setting or an out-of-range
            # value, having already printed the offending line. Stopping here
            # is what makes a misconfiguration visible.
            print(f" Aborting: run card {argv[1]} was not accepted.")
            return 1
    # ... or configure inline.
    else:
        for command in (
            "Beams:idA = 2212",
            "Beams:idB = 2212",
            "Beams:eCM = 13600.",
            "HardQCD:all = on",
            "PhaseSpace:pTHatMin = 20.",
        ):
            pythia.readString(command)

    # Reproducibility. Without setSeed the seed value is ignored entirely.
    # Set after the card, this overrides any Random:seed in it.
    pythia.readString("Random:setSeed = on")
    pythia.readString("Random:seed = 12345")

    # init() returns False rather than raising. Ignoring it means the event
    # loop below runs against an uninitialised generator.
    if not pythia.init():
        print(" Aborting: Pythia failed to initialise.")
        return 1

    # Pythia's own histogram class; no external plotting library needed.
    n_charged = pythia8.Hist("charged multiplicity", 100, -0.5, 799.5)
    pt_lead = pythia8.Hist("leading visible particle pT", 100, 0.0, 500.0)

    n_event = 1000
    n_failed = 0

    for _ in range(n_event):
        # next() returns False on a failed event. Individual failures are
        # normal; counting them is how you notice when they stop being rare.
        if not pythia.next():
            n_failed += 1
            continue

        # Every event carries a weight. It is 1 in an unweighted run, but not
        # under cross-section biasing, merging or shower variations, and a
        # histogram filled without it is then wrong in shape.
        weight = pythia.infoPython().weight()

        charged = 0
        pt_max = 0.0
        for particle in pythia.event:
            if not particle.isFinal():  # status > 0
                continue
            if particle.isCharged():
                charged += 1
            if particle.isVisible():  # neutrinos carry pT no detector sees
                pt_max = max(pt_max, particle.pT())
        n_charged.fill(charged, weight)
        pt_lead.fill(pt_max, weight)

    # Cross-section bookkeeping. sigmaGen is in mb; multiply by 1e9 for pb.
    # Normalise a weighted histogram by sigmaGen / weightSum, which reduces to
    # sigmaGen / nAccepted only when every weight is 1. Divide by the bin
    # width as well for a differential cross section.
    pythia.stat()
    info = pythia.infoPython()  # NOT pythia.info
    print(
        f"\n Generated {info.nAccepted()} events, {n_failed} failed."
        f"\n sigma = {info.sigmaGen():.6g} +- {info.sigmaErr():.3g} mb"
    )
    scale = info.sigmaGen() * 1e9 / info.weightSum()  # pb per unit weight
    n_charged *= scale
    pt_lead *= scale
    print(n_charged)
    print(pt_lead)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
