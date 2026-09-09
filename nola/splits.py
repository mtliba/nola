"""Patient splits for NoLA, revised after review.

WHY THE SPLIT IS DEFINED HERE AND NOWHERE ELSE
    SinoAudit uses the community 8-train / 2-test Mayo split, which is correct
    for its purpose and wrong for ours. Two test patients cannot support a
    patient-level statistic: a bootstrap over two items has three distinct
    resamples, and a Wilcoxon test over their slices is pseudoreplication if
    read as evidence about a patient population. Every interval reported from
    the two-patient split is narrower than the truth.

    The review requires at least four independent evaluation patients. Ten
    prepared patients allow six for adaptation and four for test, which is the
    split defined here.

WHICH FOUR, AND WHY IT IS NOT A CHOICE
    L506 and L333 are the community test patients and stay in test, so the
    numbers remain comparable with everything published from this pipeline.
    The other two are the first two of the remaining patients in sorted order.
    That rule is fixed in advance and mechanical, which matters more than which
    patients it happens to select: any rule chosen after seeing per-patient
    results would be selection on the test set.

STATISTICS THAT FOLLOW FROM THIS
    The patient is the unit. Report the per-patient metric, the mean across
    patients, and a patient-level interval. Slice-level spread may be shown as
    descriptive uncertainty WITHIN a patient and must be labelled as such. Four
    patients is still small; per-patient numbers are reported individually so a
    reader can see the spread rather than trust an interval built from it.
"""
from __future__ import annotations

# The ten Mayo-2016 LDCT Grand Challenge patients.
MAYO_PATIENTS = ("L067", "L096", "L109", "L143", "L192",
                 "L286", "L291", "L310", "L333", "L506")

# Community test patients, kept so numbers stay comparable.
_COMMUNITY_TEST = ("L506", "L333")

MAYO_TEST = _COMMUNITY_TEST + tuple(
    p for p in sorted(MAYO_PATIENTS) if p not in _COMMUNITY_TEST)[:2]
MAYO_ADAPT = tuple(p for p in MAYO_PATIENTS if p not in MAYO_TEST)

# The slice range every method produced for every patient. CoreDiff's sampler
# drops the two volume-edge slices, which fixes the list at 1..58.
EVAL_SLICE_RANGE = range(1, 59)


def eval_slices() -> dict[str, list[int]]:
    return {p: list(EVAL_SLICE_RANGE) for p in MAYO_TEST}


def patient_dir(ready, patient: str):
    """Where a patient's slices actually live on disk.

    THE DIRECTORY LAYOUT IS NOT OUR SPLIT. mayo_ready was prepared under the
    community 8-train / 2-test convention, so L506 and L333 sit in test/ and
    the other eight in train/. Our logical split moves L067 and L096 into the
    test set, and nothing on disk moves with them. Hard-coding
    ``ready / "test" / patient`` therefore fails for exactly the two patients
    the review asked us to add, which is how it failed: every adaptation ran
    and every evaluation died on a missing slice_0001.npz.

    Resolving the directory per patient keeps the logical split independent of
    a layout that predates it.
    """
    from pathlib import Path
    ready = Path(ready)
    for sub in ("test", "train"):
        if (ready / sub / patient).is_dir():
            return ready / sub / patient
    raise FileNotFoundError(f"no prepared slices for {patient} under {ready}")


def check() -> list[str]:
    problems = []
    if set(MAYO_ADAPT) & set(MAYO_TEST):
        problems.append("adaptation and test patients overlap")
    if len(MAYO_TEST) < 4:
        problems.append(f"only {len(MAYO_TEST)} test patients, review requires 4")
    if len(MAYO_ADAPT) + len(MAYO_TEST) != len(MAYO_PATIENTS):
        problems.append("split does not cover all prepared patients")
    return problems


if __name__ == "__main__":
    print(f"adapt ({len(MAYO_ADAPT)}): {', '.join(MAYO_ADAPT)}")
    print(f"test  ({len(MAYO_TEST)}): {', '.join(MAYO_TEST)}")
    print(f"slices per test patient: {len(EVAL_SLICE_RANGE)}")
    print("check:", check() or "clean")
