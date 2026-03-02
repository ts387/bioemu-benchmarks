import os
import shutil
import tempfile
from glob import glob
from pathlib import Path

import mdtraj
import numpy as np
import pytest

from bioemu_benchmarks.benchmarks import Benchmark
from bioemu_benchmarks.samples import (
    IndexedSamples,
    MissingBackbone,
    MissingTopology,
    SequenceSample,
    _find_best_sequence_match,
    filter_unphysical_samples,
    find_samples_in_dir,
    select_relevant_samples,
)

from . import TEST_DATA_DIR

TEST_SAMPLE_DIR = os.path.join(TEST_DATA_DIR, "samples_example")
DEFAULT_MULTICONF_BENCHMARK = Benchmark.MULTICONF_OOD60
DEFAULT_MULTICONF_BENCHMARK_SAMPLE_DIR = os.path.join(TEST_SAMPLE_DIR, "multiconf_ood60")


def test_find_samples_in_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        for system in ["system1", "system2"]:
            xtc = Path(temp_dir) / system / "samples.xtc"
            pdb = Path(temp_dir) / system / "topology.pdb"
            xtc.parent.mkdir(parents=True)
            xtc.touch()
            pdb.touch()
        assert sorted(find_samples_in_dir(temp_dir), key=lambda x: str(x.topology_file)) == [
            SequenceSample(
                Path(temp_dir) / "system1" / "topology.pdb",
                Path(temp_dir) / "system1" / "samples.xtc",
            ),
            SequenceSample(
                Path(temp_dir) / "system2" / "topology.pdb",
                Path(temp_dir) / "system2" / "samples.xtc",
            ),
        ]

    with tempfile.TemporaryDirectory() as temp_dir:
        xtc = Path(temp_dir) / "samples.xtc"
        pdb = Path(temp_dir) / "blah.pdb"
        xtc.touch()
        pdb.touch()
        with pytest.raises(MissingTopology):
            find_samples_in_dir(temp_dir)

    with tempfile.TemporaryDirectory() as temp_dir:
        xtc1 = Path(temp_dir) / "samples1.xtc"
        pdb1 = Path(temp_dir) / "samples1.pdb"
        xtc2 = Path(temp_dir) / "samples2.xtc"
        pdb2 = Path(temp_dir) / "samples2.pdb"
        xtc1.touch()
        pdb1.touch()
        xtc2.touch()
        pdb2.touch()
        assert sorted(find_samples_in_dir(temp_dir), key=lambda x: str(x.topology_file)) == [
            SequenceSample(pdb1, xtc1),
            SequenceSample(pdb2, xtc2),
        ]


def test_select_relevant_samples():
    sequence_samples = find_samples_in_dir(DEFAULT_MULTICONF_BENCHMARK_SAMPLE_DIR)
    assert len(sequence_samples) == 22
    q2fyi5_h = "HMSDVSLKLSAKDIYEKDFEKTMARGYRREEVDAFLDDIIADYQKMADMNNEVVKLSEENHKLKKELEELR"
    q31px7 = "MRIDELVPADPRAVSLYTPYYSQANRRRYLPYALSLYQGSSIEGSRAVEGGAPISFVATWTVTPLPADMTRCHLQFNNDAELTYEILLPNHEFLEYLIDMLMGYQRMQKTDFPGAFYRRLLGYDS"
    filtered, seq_mapping = select_relevant_samples(
        sequence_samples,
        relevant_sequences={q2fyi5_h, q31px7},
    )
    # 3 matches: Q2FYI5_6be25b9e (exact), Q2FYI5_2955f95d (fuzzy: MSDVSLK... -> HMSDVSLK...), Q31PX7 (exact)
    assert len(filtered) == 3
    # Exact matches map to themselves
    assert seq_mapping[q2fyi5_h] == q2fyi5_h
    assert seq_mapping[q31px7] == q31px7
    # Fuzzy match: MSDVSLK... maps to HMSDVSLK...
    q2fyi5_no_h = "MSDVSLKLSAKDIYEKDFEKTMARGYRREEVDAFLDDIIADYQKMADMNNEVVKLSEENHKLKKELEELR"
    assert seq_mapping[q2fyi5_no_h] == q2fyi5_h


def test_find_best_sequence_match():
    benchmark_sequences = {
        "MSDVSLKLSAKDIYEKDFEKTMARGYRREEVDAFLDDIIADYQKMADMNNEVVKLSEENHKLKKELEELR",
        "GPMVSEQSEIVTSKYQKIAVAVAQRIANGDYEVGEKLKSRTTIASTFNVSPETARKGLNILADLQILTLKHGSGAIILSKEKAIEFLNQYETSHSVAILKGKIRDNIKAQQQEMEELATLVDDFLLQTRAVSKQYPLAPYEIIVSEDSEHLGKSIGELNVWHQTGATIVAIEHEGKFIVSPGPFSVIEQGDHIFFVGDEDVYARMKTYFNLRMGL",
    }
    q2fyi5_seq = "MSDVSLKLSAKDIYEKDFEKTMARGYRREEVDAFLDDIIADYQKMADMNNEVVKLSEENHKLKKELEELR"

    # Exact match
    assert _find_best_sequence_match(q2fyi5_seq, benchmark_sequences) == q2fyi5_seq

    # His-tag appended — should still fuzzy-match
    seq_with_histag = q2fyi5_seq + "HHHHHH"
    match = _find_best_sequence_match(seq_with_histag, benchmark_sequences)
    assert match == q2fyi5_seq

    # Extra N-terminal residue — should still fuzzy-match
    seq_with_extra_n = "H" + q2fyi5_seq
    match = _find_best_sequence_match(seq_with_extra_n, benchmark_sequences)
    assert match == q2fyi5_seq

    # Completely unrelated sequence — should return None
    unrelated = "ACDEFGHIKLMNPQRSTVWY" * 5
    assert _find_best_sequence_match(unrelated, benchmark_sequences) is None

    # Very short sequence — should not match
    assert _find_best_sequence_match("ACDE", benchmark_sequences) is None


def test_select_relevant_samples_fuzzy():
    """Test that fuzzy matching works with slightly different sequences."""
    sequence_samples = find_samples_in_dir(DEFAULT_MULTICONF_BENCHMARK_SAMPLE_DIR)
    # Get a sample's actual sequence
    sample = sequence_samples[0]
    actual_seq = mdtraj.load_topology(sample.topology_file).to_fasta()[0]

    # Create a modified version of the sequence (append His-tag)
    modified_seq = actual_seq + "HHHHHH"

    # The modified sequence should fuzzy-match the actual one
    filtered, seq_mapping = select_relevant_samples(
        sequence_samples,
        relevant_sequences={modified_seq},
    )
    # Should not match — modified_seq is the *benchmark* sequence, actual_seq is the sample.
    # The sample has actual_seq which should fuzzy-match modified_seq as benchmark.
    assert len(filtered) >= 1
    assert actual_seq in seq_mapping
    assert seq_mapping[actual_seq] == modified_seq


def test_checks():
    with tempfile.TemporaryDirectory() as temp_dir:
        shutil.copytree(DEFAULT_MULTICONF_BENCHMARK_SAMPLE_DIR, temp_dir, dirs_exist_ok=True)
        xtc_files = glob(os.path.join(temp_dir, "*.xtc"))
        chosen_xtc = Path(xtc_files[0])
        chosen_pdb = chosen_xtc.with_suffix(".pdb")
        traj: mdtraj.Trajectory = mdtraj.load(chosen_xtc, top=chosen_pdb)

        # Check we raise an exception if a topology file is missing.
        os.unlink(chosen_pdb)
        with pytest.raises(MissingTopology):
            sequence_samples = find_samples_in_dir(temp_dir)
            _ = IndexedSamples.from_benchmark(
                benchmark=DEFAULT_MULTICONF_BENCHMARK, sequence_samples=sequence_samples
            )

        # Warn if sequences are missing for a given benchmark.
        os.unlink(chosen_xtc)
        sequence_samples = find_samples_in_dir(temp_dir)
        _indexed_samples = IndexedSamples.from_benchmark(
            benchmark=DEFAULT_MULTICONF_BENCHMARK, sequence_samples=sequence_samples
        )

        # Check we raise an exception if backbone atoms are missing.
        traj.save_xtc(chosen_xtc)
        traj[0].save_pdb(chosen_pdb)

        nbackbone_sel = traj.topology.select("name N")
        nbackbone_traj = traj.atom_slice(nbackbone_sel)
        nbackbone_traj.save_xtc(chosen_xtc)
        nbackbone_traj[0].save_pdb(chosen_pdb)

        with pytest.raises(MissingBackbone):
            sequence_samples = find_samples_in_dir(temp_dir)
            _indexed_samples = IndexedSamples.from_benchmark(
                benchmark=DEFAULT_MULTICONF_BENCHMARK, sequence_samples=sequence_samples
            )


def test_return_traj():
    """
    Checks that we can successfully return expected trajectories for
    a given benchmark.
    """
    sequence_samples = find_samples_in_dir(DEFAULT_MULTICONF_BENCHMARK_SAMPLE_DIR)
    indexed_samples = IndexedSamples.from_benchmark(
        benchmark=DEFAULT_MULTICONF_BENCHMARK, sequence_samples=sequence_samples
    )

    trajs = indexed_samples.get_trajs_for_test_case("Q2FYI5")
    assert len(trajs) == 2
    assert isinstance(trajs[0], mdtraj.Trajectory)

    all_trajs = indexed_samples.get_all_trajs()
    assert len(all_trajs) == 19


def test_filter():
    sequence_samples = find_samples_in_dir(DEFAULT_MULTICONF_BENCHMARK_SAMPLE_DIR)
    indexed_samples = IndexedSamples.from_benchmark(
        benchmark=DEFAULT_MULTICONF_BENCHMARK, sequence_samples=sequence_samples
    )
    with pytest.raises(AssertionError):
        # E1C7U0 is totally filtered out
        _, _ = filter_unphysical_samples(indexed_samples)

    indexed_samples.test_case_to_sequencesamples.pop("E1C7U0")
    _, kept_percs = filter_unphysical_samples(indexed_samples)

    assert np.allclose(np.sort(kept_percs["A0A377JKY9"]), np.array([0.4, 0.8]))
