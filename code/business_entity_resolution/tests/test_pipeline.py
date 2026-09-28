"""Fast unit tests for the entity-resolution pipeline (no dataset needed, ~10 s).

Run from code/business_entity_resolution:
    python -m unittest discover -s tests -v

They pin down the behaviour the submission relies on: normalisation, blocking (incl. the rule that the
optional keys_v3 extra candidates never change the normal candidates), the feature matrix shapes,
the competition metric, and the decoders (each Source 2/3 record is matched to at most one Source 1).
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from norm import norm_name, norm_addr, skel, generic_addr_tokens            # noqa: E402
from blocking import Side, candidates, candidates_extra, keys_v3            # noqa: E402
from features import pair_features, F1, F1_V3                               # noqa: E402
from evaluate import f05_macro                                              # noqa: E402
from model import decode, decode_ef_assigned, stage2_matrix                 # noqa: E402
from pipeline import build_country                                          # noqa: E402

CFG = dict(max_block=60, max_s1_block=200, topk=60, addr_stop_frac=0.0, keys_v2=True)


def _df(rows):
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


S1_ROWS = [
    ("S1-1", "Sharma Traders Pvt Ltd", "12 MG Road, Bengaluru, Karnataka", "India"),
    ("S1-2", "Pediatric Dental Atlantic Center Inc", "214 Carson Cub Court, Montgomery, TX", "US"),
    ("S1-3", "Blue Lotus Cafe", "5 Rue Victor Hugo, Lyon", "France"),
    ("S1-4", "Zenith Logistics LLC", "88 Harbor Street, Boston, MA", "US"),
]
OTH_ROWS = [
    ("S2-1", "Sharma Traders Private Limited", "12 M G Road Bangalore", "India"),     # true match of S1-1
    ("S2-2", "Pediatric Dental Atlantic Center Commission", "", "US"),               # true match of S1-2, empty address
    ("S3-1", "Blue Lotus Café SARL", "5 R Victor Hugo Lyon", "France"),               # true match of S1-3 (unseen country)
    ("S3-2", "Totally Different Bakery", "900 Elm Avenue, Denver, CO", "US"),         # matches nothing
]


class TestNormalisation(unittest.TestCase):
    def test_legal_suffixes_and_punctuation(self):
        clean, core = norm_name("Sharma & Sons Pvt. Ltd.")
        self.assertEqual(clean, "sharma and sons pvt ltd")
        self.assertEqual(core, ["sharma", "sons"])        # legal words / connectors removed from the core

    def test_address_abbreviations_states_and_zeros(self):
        clean, toks = norm_addr("12 Main St., Springfield, Illinois 0042")
        self.assertEqual(toks, ["12", "main", "street", "springfield", "il", "42"])

    def test_spelled_ordinals_only_with_flag(self):
        self.assertEqual(norm_addr("thirteenth avenue", ords=True)[1], ["13th", "avenue"])
        self.assertEqual(norm_addr("thirteenth avenue", ords=False)[1], ["thirteenth", "avenue"])

    def test_indic_script_is_transliterated_to_latin(self):
        clean, core = norm_name("शर्मा ट्रेडर्स")
        self.assertTrue(clean.isascii() and clean.startswith("sharma"))

    def test_phonetic_skeleton_absorbs_spelling_variants(self):
        self.assertEqual(skel("technologies"), skel("teknolojies"))

    def test_generic_address_words_are_learned_from_data(self):
        addrs = [f"{i} Main Street Springfield" for i in range(50)] + ["7 Oak Lane Rivertown"]
        stop = generic_addr_tokens(addrs, frac=0.5)
        self.assertIn("street", stop)
        self.assertNotIn("oak", stop)


class TestBlocking(unittest.TestCase):
    def _sides(self, kv3):
        s1 = Side(_df(S1_ROWS), kv2=True, kv3=kv3)
        oth = Side(_df(OTH_ROWS), kv2=True, kv3=kv3)
        return s1, oth

    def _pairs(self, s1, oth, cand):
        return {(s1.ids[a], oth.ids[b]) for a, b in zip(cand.r1, cand.ro)}

    def test_true_matches_become_candidates(self):
        s1, oth = self._sides(kv3=False)
        got = self._pairs(s1, oth, candidates(s1, oth, max_block=60, max_s1_block=200, topk=60))
        self.assertIn(("S1-1", "S2-1"), got)
        self.assertIn(("S1-3", "S3-1"), got)             # unseen country handled like any other

    def test_blocking_never_crosses_countries(self):
        s1, oth = self._sides(kv3=True)
        cand = candidates(s1, oth, max_block=60, max_s1_block=200, topk=60)
        ext = candidates_extra(s1, oth, cand, topk=5)
        c1 = dict(zip(s1.ids, _df(S1_ROWS).country)); co = dict(zip(oth.ids, _df(OTH_ROWS).country))
        for a, b in self._pairs(s1, oth, cand) | self._pairs(s1, oth, ext):
            self.assertEqual(c1[a], co[b])

    def test_keys_v3_off_leaves_normal_candidates_unchanged(self):
        s1a, otha = self._sides(kv3=False)
        s1b, othb = self._sides(kv3=True)
        a = candidates(s1a, otha, max_block=60, max_s1_block=200, topk=60)
        b = candidates(s1b, othb, max_block=60, max_s1_block=200, topk=60)
        pd.testing.assert_frame_equal(a, b)               # extra keys are kept in separate arrays
        self.assertEqual(len(s1a.key3), 0)

    def test_keys_v3_extras_are_new_pairs_only(self):
        s1, oth = self._sides(kv3=True)
        cand = candidates(s1, oth, max_block=60, max_s1_block=200, topk=60)
        ext = candidates_extra(s1, oth, cand, topk=5)
        self.assertFalse(self._pairs(s1, oth, cand) & self._pairs(s1, oth, ext))

    def test_keys_v3_name_word_pairs_work_with_empty_address(self):
        _, core1 = norm_name("Pediatric Dental Atlantic Center Inc")
        _, core2 = norm_name("Pediatric Dental Atlantic Center Commission")
        k1 = {k for k in keys_v3("US", core1, [], frozenset()) if k.startswith("M|")}
        k2 = {k for k in keys_v3("US", core2, [], frozenset()) if k.startswith("M|")}
        self.assertTrue(k1 & k2)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.s1 = Side(_df(S1_ROWS), kv2=True)
        self.oth = Side(_df(OTH_ROWS), kv2=True)
        self.r1 = np.array([0, 0, 2], dtype=np.int32)
        self.ro = np.array([0, 3, 2], dtype=np.int32)
        self.w = np.array([5, 1, 4], dtype=np.float32)

    def test_shapes_and_finite(self):
        X = pair_features(self.s1, self.oth, self.r1, self.ro, self.w)
        self.assertEqual(X.shape, (3, len(F1)))
        self.assertTrue(np.isfinite(X).all())
        self.s1.idf = {"n": {}, "a": {}, "mx": 1.0}
        X3 = pair_features(self.s1, self.oth, self.r1, self.ro, self.w, feat_v3=True)
        self.assertEqual(X3.shape, (3, len(F1_V3)))
        np.testing.assert_array_equal(X3[:, :len(F1)], X)   # extended features only append columns

    def test_true_pair_looks_more_similar_than_random_pair(self):
        X = pair_features(self.s1, self.oth, self.r1, self.ro, self.w)
        nr = F1.index("nr")
        self.assertGreater(X[0, nr], X[1, nr])

    def test_parallel_equals_serial(self):
        r1 = np.tile(self.r1, 20); ro = np.tile(self.ro, 20); w = np.tile(self.w, 20)
        a = pair_features(self.s1, self.oth, r1, ro, w, workers=1)
        b = pair_features(self.s1, self.oth, r1, ro, w, workers=2, chunk=7)
        np.testing.assert_array_equal(a, b)


class TestMetricAndDecoders(unittest.TestCase):
    def test_f05_macro_rules(self):
        truth = {"a": set(), "b": {"x", "y"}, "c": {"z"}}
        self.assertEqual(f05_macro({"a": set(), "b": {"x", "y"}, "c": {"z"}}, truth), 1.0)
        # correct empty = 1, half recall with full precision = 1.25*1*0.5/(0.25+0.5) = 0.8333, wrong = 0
        got = f05_macro({"b": {"x"}, "c": {"q"}}, truth)
        self.assertAlmostEqual(got, (1.0 + 1.25 * 0.5 / 0.75 + 0.0) / 3, places=6)

    def test_each_other_record_goes_to_at_most_one_s1(self):
        r1 = np.array([0, 1, 1, 2]); ro = np.array([0, 0, 1, 1]); p = np.array([0.99, 0.97, 0.99, 0.50])
        rr, oo = decode(r1, ro, p, thr=0.9, margin=0.0)
        self.assertEqual(len(set(oo.tolist())), len(oo))
        self.assertEqual(sorted(zip(rr.tolist(), oo.tolist())), [(0, 0), (1, 1)])

    def test_margin_rejects_ambiguous_records(self):
        r1 = np.array([0, 1]); ro = np.array([0, 0]); p = np.array([0.99, 0.98])
        rr, _ = decode(r1, ro, p, thr=0.9, margin=0.05)
        self.assertEqual(len(rr), 0)

    def test_expected_f05_decoder_prefers_empty_when_unsure(self):
        dec = dict(cal_x=[0.0, 1.0], cal_y=[0.0, 1.0], lam=0.0, beta2=0.25)
        keep_low = decode_ef_assigned(np.array([0, 0]), np.array([0.05, 0.04]), dec)
        keep_high = decode_ef_assigned(np.array([0, 0]), np.array([0.99, 0.98]), dec)
        self.assertFalse(keep_low.any())
        self.assertTrue(keep_high.all())

    def test_stage2_matrix_shape(self):
        r1 = np.array([0, 0, 1]); ro = np.array([0, 1, 1]); p1 = np.array([0.9, 0.2, 0.7], dtype=np.float32)
        raw = np.zeros((3, 11), dtype=np.float32)
        self.assertEqual(stage2_matrix(r1, ro, p1, raw).shape, (3, 19))
        dens = {"addr_by_ro": np.ones(3), "name_by_ro": np.ones(3)}
        self.assertEqual(stage2_matrix(r1, ro, p1, raw, density=dens).shape, (3, 21))


class TestEndToEndBlocking(unittest.TestCase):
    def test_build_country_on_tiny_folder(self):
        with tempfile.TemporaryDirectory() as d:
            _df(S1_ROWS).to_csv(f"{d}/test_source1.tsv", sep="\t", index=False)
            _df(OTH_ROWS[:2]).to_csv(f"{d}/test_source2.tsv", sep="\t", index=False)
            _df(OTH_ROWS[2:]).to_csv(f"{d}/test_source3.tsv", sep="\t", index=False)
            for c in ("India", "US", "France"):
                s1, oth, cand = build_country(d, "test", c, CFG)
                self.assertIsNotNone(cand)
            s1, oth, cand = build_country(d, "test", "France", dict(CFG, keys_v3=True))
            self.assertIn(("S1-3", "S3-1"), {(s1.ids[a], oth.ids[b]) for a, b in zip(cand.r1, cand.ro)})


if __name__ == "__main__":
    unittest.main()
