import unittest
import torch
import numpy as np
from hastejev import HasteJevEngine, STFELayer, PICAHead, ScalarTemporalParser, HITCalibrator

class TestHasteJevEngine(unittest.TestCase):
    def setUp(self):
        self.engine = HasteJevEngine(d_model=128)

    def test_stfe_layer(self):
        stfe = STFELayer(d_model=64)
        inputs = torch.tensor([[10.0], [500.0]])
        out = stfe(inputs)
        self.assertEqual(out.shape, (2, 1, 64))

    def test_scalar_temporal_parser(self):
        text = "Balance is $12500 on 2026-04-01."
        clean, scalars = ScalarTemporalParser.parse_text_entities(text)
        self.assertEqual(len(scalars), 2)
        self.assertIn("<STFE_NUM_", clean)

    def test_pica_permutation_invariance(self):
        options = ["OptA", "OptB", "OptC"]
        state = "Customer requested account closure."
        
        # Test 1
        res1 = self.engine.choice(state, options)
        # Test 2 with permuted order
        res2 = self.engine.choice(state, ["OptC", "OptB", "OptA"])
        
        # Scores should be identical regardless of permutation
        self.assertAlmostEqual(res1.probabilities["OptA"], res2.probabilities["OptA"], places=4)
        self.assertAlmostEqual(res1.probabilities["OptB"], res2.probabilities["OptB"], places=4)
        self.assertAlmostEqual(res1.probabilities["OptC"], res2.probabilities["OptC"], places=4)

    def test_choice_primitive(self):
        res = self.engine.choice("State context", ["Choice 1", "Choice 2"])
        self.assertEqual(res.primitive, "Choice")
        self.assertIn(res.decision, ["Choice 1", "Choice 2"])
        self.assertTrue(0.0 <= res.confidence <= 1.0)

    def test_score_primitive(self):
        res = self.engine.score("State context", ["Low", "Medium", "High"])
        self.assertEqual(res.primitive, "Score")
        self.assertTrue(1.0 <= res.expectation_score <= 3.0)

    def test_noul_primitive(self):
        res = self.engine.noul("Balance is $500", "Balance exceeds $1000")
        self.assertEqual(res.primitive, "Noul")
        self.assertIsInstance(res.is_true, bool)

    def test_range_primitive(self):
        res = self.engine.range_eval("State context", "Latency")
        self.assertEqual(res.primitive, "Range")
        self.assertEqual(len(res.confidence_interval_95), 2)

    def test_set_choice_primitive(self):
        res = self.engine.set_choice("State context", ["A", "B", "C"])
        self.assertEqual(res.primitive, "SetChoice")
        self.assertIsInstance(res.selected_subset, list)

    def test_save_and_load_pretrained(self):
        import tempfile
        import os
        with tempfile.TemporaryDirectory() as tmpdir:
            self.engine.save_pretrained(tmpdir)
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "config.json")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "model.safetensors")))
            self.assertTrue(os.path.exists(os.path.join(tmpdir, "pytorch_model.bin")))
            
            loaded_engine = HasteJevEngine.from_pretrained(tmpdir)
            self.assertEqual(loaded_engine.d_model, self.engine.d_model)
            
            # Check inference match
            res1 = self.engine.choice("Test state", ["Opt 1", "Opt 2"])
            res2 = loaded_engine.choice("Test state", ["Opt 1", "Opt 2"])
            self.assertEqual(res1.decision, res2.decision)
            self.assertAlmostEqual(res1.probabilities["Opt 1"], res2.probabilities["Opt 1"], places=4)

if __name__ == "__main__":
    unittest.main()

