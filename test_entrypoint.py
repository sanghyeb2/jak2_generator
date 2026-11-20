"""Lightweight command/upload boundaries only. No model or CUDA tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import ast
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import unittest

HERE = Path(__file__).resolve().parent


def plan(*arguments):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(HERE / 'main.py'),
                             *arguments, '--dry-run'], capture_output=True, text=True,
                            encoding='utf-8', check=True)
    return json.loads(result.stdout)['tasks']


class EntrypointTests(unittest.TestCase):
    def test_reinvent_training_writes_checkpoint_where_generation_reads_it(self):
        # Run the real orchestration with a lightweight stand-in for expensive fit.
        tree = ast.parse((HERE / 'generators/generate.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'train')
        namespace = {'ROOT': HERE / 'generators', 'sys': sys}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<train>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            work = Path(directory)
            output = work / 'training/reinvent'
            output.mkdir(parents=True)
            (output / 'encoded.pt').touch()
            fake = SimpleNamespace()
            def fit(lr, full=False, epochs=30, output=None):
                folder = Path(output) if output is not None else fake.WORK / 'full_6029'
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'last.pt').write_bytes(b'checkpoint')
            fake.fit = fit
            with patch.dict(sys.modules, {'reinvent': fake}):
                namespace['train']('reinvent', work)
            self.assertTrue((output / 'last.pt').is_file(), 'Missing checkpoint for generation')

    def test_plain_main_plans_all_five_trainings_before_five_generations(self):
        tasks = plan()
        self.assertEqual([x['stage'] for x in tasks],
                         [f'{action}_{model}' for action in ('train', 'generate')
                          for model in ('gp', 'gru', 'transfer', 'molgpt', 'reinvent')] + ['selection'])
        for task in tasks[5:10]:
            self.assertIn('--use-trained', task['command'])
        self.assertEqual(Path(tasks[-1]['command'][1]).name, 'selection_pipeline.py')
        self.assertIn('--stage', tasks[-1]['command'])
        self.assertIn('all', tasks[-1]['command'])

    def test_newly_generated_pools_are_the_default_full_pipeline_input(self):
        command = plan()[-1]['command']
        candidates = Path(command[command.index('--candidate-dir') + 1])
        self.assertEqual(candidates, HERE / 'runs/standalone_seed42/generation/candidates')

    def test_selector_only_is_explicit_and_can_use_preserved_inputs(self):
        tasks = plan('--stage', 'select', '--run-name', 'readonly_plan')
        self.assertEqual(len(tasks), 1)
        command = tasks[0]['command']
        self.assertEqual(Path(command[command.index('--candidate-dir') + 1]), HERE / 'inputs/candidates')

    def test_public_export_excludes_data_checkpoints_and_previous_runs(self):
        spec = importlib.util.spec_from_file_location('public_export', HERE / 'export_github.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        names = [p.relative_to(HERE).as_posix() for p in module.public_files()]
        self.assertIn('main.py', names)
        self.assertIn('run_all.ps1', names)
        self.assertIn('preflight.py', names)
        self.assertIn('report/Kaggle_competition_201112_이상협_최종.docx', names)
        self.assertIn('selection_pipeline.py', names)
        self.assertIn('generators/data/molgpt_settings.json', names)
        self.assertFalse(any(Path(n).suffix in ('.csv', '.pt', '.prior', '.safetensors', '.npy', '.npz') for n in names))
        self.assertFalse(any(n.startswith(('local_artifacts/', 'runs/', 'runtime/', 'assets/')) for n in names))


if __name__ == '__main__':
    unittest.main()
