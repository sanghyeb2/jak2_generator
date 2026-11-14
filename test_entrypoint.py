"""Lightweight command/upload boundaries only. No model or CUDA tests."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest

HERE = Path(__file__).resolve().parent


def plan(*arguments):
    result = subprocess.run([sys.executable, '-X', 'utf8', str(HERE / 'main.py'),
                             *arguments, '--dry-run'], capture_output=True, text=True,
                            encoding='utf-8', check=True)
    return json.loads(result.stdout)['tasks']


class EntrypointTests(unittest.TestCase):
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
        self.assertIn('selection_pipeline.py', names)
        self.assertIn('generators/data/molgpt_settings.json', names)
        self.assertFalse(any(Path(n).suffix in ('.csv', '.pt', '.prior', '.safetensors', '.npy', '.npz') for n in names))
        self.assertFalse(any(n.startswith(('local_artifacts/', 'runs/', 'runtime/', 'assets/')) for n in names))


if __name__ == '__main__':
    unittest.main()
