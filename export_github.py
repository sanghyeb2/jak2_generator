"""Export only the public source allowlist; never data, model weights or SMILES."""
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent


def public_files():
    files = [p for p in HERE.iterdir() if p.is_file() and
             (p.suffix in ('.py', '.md') or p.name in
              ('.gitignore', '.gitattributes', 'run.cmd', 'requirements.txt', 'setup.ps1', 'run_all.ps1', 'code_provenance.json'))]
    files += list((HERE / 'generators').glob('*.py'))
    files += [HERE / 'generators/data/molgpt_settings.json']
    files += [p for p in (HERE / 'docs').iterdir() if p.suffix in ('.md', '.json')]
    files += [HERE / 'results/summary.json', HERE / 'results/assets_manifest.json', HERE / 'inputs/README.md']
    files += list((HERE / 'report').glob('*.docx'))
    return sorted(set(files))


def main():
    output = HERE / 'github_source.zip'
    if output.exists():
        raise FileExistsError('Existing export preserved; move it before exporting again')
    files = public_files()
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for file in files:
            archive.write(file, str(file.relative_to(HERE)))
    print(output)
    print(f'{len(files)} source/document/metadata files; no CSV, weights, cache or training data.')


if __name__ == '__main__':
    main()
