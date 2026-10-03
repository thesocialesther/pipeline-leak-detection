"""Verify float32 export and, when a compiler is available, actual C++ parity."""
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from export_flow_header import ROOT, export, float32
from smart_flow_model import KNN, load, read_dataset


def main():
    cfg, original = load(ROOT/'flow_artifacts/model.json')
    export(ROOT/'flow_artifacts/model.json', ROOT/'embedded/flow_model.h')
    quantized = KNN([[float32(v) for v in row] for row in original.references],
                    original.labels, list(map(float32, original.means)),
                    list(map(float32, original.scales)), original.k)
    dataset = read_dataset(ROOT/'flow_training.csv', cfg)
    vectors = [x for rows in dataset.values() for x, _ in rows]
    expected = [original.predict(x)[0] for x in vectors]
    converted = [quantized.predict(x)[0] for x in vectors]
    mismatches = sum(a != b for a,b in zip(expected, converted))
    report = dict(feature_vectors=len(vectors), float32_prediction_mismatches=mismatches,
                  cpp_compiled=False, cpp_prediction_mismatches=None,
                  esp32_hardware_tested=False)
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler:
        with tempfile.TemporaryDirectory(prefix='flow-parity-') as tmp:
            executable = Path(tmp)/'parity.exe'
            subprocess.run([compiler,'-std=c++11','-O2',str(ROOT/'embedded/parity_check.cpp'),
                            '-o',str(executable)], check=True)
            run = subprocess.run([str(executable)], input='\n'.join(' '.join(map(repr,x)) for x in vectors),
                                 text=True, capture_output=True, check=True)
            actual = list(map(int,run.stdout.split()))
            if len(actual) != len(expected):
                raise RuntimeError('C++ returned the wrong number of predictions')
            report['cpp_compiled'] = True
            report['cpp_prediction_mismatches'] = sum(a != b for a,b in zip(expected,actual))
    else:
        report['cpp_status'] = 'Not run: g++/clang++ not found. Python float32 check is not compiled C++ validation.'
    (ROOT/'embedded/verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if mismatches or report['cpp_prediction_mismatches']:
        raise SystemExit('Prediction mismatch: review before deployment')


if __name__ == '__main__':
    main()
