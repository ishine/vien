from pathlib import Path


class CSVLogger:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, s):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(s)
            f.write('\n')

    def log(self, d: dict):
        if not self.log_path.exists():
            self._write(f'{",".join([k for k, v in d.items()])}')
        self._write(f'{",".join([str(v) for k, v in d.items()])}')
