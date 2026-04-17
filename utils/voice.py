import shutil
import subprocess
from typing import Optional


class VoiceAnnouncer:
    def __init__(self, enabled: bool = False, voice_name: Optional[str] = None):
        self.enabled = enabled
        self.voice_name = voice_name
        self._say_cmd = shutil.which('say')

    def announce_rep(self, reps: int):
        if not self.enabled or reps <= 0 or self._say_cmd is None:
            return
        cmd = [self._say_cmd]
        if self.voice_name:
            cmd.extend(['-v', self.voice_name])
        cmd.append(f'Rep {reps}')
        try:
            subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError:
            pass
